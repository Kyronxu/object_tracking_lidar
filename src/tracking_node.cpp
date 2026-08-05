#include "point_cloud_filter.hpp"
#include "clusterer.hpp"
#include "data_associator.hpp"
#include "object_tracker.hpp"

#include <chrono>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl_conversions/pcl_conversions.h>

namespace object_tracking_lidar
{

// ============================================================================
// Main ROS2 node: composes filter, clusterer, associator, and tracker via polymorphism
// ============================================================================
class ObjectTrackingLidarNode : public rclcpp::Node
{
public:
  ObjectTrackingLidarNode() : Node("object_tracking_lidar")
  {
    declareParameters();
    loadParameters();
    initComponents();
    initPubSub();

    RCLCPP_INFO(this->get_logger(),
      "ObjectTrackingLidar node initialized | sub: %s | max_objects: %d | "
      "filter: %s | clusterer: %s | associator: %s | tracker: %s | "
      "range=[x:%.1f~%.1f, y:%.1f~%.1f, z:%.1f~%.1f] | voxel=[%.3f, %.3f, %.3f]",
      input_topic_.c_str(), max_objects_,
      filter_->name().c_str(), clusterer_->name().c_str(),
      associator_->name().c_str(), tracker_name_.c_str(),
      range_params_.min_x, range_params_.max_x,
      range_params_.min_y, range_params_.max_y,
      range_params_.min_z, range_params_.max_z,
      voxel_params_.voxel_size_x, voxel_params_.voxel_size_y, voxel_params_.voxel_size_z);
  }

private:
  // ---- Parameter declaration ----
  void declareParameters()
  {
    this->declare_parameter<int>("max_tracking_objects", 4);
    this->declare_parameter<double>("cluster_tolerance", 0.4);
    this->declare_parameter<int>("min_cluster_size", 100);
    this->declare_parameter<int>("max_cluster_size", 2500);
    this->declare_parameter<double>("kf_process_noise", 0.01);
    this->declare_parameter<double>("kf_measurement_noise", 0.1);
    this->declare_parameter<std::string>("input_topic", "/lidar/points_raw");
    this->declare_parameter<std::string>("output_frame", "base_link");

    // Range filter parameters
    this->declare_parameter<double>("range_min_x", -3.6);
    this->declare_parameter<double>("range_max_x", 3.6);
    this->declare_parameter<double>("range_min_y", -2.8);
    this->declare_parameter<double>("range_max_y", 2.8);
    this->declare_parameter<double>("range_min_z", -0.1);
    this->declare_parameter<double>("range_max_z", 2.5);

    // Voxel filter parameters
    this->declare_parameter<double>("voxel_size_x", 0.1);
    this->declare_parameter<double>("voxel_size_y", 0.1);
    this->declare_parameter<double>("voxel_size_z", 0.1);
  }

  // ---- Parameter loading ----
  void loadParameters()
  {
    max_objects_ = this->get_parameter("max_tracking_objects").as_int();
    input_topic_ = this->get_parameter("input_topic").as_string();
    output_frame_ = this->get_parameter("output_frame").as_string();

    // Range filter params
    range_params_.min_x = this->get_parameter("range_min_x").as_double();
    range_params_.max_x = this->get_parameter("range_max_x").as_double();
    range_params_.min_y = this->get_parameter("range_min_y").as_double();
    range_params_.max_y = this->get_parameter("range_max_y").as_double();
    range_params_.min_z = this->get_parameter("range_min_z").as_double();
    range_params_.max_z = this->get_parameter("range_max_z").as_double();

    // Voxel filter params
    voxel_params_.voxel_size_x = this->get_parameter("voxel_size_x").as_double();
    voxel_params_.voxel_size_y = this->get_parameter("voxel_size_y").as_double();
    voxel_params_.voxel_size_z = this->get_parameter("voxel_size_z").as_double();

    // Clusterer params
    cluster_params_.cluster_tolerance = this->get_parameter("cluster_tolerance").as_double();
    cluster_params_.min_cluster_size = this->get_parameter("min_cluster_size").as_int();
    cluster_params_.max_cluster_size = this->get_parameter("max_cluster_size").as_int();

    // Kalman tracker params
    kf_params_.process_noise = static_cast<float>(this->get_parameter("kf_process_noise").as_double());
    kf_params_.measurement_noise = static_cast<float>(this->get_parameter("kf_measurement_noise").as_double());
  }

  // ---- Initialize polymorphic components ----
  void initComponents()
  {
    // Build composite filter pipeline: RangeFilter -> VoxelFilter
    auto composite = std::make_unique<CompositeFilter>();
    composite->addFilter(std::make_unique<RangeFilter>(range_params_));
    composite->addFilter(std::make_unique<VoxelFilter>(voxel_params_));
    filter_ = std::move(composite);

    // Clusterer (swap EuclideanClusterer for any other Clusterer subclass)
    clusterer_ = std::make_unique<EuclideanClusterer>(cluster_params_);

    // Data associator (swap GreedyAssociator for any other DataAssociator subclass)
    associator_ = std::make_unique<GreedyAssociator>(max_objects_);

    // Object trackers (swap KalmanObjectTracker for any other ObjectTracker subclass)
    trackers_.clear();
    for (int i = 0; i < max_objects_; i++) {
      trackers_.push_back(std::make_unique<KalmanObjectTracker>(i, kf_params_));
    }

    tracker_name_ = "KalmanObjectTracker";
    first_frame_ = true;
  }

  // ---- Initialize publishers and subscribers ----
  void initPubSub()
  {
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ObjectTrackingLidarNode::cloudCallback, this, std::placeholders::_1));

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/tracking/markers", 10);
    obj_id_pub_ = this->create_publisher<std_msgs::msg::Int32MultiArray>("/tracking/object_ids", 10);

    cluster_pubs_.resize(max_objects_);
    for (int i = 0; i < max_objects_; i++) {
      cluster_pubs_[i] = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/tracking/cluster_" + std::to_string(i), 10);
    }
  }

  // ---- Publish point cloud helper ----
  void publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.frame_id = output_frame_;
    msg.header.stamp = this->now();
    pub->publish(msg);
  }

  // ---- Publish visualization markers ----
  void publishMarkers(const std::vector<geometry_msgs::msg::Point> & positions)
  {
    visualization_msgs::msg::MarkerArray marker_array;
    for (int i = 0; i < max_objects_; i++) {
      visualization_msgs::msg::Marker m;
      m.id = i;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.header.frame_id = output_frame_;
      m.header.stamp = this->now();
      m.scale.x = 0.2;
      m.scale.y = 0.2;
      m.scale.z = 0.2;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.color.a = 1.0;
      m.color.r = i % 2 ? 1 : 0;
      m.color.g = i % 3 ? 1 : 0;
      m.color.b = i % 4 ? 1 : 0;
      m.pose.position = positions[i];
      marker_array.markers.push_back(m);
    }
    marker_pub_->publish(marker_array);
  }

  // ---- Core callback: pipeline = range_filter -> voxel_filter -> cluster -> associate -> track ----
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr input)
  {
    auto t_total_start = std::chrono::high_resolution_clock::now();

    // Step 1: Convert ROS message to PCL point cloud + remove NaN
    auto t0 = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input, *input_cloud);

    if (input_cloud->points.empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty point cloud");
      return;
    }

    std::vector<int> valid_indices;
    pcl::removeNaNFromPointCloud(*input_cloud, *input_cloud, valid_indices);
    if (input_cloud->points.empty()) {
      RCLCPP_WARN(this->get_logger(), "Point cloud empty after NaN removal");
      return;
    }
    input_cloud->width = input_cloud->points.size();
    input_cloud->height = 1;
    input_cloud->is_dense = true;
    auto t1 = std::chrono::high_resolution_clock::now();

    // Step 2: Range filter (first stage of composite pipeline)
    size_t pre_filter_size = input_cloud->points.size();
    auto * composite = dynamic_cast<CompositeFilter *>(filter_.get());
    if (composite && composite->filterCount() >= 2) {
      input_cloud = composite->filterStage(0, input_cloud);  // RangeFilter
    } else {
      input_cloud = filter_->filter(input_cloud);
    }

    if (input_cloud->points.empty()) {
      RCLCPP_DEBUG(this->get_logger(), "Point cloud empty after range filter (was %zu points)", pre_filter_size);
      return;
    }
    size_t after_range = input_cloud->points.size();
    auto t2 = std::chrono::high_resolution_clock::now();

    // Step 3: Voxel downsample (second stage of composite pipeline)
    if (composite && composite->filterCount() >= 2) {
      input_cloud = composite->filterStage(1, input_cloud);  // VoxelFilter
    }
    if (input_cloud->points.empty()) {
      RCLCPP_DEBUG(this->get_logger(), "Point cloud empty after voxel filter (was %zu points)", after_range);
      return;
    }
    size_t after_voxel = input_cloud->points.size();
    auto t3 = std::chrono::high_resolution_clock::now();

    RCLCPP_DEBUG(this->get_logger(), "Filter: %zu(raw) -> %zu(range) -> %zu(voxel)",
      pre_filter_size, after_range, after_voxel);

    // Step 4: Cluster the filtered point cloud (polymorphic)
    auto cluster_results = clusterer_->cluster(input_cloud);
    RCLCPP_DEBUG(this->get_logger(), "%s detected %zu clusters",
      clusterer_->name().c_str(), cluster_results.size());

    // Pad clusters to max_objects_ with empty placeholders
    while (static_cast<int>(cluster_results.size()) < max_objects_) {
      ClusterResult empty;
      empty.cloud->points.push_back(pcl::PointXYZ(0, 0, 0));
      empty.cloud->width = 1;
      empty.cloud->height = 1;
      empty.cloud->is_dense = true;
      empty.centroid = pcl::PointXYZ(0, 0, 0);
      cluster_results.push_back(std::move(empty));
    }
    if (static_cast<int>(cluster_results.size()) > max_objects_) {
      cluster_results.resize(max_objects_);
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    // Step 5: First frame - initialize trackers
    if (first_frame_) {
      for (int i = 0; i < max_objects_; i++) {
        geometry_msgs::msg::Point pos;
        pos.x = cluster_results[i].centroid.x;
        pos.y = cluster_results[i].centroid.y;
        pos.z = cluster_results[i].centroid.z;
        trackers_[i]->init(pos);
      }
      first_frame_ = false;

      RCLCPP_INFO(this->get_logger(),
        "First frame: initialized %d %s trackers, detected %zu clusters (points: %zu -> %zu -> %zu)",
        max_objects_, tracker_name_.c_str(), cluster_results.size(),
        pre_filter_size, after_range, after_voxel);
      return;
    }

    // Step 6: Predict next state for all trackers (polymorphic)
    std::vector<geometry_msgs::msg::Point> predictions;
    for (int i = 0; i < max_objects_; i++) {
      predictions.push_back(trackers_[i]->predict());
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    // Step 7: Build cluster measurement list
    std::vector<geometry_msgs::msg::Point> measurements;
    for (int i = 0; i < max_objects_; i++) {
      geometry_msgs::msg::Point pt;
      pt.x = cluster_results[i].centroid.x;
      pt.y = cluster_results[i].centroid.y;
      pt.z = cluster_results[i].centroid.z;
      measurements.push_back(pt);
    }

    // Step 8: Data association (polymorphic)
    auto association = associator_->associate(predictions, measurements);

    // Step 9: Publish visualization markers at predicted positions
    publishMarkers(predictions);

    // Step 10: Publish object IDs
    std_msgs::msg::Int32MultiArray obj_id_msg;
    for (int i = 0; i < max_objects_; i++) {
      obj_id_msg.data.push_back(association.obj_id[i]);
    }
    obj_id_pub_->publish(obj_id_msg);

    // Step 11: Update trackers with associated measurements (polymorphic)
    for (int i = 0; i < max_objects_; i++) {
      int cluster_idx = association.obj_id[i];
      if (cluster_idx >= 0 && cluster_idx < static_cast<int>(measurements.size())) {
        trackers_[i]->update(measurements[cluster_idx]);
      }
    }

    // Step 12: Publish per-cluster point clouds
    for (int i = 0; i < max_objects_; i++) {
      int cluster_idx = association.obj_id[i];
      if (cluster_idx >= 0 && cluster_idx < static_cast<int>(cluster_results.size())) {
        publishCloud(cluster_pubs_[i], cluster_results[cluster_idx].cloud);
      }
    }
    auto t6 = std::chrono::high_resolution_clock::now();

    // Compute and log timing for each pipeline stage
    double ms_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_range      = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double ms_voxel      = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double ms_cluster    = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double ms_predict    = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double ms_assoc_upd  = std::chrono::duration<double, std::milli>(t6 - t5).count();
    double ms_total      = std::chrono::duration<double, std::milli>(t6 - t_total_start).count();

    RCLCPP_INFO(this->get_logger(),
      "[Timing] total: %.2f ms | preprocess: %.2f | range: %.2f | voxel: %.2f | cluster: %.2f | predict: %.2f | assoc+update: %.2f | pts: %zu->%zu->%zu",
      ms_total, ms_preprocess, ms_range, ms_voxel, ms_cluster, ms_predict, ms_assoc_upd,
      pre_filter_size, after_range, after_voxel);
  }

  // ---- Member variables ----
  // Parameters
  int max_objects_;
  std::string input_topic_;
  std::string output_frame_;
  RangeFilter::Params range_params_;
  VoxelFilter::Params voxel_params_;
  EuclideanClusterer::Params cluster_params_;
  KalmanObjectTracker::Params kf_params_;

  // Polymorphic components (swap implementations here for extensibility)
  std::unique_ptr<PointCloudFilter> filter_;
  std::unique_ptr<Clusterer> clusterer_;
  std::unique_ptr<DataAssociator> associator_;
  std::vector<std::unique_ptr<ObjectTracker>> trackers_;
  std::string tracker_name_;

  bool first_frame_;

  // ROS2 interface
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr obj_id_pub_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> cluster_pubs_;
};

}  // namespace object_tracking_lidar

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<object_tracking_lidar::ObjectTrackingLidarNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
