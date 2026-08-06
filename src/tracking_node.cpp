#include "point_cloud_filter.hpp"
#include "clusterer.hpp"
#include "data_associator.hpp"
#include "object_tracker.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <rcutils/logging.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl_conversions/pcl_conversions.h>

namespace object_tracking_lidar
{

// ============================================================================
// Main ROS2 node: UWB-LiDAR fusion tracking in odom frame
//
// Coordinate frame design:
//   - output_frame_ (e.g., "base_link"): robot body frame for final output
//   - tracking_frame_ (e.g., "odom"): odometry frame, stable for tracking
//   - All Kalman tracking and fusion is done in tracking_frame_ (odom)
//   - UWB (base_link) → odom, LiDAR (laser) → odom
//   - Final results transformed back to output_frame_ (base_link) for publishing
//
// Fusion pipeline:
//   UWB callback:
//     1. Transform UWB from base_link to odom
//     2. Initialize or update FusionTracker2d with UWB measurement
//     3. Immediately publish fused result in base_link
//
//   Cloud callback:
//     1. Filter + cluster in laser frame (original pipeline)
//     2. Transform cluster centroids to odom
//     3. Multi-object tracking in odom (KalmanObjectTracker, dt-based predict)
//     4. Find nearest cluster to fusion tracker prediction (ROI-based)
//     5. Confirmation frames check
//     6. Update FusionTracker2d with LiDAR cluster (Mahalanobis gating)
//     7. Publish fused result in base_link
// ============================================================================

class ObjectTrackingLidarNode : public rclcpp::Node
{
public:
  ObjectTrackingLidarNode()
  : Node("object_tracking_lidar"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // Suppress TF_OLD_DATA warnings: during bag replay, TF data often arrives
    // out-of-order or from the past. This is expected and harmless when we use
    // TimePointZero for lookups. Set the tf2 logger level to ERROR to hide these.
    rcutils_ret_t ret = rcutils_logging_set_logger_level(
      "tf2", RCUTILS_LOG_SEVERITY_ERROR);
    if (ret != RCUTILS_RET_OK) {
      RCLCPP_WARN(this->get_logger(),
        "Failed to set tf2 logger level (ret=%d), TF_OLD_DATA warnings may appear", ret);
    }

    declareParameters();
    loadParameters();
    initComponents();
    initPubSub();

    RCLCPP_INFO(this->get_logger(),
      "ObjectTrackingLidar node initialized | sub: %s | max_objects: %d | "
      "output_frame: %s | tracking_frame: %s | "
      "uwb_topic: %s | fused_topic: %s | uwb_var: %.4f | lidar_var: %.4f | "
      "mahalanobis_gate²: %.2f | confirm_frames: %d | "
      "kf_accel_noise: %.4f | max_vel: %.2f | max_reject: %d | enable_markers: %s",
      input_topic_.c_str(), max_objects_,
      output_frame_.c_str(), tracking_frame_.c_str(),
      uwb_topic_.c_str(), fused_output_topic_.c_str(),
      fusion_params_.uwb_variance, fusion_params_.lidar_variance,
      fusion_params_.mahalanobis_gate_squared, confirmation_frames_,
      kf_accel_noise_, fusion_params_.max_velocity, fusion_params_.max_rejected_updates,
      enable_markers_ ? "true" : "false");
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
    this->declare_parameter<double>("kf_acceleration_noise", 2.0);
    this->declare_parameter<std::string>("input_topic", "/lidar/points_raw");
    this->declare_parameter<std::string>("output_frame", "base_link");
    this->declare_parameter<std::string>("tracking_frame", "odom");

    // TF2
    this->declare_parameter<double>("tf_timeout_s", 0.20);

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

    // UWB fusion parameters
    this->declare_parameter<std::string>("uwb_topic", "/uwb/target_position");
    this->declare_parameter<std::string>("fused_output_topic", "/uwb/target_position_fused");
    this->declare_parameter<double>("uwb_fusion_distance_threshold", 0.5);
    this->declare_parameter<double>("uwb_variance", 0.36);
    this->declare_parameter<double>("lidar_variance", 0.0225);
    this->declare_parameter<double>("acceleration_variance", 2.0);
    this->declare_parameter<double>("mahalanobis_gate_squared", 9.21);
    this->declare_parameter<double>("max_fusion_velocity", 3.0);
    this->declare_parameter<int>("max_rejected_updates", 5);
    this->declare_parameter<int>("confirmation_frames", 3);
    this->declare_parameter<double>("confirmation_distance_m", 0.35);
    this->declare_parameter<double>("track_reset_timeout_s", 1.0);

    // Marker switch
    this->declare_parameter<bool>("enable_markers", true);
  }

  // ---- Parameter loading ----
  void loadParameters()
  {
    max_objects_ = this->get_parameter("max_tracking_objects").as_int();
    input_topic_ = this->get_parameter("input_topic").as_string();
    output_frame_ = this->get_parameter("output_frame").as_string();
    tracking_frame_ = this->get_parameter("tracking_frame").as_string();
    tf_timeout_s_ = this->get_parameter("tf_timeout_s").as_double();

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

    // Multi-object Kalman tracker params
    kf_params_.process_noise = static_cast<float>(this->get_parameter("kf_process_noise").as_double());
    kf_params_.measurement_noise = static_cast<float>(this->get_parameter("kf_measurement_noise").as_double());
    kf_accel_noise_ = this->get_parameter("kf_acceleration_noise").as_double();
    kf_params_.acceleration_noise = static_cast<float>(kf_accel_noise_);

    // UWB fusion params
    uwb_topic_ = this->get_parameter("uwb_topic").as_string();
    fused_output_topic_ = this->get_parameter("fused_output_topic").as_string();
    uwb_fusion_distance_threshold_ = this->get_parameter("uwb_fusion_distance_threshold").as_double();

    // Fusion tracker params
    fusion_params_.uwb_variance = this->get_parameter("uwb_variance").as_double();
    fusion_params_.lidar_variance = this->get_parameter("lidar_variance").as_double();
    fusion_params_.acceleration_variance = this->get_parameter("acceleration_variance").as_double();
    fusion_params_.mahalanobis_gate_squared = this->get_parameter("mahalanobis_gate_squared").as_double();
    fusion_params_.max_velocity = this->get_parameter("max_fusion_velocity").as_double();
    fusion_params_.max_rejected_updates = this->get_parameter("max_rejected_updates").as_int();

    confirmation_frames_ = this->get_parameter("confirmation_frames").as_int();
    confirmation_distance_m_ = this->get_parameter("confirmation_distance_m").as_double();
    track_reset_timeout_s_ = this->get_parameter("track_reset_timeout_s").as_double();

    // Marker switch
    enable_markers_ = this->get_parameter("enable_markers").as_bool();
  }

  // ---- Initialize components ----
  void initComponents()
  {
    // Composite filter pipeline
    auto composite = std::make_unique<CompositeFilter>();
    composite->addFilter(std::make_unique<RangeFilter>(range_params_));
    composite->addFilter(std::make_unique<VoxelFilter>(voxel_params_));
    filter_ = std::move(composite);

    // Clusterer
    clusterer_ = std::make_unique<EuclideanClusterer>(cluster_params_);

    // Data associator for multi-object tracking
    associator_ = std::make_unique<GreedyAssociator>(max_objects_);

    // Multi-object trackers (operate in odom frame)
    trackers_.clear();
    for (int i = 0; i < max_objects_; i++) {
      trackers_.push_back(std::make_unique<KalmanObjectTracker>(i, kf_params_));
    }
    tracker_name_ = "KalmanObjectTracker";
    first_frame_ = true;

    // UWB-LiDAR fusion tracker (single target, odom frame)
    // Set params so internal clampVelocity/inflateCovariance can access them
    fusion_tracker_.setParams(fusion_params_);
  }

  // ---- Initialize publishers and subscribers ----
  void initPubSub()
  {
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ObjectTrackingLidarNode::cloudCallback, this, std::placeholders::_1));

    uwb_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
      uwb_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ObjectTrackingLidarNode::uwbCallback, this, std::placeholders::_1));

    if (enable_markers_) {
      marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/tracking/markers", 10);
    }

    obj_id_pub_ = this->create_publisher<std_msgs::msg::Int32MultiArray>("/tracking/object_ids", 10);

    fused_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(fused_output_topic_, 10);

    cluster_pubs_.resize(max_objects_);
    for (int i = 0; i < max_objects_; i++) {
      cluster_pubs_[i] = this->create_publisher<sensor_msgs::msg::PointCloud2>("/tracking/cluster_" + std::to_string(i), 10);
    }
  }

  // ========================================================================
  // TF2 utilities — use TimePointZero to avoid TF_OLD_DATA in bag replay
  // ========================================================================

  // Transform a PointStamped to target_frame using latest available TF
  std::optional<geometry_msgs::msg::PointStamped> transformPointStamped(
    const geometry_msgs::msg::PointStamped & input,
    const std::string & target_frame)
  {
    try {
      // Use TimePointZero to get the latest available transform
      // This avoids TF_OLD_DATA warnings during bag replay
      const auto transform = tf_buffer_.lookupTransform(
        target_frame, input.header.frame_id, tf2::TimePointZero,
        tf2::durationFromSec(tf_timeout_s_));
      geometry_msgs::msg::PointStamped output;
      tf2::doTransform(input, output, transform);
      output.header.frame_id = target_frame;
      output.header.stamp = input.header.stamp;  // Keep original stamp for output
      return output;
    } catch (const tf2::TransformException & error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Cannot transform from %s to %s: %s",
        input.header.frame_id.c_str(), target_frame.c_str(), error.what());
      return std::nullopt;
    }
  }

  // Transform a Point (no stamp) using this->now()
  std::optional<geometry_msgs::msg::Point> transformPoint(
    const geometry_msgs::msg::Point & point_in,
    const std::string & source_frame,
    const std::string & target_frame)
  {
    geometry_msgs::msg::PointStamped in_stamp;
    in_stamp.header.frame_id = source_frame;
    in_stamp.header.stamp = this->now();
    in_stamp.point = point_in;

    auto result = transformPointStamped(in_stamp, target_frame);
    if (result.has_value()) {
      return result->point;
    }
    return std::nullopt;
  }

  // Lookup transform from source to tracking_frame_ using latest available TF
  std::optional<geometry_msgs::msg::TransformStamped> lookupTransformToTracking(
    const std::string & source_frame)
  {
    try {
      // Use TimePointZero to get the latest available transform
      return tf_buffer_.lookupTransform(
        tracking_frame_, source_frame, tf2::TimePointZero,
        tf2::durationFromSec(tf_timeout_s_));
    } catch (const tf2::TransformException & error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Cannot lookup transform %s -> %s: %s",
        source_frame.c_str(), tracking_frame_.c_str(), error.what());
      return std::nullopt;
    }
  }

  // Transform fused position from tracking_frame_ to output_frame_
  std::optional<geometry_msgs::msg::PointStamped> transformFusedToOutput(
    const std::array<double, 2> & position_in_tracking,
    const rclcpp::Time & stamp)
  {
    geometry_msgs::msg::PointStamped in_tracking;
    in_tracking.header.frame_id = tracking_frame_;
    in_tracking.header.stamp = stamp;
    in_tracking.point.x = position_in_tracking[0];
    in_tracking.point.y = position_in_tracking[1];
    in_tracking.point.z = 0.0;

    if (output_frame_ == tracking_frame_) {
      return in_tracking;
    }

    auto result = transformPointStamped(in_tracking, output_frame_);
    if (result.has_value()) {
      result->point.z = 0.0;  // 2D tracker
    }
    return result;
  }

  // ========================================================================
  // Fusion tracker time advance (matching uwb_fusion approach)
  // ========================================================================

  void advanceFusionTracker(const rclcpp::Time & stamp)
  {
    if (!fusion_tracker_stamp_.has_value()) {
      fusion_tracker_stamp_ = stamp;
      return;
    }
    const double dt = (stamp - *fusion_tracker_stamp_).seconds();
    if (dt > 0.0) {
      fusion_tracker_.predict(dt, fusion_params_.acceleration_variance);
      fusion_tracker_stamp_ = stamp;
    }
  }

  // ========================================================================
  // UWB callback: transform to odom, update fusion tracker, publish
  // ========================================================================
  void uwbCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    // Step 1: Transform UWB from base_link to odom
    auto transformed = transformPointStamped(*msg, tracking_frame_);
    if (!transformed.has_value()) {
      return;
    }

    const rclcpp::Time stamp(transformed->header.stamp, get_clock()->get_clock_type());
    const double meas_x = transformed->point.x;
    const double meas_y = transformed->point.y;

    // Step 2: Check for track reset timeout
    if (last_measurement_received_.has_value() && (get_clock()->now() - *last_measurement_received_).seconds() > track_reset_timeout_s_) {
      fusion_tracker_.reset();
      fusion_tracker_stamp_.reset();
      pending_cluster_.reset();
      pending_count_ = 0;
    }

    // Step 3: Initialize or update fusion tracker with UWB measurement
    if (!fusion_tracker_.initialized()) {
      fusion_tracker_.initialize(meas_x, meas_y, fusion_params_.uwb_variance);
      fusion_tracker_stamp_ = stamp;
    } else {
      advanceFusionTracker(stamp);

      // If too many consecutive updates were rejected, reinitialize at current UWB position
      // This prevents the fused point from drifting far away from actual target
      if (fusion_tracker_.consecutiveRejected() >= fusion_params_.max_rejected_updates) {
        RCLCPP_WARN(get_logger(),
          "Fusion tracker: %d consecutive rejections, reinitializing at UWB (%.3f, %.3f)",
          fusion_tracker_.consecutiveRejected(), meas_x, meas_y);
        fusion_tracker_.initialize(meas_x, meas_y, fusion_params_.uwb_variance);
        fusion_tracker_stamp_ = stamp;
        pending_cluster_.reset();
        pending_count_ = 0;
      } else {
        fusion_tracker_.update(meas_x, meas_y,
          fusion_params_.uwb_variance, fusion_params_.mahalanobis_gate_squared);
      }
    }

    last_uwb_stamp_ = stamp;
    last_uwb_position_odom_ = {meas_x, meas_y};
    last_measurement_received_ = get_clock()->now();

    // Step 4: Publish fused result immediately (UWB-driven), transform to base_link
    publishFusedPosition(stamp);

    RCLCPP_DEBUG(get_logger(),
      "UWB update in %s: (%.3f, %.3f) → fused(odom): (%.3f, %.3f)",
      tracking_frame_.c_str(), meas_x, meas_y,
      fusion_tracker_.position()[0], fusion_tracker_.position()[1]);
  }

  // ========================================================================
  // Cloud callback: filter → cluster → transform → track → fuse
  // ========================================================================
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr input)
  {
    auto t_total_start = std::chrono::high_resolution_clock::now();

    // Record the input cloud's frame_id for later use
    const std::string cloud_frame_id = input->header.frame_id;

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

    // Step 2: Range filter (in laser frame)
    size_t pre_filter_size = input_cloud->points.size();
    auto * composite = dynamic_cast<CompositeFilter *>(filter_.get());
    if (composite && composite->filterCount() >= 2) {
      input_cloud = composite->filterStage(0, input_cloud);
    } else {
      input_cloud = filter_->filter(input_cloud);
    }
    if (input_cloud->points.empty()) {
      RCLCPP_DEBUG(this->get_logger(), "Point cloud empty after range filter");
      return;
    }
    size_t after_range = input_cloud->points.size();
    auto t2 = std::chrono::high_resolution_clock::now();

    // Step 3: Voxel downsample (in laser frame)
    if (composite && composite->filterCount() >= 2) {
      input_cloud = composite->filterStage(1, input_cloud);
    }
    if (input_cloud->points.empty()) {
      RCLCPP_DEBUG(this->get_logger(), "Point cloud empty after voxel filter");
      return;
    }
    size_t after_voxel = input_cloud->points.size();
    auto t3 = std::chrono::high_resolution_clock::now();

    // Step 4: Cluster in laser frame
    auto cluster_results = clusterer_->cluster(input_cloud);

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

    // Step 5: Get laser→odom transform (use latest available TF)
    auto cloud_transform = lookupTransformToTracking(cloud_frame_id);

    // Step 6: Transform cluster centroids from laser to odom
    std::vector<geometry_msgs::msg::Point> centroids_in_odom;
    centroids_in_odom.resize(cluster_results.size());
    if (cloud_transform.has_value()) {
      for (size_t i = 0; i < cluster_results.size(); i++) {
        geometry_msgs::msg::PointStamped laser_pt;
        laser_pt.header.frame_id = cloud_frame_id;
        laser_pt.header.stamp = input->header.stamp;
        laser_pt.point.x = cluster_results[i].centroid.x;
        laser_pt.point.y = cluster_results[i].centroid.y;
        laser_pt.point.z = cluster_results[i].centroid.z;
        geometry_msgs::msg::PointStamped odom_pt;
        tf2::doTransform(laser_pt, odom_pt, *cloud_transform);
        centroids_in_odom[i] = odom_pt.point;
      }
    } else {
      // Fallback: use raw centroids (will be in laser frame, not ideal)
      for (size_t i = 0; i < cluster_results.size(); i++) {
        centroids_in_odom[i].x = cluster_results[i].centroid.x;
        centroids_in_odom[i].y = cluster_results[i].centroid.y;
        centroids_in_odom[i].z = cluster_results[i].centroid.z;
      }
    }

    // Step 7: Multi-object tracking in odom frame (with dt-based prediction)
    rclcpp::Time cloud_stamp(input->header.stamp, get_clock()->get_clock_type());

    if (first_frame_) {
      for (int i = 0; i < max_objects_; i++) {
        trackers_[i]->init(centroids_in_odom[i]);
      }
      first_frame_ = false;
      last_cloud_stamp_ = cloud_stamp;
      RCLCPP_INFO(this->get_logger(),
        "First frame: initialized %d %s trackers in %s, detected %zu clusters",
        max_objects_, tracker_name_.c_str(), tracking_frame_.c_str(),
        cluster_results.size());
      return;
    }

    // Compute dt for multi-object tracker prediction
    double cloud_dt = 0.0;
    if (last_cloud_stamp_.has_value()) {
      cloud_dt = (cloud_stamp - *last_cloud_stamp_).seconds();
      if (cloud_dt <= 0.0 || cloud_dt > 1.0) {
        cloud_dt = 0.1;  // Default fallback for out-of-order or large gaps
      }
    } else {
      cloud_dt = 0.1;  // Default dt
    }
    last_cloud_stamp_ = cloud_stamp;

    // Predict for all multi-object trackers (in odom, with dt)
    std::vector<geometry_msgs::msg::Point> predictions_in_odom;
    for (int i = 0; i < max_objects_; i++) {
      predictions_in_odom.push_back(trackers_[i]->predict(cloud_dt));
    }

    // Data association in odom
    auto association = associator_->associate(predictions_in_odom, centroids_in_odom);

    // Update multi-object trackers in odom
    for (int i = 0; i < max_objects_; i++) {
      int cluster_idx = association.obj_id[i];
      if (cluster_idx >= 0 && cluster_idx < static_cast<int>(centroids_in_odom.size())) {
        trackers_[i]->update(centroids_in_odom[cluster_idx]);
      }
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    // Step 8: UWB-LiDAR fusion (only if fusion tracker is initialized)
    if (fusion_tracker_.initialized() && last_uwb_stamp_.has_value()) {
      // Check time alignment between cloud and last UWB
      if (std::abs((cloud_stamp - *last_uwb_stamp_).seconds()) <=
          std::max(0.15, track_reset_timeout_s_)) {
        advanceFusionTracker(cloud_stamp);

        // Find nearest cluster to fusion tracker prediction in odom
        auto predicted_pos = fusion_tracker_.position();
        double threshold_sq = uwb_fusion_distance_threshold_ * uwb_fusion_distance_threshold_;
        double best_dist_sq = threshold_sq;
        int best_idx = -1;

        for (size_t i = 0; i < centroids_in_odom.size(); i++) {
          double dx = centroids_in_odom[i].x - predicted_pos[0];
          double dy = centroids_in_odom[i].y - predicted_pos[1];
          double dist_sq = dx * dx + dy * dy;
          if (dist_sq <= best_dist_sq) {
            best_dist_sq = dist_sq;
            best_idx = static_cast<int>(i);
          }
        }

        if (best_idx >= 0) {
          double cx = centroids_in_odom[best_idx].x;
          double cy = centroids_in_odom[best_idx].y;

          // Confirmation frames check
          if (pending_cluster_.has_value() &&
              std::hypot(cx - (*pending_cluster_)[0],
                         cy - (*pending_cluster_)[1]) <= confirmation_distance_m_) {
            ++pending_count_;
          } else {
            pending_count_ = 1;
          }
          pending_cluster_ = {cx, cy};

          // Update fusion tracker with LiDAR cluster (Mahalanobis gating)
          if (pending_count_ >= confirmation_frames_) {
            bool accepted = fusion_tracker_.update(cx, cy,
              fusion_params_.lidar_variance, fusion_params_.mahalanobis_gate_squared);
            if (accepted) {
              RCLCPP_DEBUG(get_logger(),
                "LiDAR cluster_%d accepted by fusion tracker in %s: (%.3f, %.3f)",
                best_idx, tracking_frame_.c_str(), cx, cy);
            } else {
              RCLCPP_DEBUG(get_logger(),
                "LiDAR cluster_%d rejected by Mahalanobis gate in %s", best_idx,
                tracking_frame_.c_str());
            }
          }
        } else {
          // No cluster within threshold - reset pending
          pending_cluster_.reset();
          pending_count_ = 0;
        }

        last_measurement_received_ = get_clock()->now();
      }
    }

    // Step 9: Publish fused result (transform odom → base_link)
    publishFusedPosition(cloud_stamp);

    // Step 10: Publish per-cluster point clouds (in original laser frame)
    for (int i = 0; i < max_objects_; i++) {
      int cluster_idx = association.obj_id[i];
      if (cluster_idx >= 0 && cluster_idx < static_cast<int>(cluster_results.size())) {
        publishCloud(cluster_pubs_[i], cluster_results[cluster_idx].cloud, cloud_frame_id);
      }
    }

    // Step 11: Publish visualization markers (transform predictions back to output_frame)
    if (enable_markers_ && marker_pub_) {
      std::vector<geometry_msgs::msg::Point> predictions_in_output;
      predictions_in_output.resize(predictions_in_odom.size());
      for (size_t i = 0; i < predictions_in_odom.size(); i++) {
        auto transformed = transformPoint(
          predictions_in_odom[i], tracking_frame_, output_frame_);
        predictions_in_output[i] = transformed.value_or(predictions_in_odom[i]);
      }
      publishMarkers(predictions_in_output);
    }

    // Step 12: Publish object IDs
    std_msgs::msg::Int32MultiArray obj_id_msg;
    for (int i = 0; i < max_objects_; i++) {
      obj_id_msg.data.push_back(association.obj_id[i]);
    }
    obj_id_pub_->publish(obj_id_msg);

    auto t6 = std::chrono::high_resolution_clock::now();

    // Timing
    double ms_total = std::chrono::duration<double, std::milli>(t6 - t_total_start).count();
    auto fused_pos = fusion_tracker_.position();
    RCLCPP_INFO(this->get_logger(),
      "[Timing] total: %.2f ms | dt: %.3f | pts: %zu→%zu→%zu | fused(odom): (%.3f, %.3f)",
      ms_total, cloud_dt, pre_filter_size, after_range, after_voxel,
      fused_pos[0], fused_pos[1]);
  }

  // ========================================================================
  // Publish fused position (transform from odom to output_frame/base_link)
  // ========================================================================
  void publishFusedPosition(const rclcpp::Time & stamp)
  {
    if (!fusion_tracker_.initialized()) {
      return;
    }

    auto pos = fusion_tracker_.position();
    auto output = transformFusedToOutput(pos, stamp);
    if (output.has_value()) {
      fused_pub_->publish(*output);
    } else {
      // Fallback: publish in tracking_frame
      geometry_msgs::msg::PointStamped msg;
      msg.header.frame_id = tracking_frame_;
      msg.header.stamp = stamp;
      msg.point.x = pos[0];
      msg.point.y = pos[1];
      msg.point.z = 0.0;
      fused_pub_->publish(msg);
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Fused output transform to %s failed, publishing in %s",
        output_frame_.c_str(), tracking_frame_.c_str());
    }
  }

  // ---- Publish point cloud helper (in the original sensor frame) ----
  void publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    const std::string & frame_id)
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.frame_id = frame_id;
    msg.header.stamp = this->now();
    pub->publish(msg);
  }

  // ---- Publish visualization markers ----
  void publishMarkers(const std::vector<geometry_msgs::msg::Point> & positions_in_output)
  {
    if (!enable_markers_ || !marker_pub_) {
      return;
    }

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
      m.pose.position = positions_in_output[i];
      marker_array.markers.push_back(m);
    }
    marker_pub_->publish(marker_array);
  }

  // ========================================================================
  // Member variables
  // ========================================================================

  // Parameters
  int max_objects_;
  std::string input_topic_;
  std::string output_frame_;
  std::string tracking_frame_;
  double tf_timeout_s_;

  RangeFilter::Params range_params_;
  VoxelFilter::Params voxel_params_;
  EuclideanClusterer::Params cluster_params_;
  KalmanObjectTracker::Params kf_params_;
  double kf_accel_noise_;

  // UWB fusion params
  std::string uwb_topic_;
  std::string fused_output_topic_;
  double uwb_fusion_distance_threshold_;

  // Fusion tracker params
  FusionTracker2d::Params fusion_params_;
  int confirmation_frames_;
  double confirmation_distance_m_;
  double track_reset_timeout_s_;

  // Marker switch
  bool enable_markers_;

  // Multi-object tracking components
  std::unique_ptr<PointCloudFilter> filter_;
  std::unique_ptr<Clusterer> clusterer_;
  std::unique_ptr<DataAssociator> associator_;
  std::vector<std::unique_ptr<ObjectTracker>> trackers_;
  std::string tracker_name_;
  bool first_frame_;
  std::optional<rclcpp::Time> last_cloud_stamp_;

  // UWB-LiDAR fusion tracker (single target, in odom frame)
  FusionTracker2d fusion_tracker_;
  std::optional<rclcpp::Time> fusion_tracker_stamp_;
  std::optional<std::array<double, 2>> pending_cluster_;
  int pending_count_ = 0;
  std::optional<rclcpp::Time> last_uwb_stamp_;
  std::optional<std::array<double, 2>> last_uwb_position_odom_;
  std::optional<rclcpp::Time> last_measurement_received_;

  // TF2
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // ROS2 interface
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr uwb_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr obj_id_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr fused_pub_;
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
