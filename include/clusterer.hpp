#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <string>
#include <vector>

namespace object_tracking_lidar
{

// ============================================================================
// Cluster result: a single cluster with its point cloud and centroid
// ============================================================================
struct ClusterResult
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  pcl::PointXYZ centroid;
  int cluster_id;

  ClusterResult() : cloud(new pcl::PointCloud<pcl::PointXYZ>), centroid(0, 0, 0), cluster_id(-1) {}
};

// ============================================================================
// Abstract base class for point cloud clustering
// Subclass this to implement custom clustering strategies (e.g., Euclidean, DBSCAN, etc.)
// ============================================================================
class Clusterer
{
public:
  virtual ~Clusterer() = default;

  // Perform clustering on the input cloud, return vector of cluster results
  virtual std::vector<ClusterResult> cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) = 0;

  // Get clusterer name for logging
  virtual std::string name() const = 0;
};

// ============================================================================
// Euclidean clustering implementation using PCL
// ============================================================================
class EuclideanClusterer : public Clusterer
{
public:
  struct Params
  {
    double cluster_tolerance = 0.3;
    int min_cluster_size = 10;
    int max_cluster_size = 600;
  };

  explicit EuclideanClusterer(const Params & params) : params_(params) {}

  std::vector<ClusterResult> cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) override
  {
    std::vector<ClusterResult> results;

    if (cloud->points.empty()) {
      return results;
    }

    // Create KD-tree for clustering
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    // Euclidean cluster extraction
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(params_.cluster_tolerance);
    ec.setMinClusterSize(params_.min_cluster_size);
    ec.setMaxClusterSize(params_.max_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    // Extract cluster point clouds and centroids
    for (size_t i = 0; i < cluster_indices.size(); i++) {
      ClusterResult result;
      result.cluster_id = static_cast<int>(i);

      float sum_x = 0.0f, sum_y = 0.0f;
      int num_pts = 0;

      for (const auto & idx : cluster_indices[i].indices) {
        result.cloud->points.push_back(cloud->points[idx]);
        sum_x += cloud->points[idx].x;
        sum_y += cloud->points[idx].y;
        num_pts++;
      }

      result.cloud->width = result.cloud->points.size();
      result.cloud->height = 1;
      result.cloud->is_dense = true;

      result.centroid.x = sum_x / num_pts;
      result.centroid.y = sum_y / num_pts;
      result.centroid.z = 0.0f;

      results.push_back(std::move(result));
    }

    return results;
  }

  std::string name() const override { return "EuclideanClusterer"; }

  const Params & params() const { return params_; }

private:
  Params params_;
};

}  // namespace object_tracking_lidar
