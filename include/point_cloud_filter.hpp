#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>

namespace object_tracking_lidar
{

// ============================================================================
// Abstract base class for point cloud filtering
// Subclass this to implement custom filtering strategies (e.g., range, voxel, etc.)
// ============================================================================
class PointCloudFilter
{
public:
  virtual ~PointCloudFilter() = default;

  // Apply filter to the input cloud, return filtered cloud
  virtual pcl::PointCloud<pcl::PointXYZ>::Ptr filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) = 0;

  // Get filter name for logging
  virtual std::string name() const = 0;
};

// ============================================================================
// Range-based point cloud filter: removes points outside axis-aligned bounding box
// ============================================================================
class RangeFilter : public PointCloudFilter
{
public:
  struct Params
  {
    double min_x = -3.6;
    double max_x = 3.6;
    double min_y = -2.8;
    double max_y = 2.8;
    double min_z = -0.1;
    double max_z = 2.5;
  };

  explicit RangeFilter(const Params & params) : params_(params) {}

  pcl::PointCloud<pcl::PointXYZ>::Ptr filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) override
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto & pt : cloud->points) {
      if (pt.x >= params_.min_x && pt.x <= params_.max_x &&
          pt.y >= params_.min_y && pt.y <= params_.max_y &&
          pt.z >= params_.min_z && pt.z <= params_.max_z) {
        filtered->points.push_back(pt);
      }
    }
    filtered->width = filtered->points.size();
    filtered->height = 1;
    filtered->is_dense = true;
    return filtered;
  }

  std::string name() const override { return "RangeFilter"; }

  const Params & params() const { return params_; }

private:
  Params params_;
};

}  // namespace object_tracking_lidar
