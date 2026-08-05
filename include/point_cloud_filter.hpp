#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <string>
#include <vector>
#include <memory>

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

// ============================================================================
// Voxel grid downsampling filter: reduces point density by voxel grid averaging
// ============================================================================
class VoxelFilter : public PointCloudFilter
{
public:
  struct Params
  {
    double voxel_size_x = 0.1;
    double voxel_size_y = 0.1;
    double voxel_size_z = 0.1;
  };

  explicit VoxelFilter(const Params & params) : params_(params) {}

  pcl::PointCloud<pcl::PointXYZ>::Ptr filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) override
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(
      static_cast<float>(params_.voxel_size_x),
      static_cast<float>(params_.voxel_size_y),
      static_cast<float>(params_.voxel_size_z));
    voxel_grid.filter(*filtered);
    filtered->width = filtered->points.size();
    filtered->height = 1;
    filtered->is_dense = true;
    return filtered;
  }

  std::string name() const override { return "VoxelFilter"; }

  const Params & params() const { return params_; }

private:
  Params params_;
};

// ============================================================================
// Self body filter: removes points within the robot's own bounding box
// Prevents the robot's own structure from being treated as obstacles
// Inspired by the in_robot_frame check in lidar_tracker.hpp
// ============================================================================
class SelfFilter : public PointCloudFilter
{
public:
  struct Params
  {
    double min_x = -0.7;   // 后方排除范围 (x轴负方向)
    double max_x = 0.1;    // 前方排除范围 (x轴正方向)
    double min_y = -0.25;  // 右侧排除范围 (y轴负方向)
    double max_y = 0.25;   // 左侧排除范围 (y轴正方向)
    double min_z = -0.1;   // 下方排除范围 (z轴负方向)
    double max_z = 0.1;    // 上方排除范围 (z轴正方向)
  };

  explicit SelfFilter(const Params & params) : params_(params) {}

  pcl::PointCloud<pcl::PointXYZ>::Ptr filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) override
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto & pt : cloud->points) {
      // 落在机器人本体包围盒内的点予以滤除
      if (pt.x >= params_.min_x && pt.x <= params_.max_x &&
          pt.y >= params_.min_y && pt.y <= params_.max_y &&
          pt.z >= params_.min_z && pt.z <= params_.max_z) {
        continue;
      }
      filtered->points.push_back(pt);
    }
    filtered->width = filtered->points.size();
    filtered->height = 1;
    filtered->is_dense = true;
    return filtered;
  }

  std::string name() const override { return "SelfFilter"; }

  const Params & params() const { return params_; }

private:
  Params params_;
};

// ============================================================================
// Composite filter: chains multiple PointCloudFilter instances sequentially
// Filters are applied in the order they are added
// ============================================================================
class CompositeFilter : public PointCloudFilter
{
public:
  CompositeFilter() = default;

  // Add a filter to the pipeline (order matters: first added = first applied)
  void addFilter(std::unique_ptr<PointCloudFilter> f)
  {
    filter_chain_.push_back(std::move(f));
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) override
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr result = cloud;
    for (auto & f : filter_chain_) {
      result = f->filter(result);
      if (result->points.empty()) {
        break;  // Early exit if a filter produces empty cloud
      }
    }
    return result;
  }

  std::string name() const override
  {
    std::string names = "CompositeFilter[";
    for (size_t i = 0; i < filter_chain_.size(); i++) {
      if (i > 0) names += " -> ";
      names += filter_chain_[i]->name();
    }
    names += "]";
    return names;
  }

  // Get individual filter names for per-stage logging
  std::vector<std::string> filterNames() const
  {
    std::vector<std::string> names;
    for (const auto & f : filter_chain_) {
      names.push_back(f->name());
    }
    return names;
  }

  // Access individual filter for per-stage timing
  size_t filterCount() const { return filter_chain_.size(); }

  pcl::PointCloud<pcl::PointXYZ>::Ptr filterStage(
    size_t index,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
  {
    if (index < filter_chain_.size()) {
      return filter_chain_[index]->filter(cloud);
    }
    return cloud;
  }

private:
  std::vector<std::unique_ptr<PointCloudFilter>> filter_chain_;
};

}  // namespace object_tracking_lidar
