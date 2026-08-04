#pragma once

#include <geometry_msgs/msg/point.hpp>
#include <limits>
#include <string>
#include <vector>
#include <utility>

namespace object_tracking_lidar
{

// ============================================================================
// Association result: maps tracker index to cluster index
// ============================================================================
struct AssociationResult
{
  // obj_id[tracker_index] = cluster_index
  std::vector<int> obj_id;
};

// ============================================================================
// Abstract base class for data association
// Subclass this to implement custom association strategies (e.g., Hungarian, GNN, JPDA, etc.)
// ============================================================================
class DataAssociator
{
public:
  virtual ~DataAssociator() = default;

  // Associate trackers with cluster measurements
  // predictions: predicted positions from trackers (size = num_trackers)
  // measurements: measured cluster positions (size = num_measurements)
  // Returns association result mapping tracker index to measurement index
  virtual AssociationResult associate(
    const std::vector<geometry_msgs::msg::Point> & predictions,
    const std::vector<geometry_msgs::msg::Point> & measurements) = 0;

  // Get associator name for logging
  virtual std::string name() const = 0;

protected:
  // Compute Euclidean distance between two points
  static double euclideanDistance(
    const geometry_msgs::msg::Point & p1,
    const geometry_msgs::msg::Point & p2)
  {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }
};

// ============================================================================
// Greedy nearest-neighbor data association
// Iteratively selects the globally minimum distance pair from the cost matrix
// ============================================================================
class GreedyAssociator : public DataAssociator
{
public:
  explicit GreedyAssociator(int num_trackers) : num_trackers_(num_trackers) {}

  AssociationResult associate(
    const std::vector<geometry_msgs::msg::Point> & predictions,
    const std::vector<geometry_msgs::msg::Point> & measurements) override
  {
    AssociationResult result;
    int n = num_trackers_;
    result.obj_id.resize(n, -1);

    // Build cost (distance) matrix: rows = trackers, cols = measurements
    std::vector<std::vector<float>> dist_mat(n, std::vector<float>(n, 10000.0f));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n && j < static_cast<int>(measurements.size()); j++) {
        dist_mat[i][j] = static_cast<float>(euclideanDistance(predictions[i], measurements[j]));
      }
    }

    // Greedy assignment: pick global minimum, then mask row/col
    for (int count = 0; count < n; count++) {
      auto [row, col] = findIndexOfMin(dist_mat);
      result.obj_id[row] = col;

      // Mask this row and column
      dist_mat[row] = std::vector<float>(n, 10000.0f);
      for (int r = 0; r < n; r++) {
        dist_mat[r][col] = 10000.0f;
      }
    }

    return result;
  }

  std::string name() const override { return "GreedyAssociator"; }

private:
  int num_trackers_;

  // Find the index of the minimum value in the distance matrix
  static std::pair<int, int> findIndexOfMin(
    const std::vector<std::vector<float>> & dist_mat)
  {
    std::pair<int, int> min_index(0, 0);
    float min_val = std::numeric_limits<float>::max();
    for (size_t i = 0; i < dist_mat.size(); i++) {
      for (size_t j = 0; j < dist_mat[i].size(); j++) {
        if (dist_mat[i][j] < min_val) {
          min_val = dist_mat[i][j];
          min_index = {static_cast<int>(i), static_cast<int>(j)};
        }
      }
    }
    return min_index;
  }
};

}  // namespace object_tracking_lidar
