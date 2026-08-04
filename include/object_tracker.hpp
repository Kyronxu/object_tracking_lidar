#pragma once

#include <geometry_msgs/msg/point.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <vector>

namespace object_tracking_lidar
{

// ============================================================================
// TrackedObject: represents the state of a single tracked object
// ============================================================================
struct TrackedObject
{
  geometry_msgs::msg::Point position;   // Current estimated position
  geometry_msgs::msg::Point velocity;   // Current estimated velocity
  int track_id = -1;                    // Unique track ID
  bool is_active = false;               // Whether this tracker is initialized
};

// ============================================================================
// Abstract base class for object tracking
// Subclass this to implement custom tracking strategies (e.g., Kalman, EKF, UKF, Particle, etc.)
// ============================================================================
class ObjectTracker
{
public:
  virtual ~ObjectTracker() = default;

  // Initialize tracker with a starting position
  virtual void init(const geometry_msgs::msg::Point & position) = 0;

  // Predict the next state, return predicted position
  virtual geometry_msgs::msg::Point predict() = 0;

  // Update tracker with a measurement, return updated position
  virtual geometry_msgs::msg::Point update(const geometry_msgs::msg::Point & measurement) = 0;

  // Get the current tracked object state
  virtual TrackedObject getState() const = 0;

  // Whether this tracker has been initialized
  virtual bool isInitialized() const = 0;

  // Get tracker type name for logging
  virtual std::string name() const = 0;
};

// ============================================================================
// Kalman filter based object tracker
// State vector: [x, y, vx, vy], Measurement vector: [zx, zy]
// ============================================================================
class KalmanObjectTracker : public ObjectTracker
{
public:
  struct Params
  {
    float process_noise = 0.01f;      // Process noise covariance (Q)
    float measurement_noise = 0.1f;   // Measurement noise covariance (R)
  };

  KalmanObjectTracker() : track_id_(-1), initialized_(false)
  {
    // Default construct, init() must be called afterwards
  }

  KalmanObjectTracker(int track_id, const Params & params) : track_id_(track_id), params_(params), initialized_(false) {}
  void init(const geometry_msgs::msg::Point & position) override
  {
    // State dim = 4 [x, y, vx, vy], Measurement dim = 2 [zx, zy]
    kf_ = cv::KalmanFilter(4, 2, 0, CV_32F);

    // State transition matrix A (constant velocity model)
    // [1 0 1 0]
    // [0 1 0 1]
    // [0 0 1 0]
    // [0 0 0 1]
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
      1, 0, 1, 0,
      0, 1, 0, 1,
      0, 0, 1, 0,
      0, 0, 0, 1);

    // Measurement matrix H
    cv::setIdentity(kf_.measurementMatrix);

    // Process noise covariance Q
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(params_.process_noise));

    // Measurement noise covariance R
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar(params_.measurement_noise));

    // Initialize state
    kf_.statePre.at<float>(0) = static_cast<float>(position.x);
    kf_.statePre.at<float>(1) = static_cast<float>(position.y);
    kf_.statePre.at<float>(2) = 0.0f;
    kf_.statePre.at<float>(3) = 0.0f;

    kf_.statePost.at<float>(0) = static_cast<float>(position.x);
    kf_.statePost.at<float>(1) = static_cast<float>(position.y);
    kf_.statePost.at<float>(2) = 0.0f;
    kf_.statePost.at<float>(3) = 0.0f;

    initialized_ = true;
  }

  geometry_msgs::msg::Point predict() override
  {
    geometry_msgs::msg::Point pt;
    if (!initialized_) {
      pt.x = pt.y = pt.z = 0.0;
      return pt;
    }
    cv::Mat prediction = kf_.predict();
    pt.x = prediction.at<float>(0);
    pt.y = prediction.at<float>(1);
    pt.z = 0.0;
    return pt;
  }

  geometry_msgs::msg::Point update(const geometry_msgs::msg::Point & measurement) override
  {
    geometry_msgs::msg::Point pt;
    if (!initialized_) {
      pt.x = pt.y = pt.z = 0.0;
      return pt;
    }
    // Skip invalid measurements (0, 0)
    if (measurement.x == 0.0 && measurement.y == 0.0) {
      pt.x = kf_.statePre.at<float>(0);
      pt.y = kf_.statePre.at<float>(1);
      pt.z = 0.0;
      return pt;
    }
    cv::Mat meas = (cv::Mat_<float>(2, 1) <<
      static_cast<float>(measurement.x),
      static_cast<float>(measurement.y));
    cv::Mat estimated = kf_.correct(meas);
    pt.x = estimated.at<float>(0);
    pt.y = estimated.at<float>(1);
    pt.z = 0.0;
    return pt;
  }

  TrackedObject getState() const override
  {
    TrackedObject state;
    state.track_id = track_id_;
    state.is_active = initialized_;
    if (initialized_) {
      state.position.x = kf_.statePre.at<float>(0);
      state.position.y = kf_.statePre.at<float>(1);
      state.position.z = 0.0;
      state.velocity.x = kf_.statePre.at<float>(2);
      state.velocity.y = kf_.statePre.at<float>(3);
      state.velocity.z = 0.0;
    }
    return state;
  }

  bool isInitialized() const override { return initialized_; }

  std::string name() const override { return "KalmanObjectTracker"; }

  int trackId() const { return track_id_; }

private:
  int track_id_;
  Params params_;
  cv::KalmanFilter kf_;
  bool initialized_;
};

}  // namespace object_tracking_lidar
