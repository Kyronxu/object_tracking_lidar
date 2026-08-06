#pragma once

#include <geometry_msgs/msg/point.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <vector>
#include <array>
#include <cmath>

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

  // Predict the next state with time step dt, return predicted position
  virtual geometry_msgs::msg::Point predict(double dt = 1.0) = 0;

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
// Kalman filter based object tracker (for multi-object tracking)
// State vector: [x, y, vx, vy], Measurement vector: [zx, zy]
// Operates in tracking_frame_ (odom) to avoid ego-rotation-induced drift
// Uses dt-based transition matrix for proper time-step handling
// ============================================================================
class KalmanObjectTracker : public ObjectTracker
{
public:
  struct Params
  {
    float process_noise = 0.01f;        // Process noise covariance (Q)
    float measurement_noise = 0.1f;     // Measurement noise covariance (R)
    float acceleration_noise = 2.0f;    // Acceleration noise variance for dt-based Q
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

    // State transition matrix A (constant velocity model, will be updated with dt in predict)
    // Default identity, actual values set in predict()
    cv::setIdentity(kf_.transitionMatrix);

    // Measurement matrix H
    cv::setIdentity(kf_.measurementMatrix);

    // Process noise covariance Q (will be updated with dt in predict)
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

  geometry_msgs::msg::Point predict(double dt = 1.0) override
  {
    geometry_msgs::msg::Point pt;
    if (!initialized_) {
      pt.x = pt.y = pt.z = 0.0;
      return pt;
    }

    // Update transition matrix with actual dt
    // [1  0  dt 0 ]
    // [0  1  0  dt]
    // [0  0  1  0 ]
    // [0  0  0  1 ]
    float fdt = static_cast<float>(dt);
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
      1, 0, fdt, 0,
      0, 1, 0, fdt,
      0, 0, 1, 0,
      0, 0, 0, 1);

    // Update process noise covariance Q with dt-dependent model
    // Piecewise constant white noise jerk model:
    // Q = [dt^4/4  0      dt^3/2  0     ]
    //     [0       dt^4/4  0       dt^3/2]
    //     [dt^3/2  0       dt^2    0     ]
    //     [0       dt^3/2  0       dt^2  ] * acceleration_noise
    float dt2 = fdt * fdt;
    float dt3 = dt2 * fdt;
    float dt4 = dt3 * fdt;
    float a = params_.acceleration_noise;
    kf_.processNoiseCov = (cv::Mat_<float>(4, 4) <<
      dt4/4, 0,     dt3/2, 0,
      0,     dt4/4, 0,     dt3/2,
      dt3/2, 0,     dt2,   0,
      0,     dt3/2, 0,     dt2) * a;

    // Ensure Q has minimum floor to prevent singularity
    cv::Mat identity_floor = cv::Mat::eye(4, 4, CV_32F) * params_.process_noise;
    cv::max(kf_.processNoiseCov, identity_floor, kf_.processNoiseCov);

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

// ============================================================================
// FusionTracker2d: 2D constant-velocity Kalman tracker with Mahalanobis gating
// Modeled after uwb_fusion::ConstantVelocityTracker2d
//
// State vector: [x, y, vx, vy]  (all in tracking_frame / odom)
// Measurement vector: [zx, zy]
// Supports separate UWB and LiDAR measurement variances
// Mahalanobis distance gating rejects outlier measurements
// ============================================================================
class FusionTracker2d
{
public:
  struct Params
  {
    double uwb_variance = 0.36;            // UWB measurement variance (σ² = 0.6² m²)
    double lidar_variance = 0.0225;        // LiDAR measurement variance (σ² = 0.15² m²)
    double acceleration_variance = 2.0;    // Process/acceleration noise variance
    double mahalanobis_gate_squared = 9.21; // 99% chi-square gate for 2D
    double max_velocity = 3.0;             // Maximum allowed velocity (m/s), clamp to prevent runaway
    int max_rejected_updates = 5;          // Reinitialize after N consecutive rejected updates
  };

  FusionTracker2d() : initialized_(false), consecutive_rejected_(0) {}

  void reset() { initialized_ = false; consecutive_rejected_ = 0; }

  bool initialized() const { return initialized_; }

  // Set params (must be called before initialize/predict/update)
  void setParams(const Params & params) { params_ = params; }

  // How many consecutive updates have been rejected (for external reinit logic)
  int consecutiveRejected() const { return consecutive_rejected_; }

  // Initialize with a 2D position and position variance
  void initialize(double x, double y, double position_variance)
  {
    state_ = {x, y, 0.0, 0.0};
    covariance_ = {};
    covariance_[0][0] = covariance_[1][1] = position_variance;
    covariance_[2][2] = covariance_[3][3] = 4.0;  // Initial velocity uncertainty
    initialized_ = true;
    consecutive_rejected_ = 0;
  }

  // Predict with time step dt and acceleration noise
  void predict(double dt, double acceleration_variance)
  {
    if (!initialized_ || dt <= 0.0) {
      return;
    }
    dt = std::min(dt, 1.0);  // Cap dt to prevent large jumps

    // State prediction: x += dt * vx, y += dt * vy
    state_[0] += dt * state_[2];
    state_[1] += dt * state_[3];

    // Covariance prediction per axis pair (position, velocity)
    for (const auto axes : {std::array<size_t, 2>{0U, 2U}, std::array<size_t, 2>{1U, 3U}}) {
      const auto p = axes[0];
      const auto v = axes[1];
      const double pp = covariance_[p][p];
      const double pv = covariance_[p][v];
      const double vv = covariance_[v][v];
      const double dt2 = dt * dt;
      covariance_[p][p] = pp + 2.0 * dt * pv + dt2 * vv +
                           0.25 * dt2 * dt2 * acceleration_variance;
      covariance_[p][v] = covariance_[v][p] =
          pv + dt * vv + 0.5 * dt2 * dt * acceleration_variance;
      covariance_[v][v] = vv + dt2 * acceleration_variance;
    }

    // Clamp velocity to prevent runaway state from carrying position far away
    clampVelocity();
  }

  // Update with a 2D measurement, with Mahalanobis gating
  // Returns true if the measurement was accepted (within gate), false if rejected
  bool update(double meas_x, double meas_y, double measurement_variance,
              double mahalanobis_gate_squared)
  {
    if (!initialized_ || measurement_variance <= 0.0) {
      return false;
    }

    // Mahalanobis distance check
    const double dx = meas_x - state_[0];
    const double dy = meas_y - state_[1];
    const double sx = covariance_[0][0] + measurement_variance;
    const double sy = covariance_[1][1] + measurement_variance;

    if (dx * dx / sx + dy * dy / sy > mahalanobis_gate_squared) {
      consecutive_rejected_++;
      // Inflate covariance when rejected to widen the gate for next update
      inflateCovariance(1.5);
      return false;  // Rejected by Mahalanobis gate
    }

    // Kalman update per axis
    updateAxis(0U, 2U, meas_x, measurement_variance);
    updateAxis(1U, 3U, meas_y, measurement_variance);
    consecutive_rejected_ = 0;
    return true;
  }

  // Get current position estimate
  std::array<double, 2> position() const
  {
    return {state_[0], state_[1]};
  }

  // Get current velocity estimate
  std::array<double, 2> velocity() const
  {
    return {state_[2], state_[3]};
  }

private:
  // Clamp velocity magnitude to max_velocity_
  void clampVelocity()
  {
    const double speed = std::hypot(state_[2], state_[3]);
    if (speed > params_.max_velocity && speed > 0.0) {
      const double scale = params_.max_velocity / speed;
      state_[2] *= scale;
      state_[3] *= scale;
    }
  }

  // Inflate covariance diagonals by a factor (widens Mahalanobis gate)
  void inflateCovariance(double factor)
  {
    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 4; j++) {
        if (i == j) {
          covariance_[i][i] *= factor;
        } else {
          covariance_[i][j] *= std::sqrt(factor);
        }
      }
    }
  }

  void updateAxis(size_t p, size_t v, double measurement, double measurement_variance)
  {
    const double innovation = measurement - state_[p];
    const double innovation_variance = covariance_[p][p] + measurement_variance;
    const double kp = covariance_[p][p] / innovation_variance;
    const double kv = covariance_[v][p] / innovation_variance;
    const double pp = covariance_[p][p];
    const double pv = covariance_[p][v];
    state_[p] += kp * innovation;
    state_[v] += kv * innovation;
    covariance_[p][p] -= kp * pp;
    covariance_[p][v] -= kp * pv;
    covariance_[v][p] -= kv * pp;
    covariance_[v][v] -= kv * pv;
  }

  bool initialized_{false};
  int consecutive_rejected_{0};
  Params params_;  // Store params for access in predict/update
  std::array<double, 4> state_{};  // [x, y, vx, vy]
  std::array<std::array<double, 4>, 4> covariance_{};
};

}  // namespace object_tracking_lidar
