#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

namespace object_tracking_lidar
{

// Single Kalman tracker with state vector [x, y, v_x, v_y] and measurement vector [z_x, z_y]
class KalmanTracker
{
public:
  KalmanTracker()
  {
    // Default constructor, init() must be called afterwards
    initialized_ = false;
  }

  KalmanTracker(float x, float y, float sigma_p = 0.01, float sigma_q = 0.1)
  {
    init(x, y, sigma_p, sigma_q);
  }

  void init(float x, float y, float sigma_p = 0.01, float sigma_q = 0.1)
  {
    // State dim = 4 [x, y, vx, vy], Measurement dim = 2 [zx, zy]
    kf_ = cv::KalmanFilter(4, 2, 0, CV_32F);

    // State transition matrix A
    // [1 0 1 0]
    // [0 1 0 1]
    // [0 0 1 0]
    // [0 0 0 1]
    kf_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

    // Measurement matrix H
    cv::setIdentity(kf_.measurementMatrix);

    // Process noise covariance Q
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(sigma_p));

    // Measurement noise covariance R
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar(sigma_q));

    // Initial state
    kf_.statePre.at<float>(0) = x;
    kf_.statePre.at<float>(1) = y;
    kf_.statePre.at<float>(2) = 0.0f; // Initial velocity vx
    kf_.statePre.at<float>(3) = 0.0f; // Initial velocity vy

    kf_.statePost.at<float>(0) = x;
    kf_.statePost.at<float>(1) = y;
    kf_.statePost.at<float>(2) = 0.0f;
    kf_.statePost.at<float>(3) = 0.0f;

    initialized_ = true;
  }

  // Predict the next state
  cv::Mat predict()
  {
    if (!initialized_) {
      return cv::Mat::zeros(4, 1, CV_32F);
    }
    return kf_.predict();
  }

  // Update with measurement
  cv::Mat update(float meas_x, float meas_y)
  {
    if (!initialized_) {
      return cv::Mat::zeros(4, 1, CV_32F);
    }
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << meas_x, meas_y);
    // Skip invalid measurements (0, 0)
    if (meas_x == 0.0f && meas_y == 0.0f) {
      return kf_.statePre;
    }
    return kf_.correct(measurement);
  }

  // Get current predicted position
  float getX() const { return initialized_ ? kf_.statePre.at<float>(0) : 0.0f; }
  float getY() const { return initialized_ ? kf_.statePre.at<float>(1) : 0.0f; }
  float getVx() const { return initialized_ ? kf_.statePre.at<float>(2) : 0.0f; }
  float getVy() const { return initialized_ ? kf_.statePre.at<float>(3) : 0.0f; }
  bool isInitialized() const { return initialized_; }

private:
  cv::KalmanFilter kf_;
  bool initialized_;
};

}  // namespace object_tracking_lidar
