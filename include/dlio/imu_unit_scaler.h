/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 ***********************************************************/

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace dlio {

struct ImuUnitScaleConfig {
  bool enabled = true;
  bool auto_scale = true;
  int min_samples = 50;
  double max_wait_sec = 0.5;
  double warn_period_sec = 2.0;
  bool assume_deg_per_sec_when_accel_g = true;
  double accel_scale_override = 0.0;
  double gyro_scale_override = 0.0;
};

struct ImuUnitScaleDecision {
  bool locked = false;
  double accel_scale = 1.0;
  double gyro_scale = 1.0;
  double raw_accel_median = 0.0;
  double raw_gyro_median = 0.0;
  std::string classification = "uninitialized";

  bool nonIdentity() const {
    return std::abs(accel_scale - 1.0) > 1.0e-9 || std::abs(gyro_scale - 1.0) > 1.0e-9;
  }
};

class ImuUnitScaler {
public:
  void configure(ImuUnitScaleConfig config) {
    config.min_samples = std::max(config.min_samples, 1);
    config.max_wait_sec = std::max(config.max_wait_sec, 0.0);
    config.warn_period_sec = std::max(config.warn_period_sec, 0.1);
    if (config.accel_scale_override < 0.0) {
      config.accel_scale_override = 0.0;
    }
    if (config.gyro_scale_override < 0.0) {
      config.gyro_scale_override = 0.0;
    }
    config_ = config;
  }

  const ImuUnitScaleConfig& config() const { return config_; }
  const ImuUnitScaleDecision& decision() const { return decision_; }

  const ImuUnitScaleDecision& observe(double accel_norm,
                                      double gyro_norm,
                                      double stamp_sec,
                                      double gravity_abs) {
    if (decision_.locked) {
      return decision_;
    }

    if (!std::isfinite(gravity_abs) || gravity_abs <= 0.0) {
      gravity_abs = 9.80665;
    }

    if (!first_stamp_valid_ && std::isfinite(stamp_sec)) {
      first_stamp_valid_ = true;
      first_stamp_sec_ = stamp_sec;
    }

    if (std::isfinite(accel_norm)) {
      raw_accel_norms_.push_back(accel_norm);
    }
    if (std::isfinite(gyro_norm)) {
      raw_gyro_norms_.push_back(gyro_norm);
    }

    updateCurrentDecision(gravity_abs);

    const bool overrides_force_lock =
        config_.accel_scale_override > 0.0 || config_.gyro_scale_override > 0.0;
    const bool enough_samples =
        static_cast<int>(raw_accel_norms_.size()) >= config_.min_samples;
    const bool waited_long_enough =
        first_stamp_valid_ && std::isfinite(stamp_sec) &&
        (stamp_sec - first_stamp_sec_) >= config_.max_wait_sec;

    if (!config_.enabled || !config_.auto_scale || overrides_force_lock ||
        enough_samples || waited_long_enough) {
      updateCurrentDecision(gravity_abs);
      decision_.locked = true;
    }

    return decision_;
  }

private:
  static double median(std::vector<double> values) {
    if (values.empty()) {
      return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if ((values.size() % 2) == 0) {
      return 0.5 * (values[mid - 1] + values[mid]);
    }
    return values[mid];
  }

  void updateCurrentDecision(double gravity_abs) {
    decision_.raw_accel_median = median(raw_accel_norms_);
    decision_.raw_gyro_median = median(raw_gyro_norms_);

    if (!config_.enabled) {
      decision_.accel_scale = config_.accel_scale_override > 0.0 ? config_.accel_scale_override : 1.0;
      decision_.gyro_scale = config_.gyro_scale_override > 0.0 ? config_.gyro_scale_override : 1.0;
      decision_.classification =
          decision_.nonIdentity() ? "unit_check_disabled_manual_override" : "unit_check_disabled_identity";
      return;
    }

    const bool accel_override = config_.accel_scale_override > 0.0;
    const bool gyro_override = config_.gyro_scale_override > 0.0;

    std::string accel_class = "accel_unknown_identity";
    if (accel_override) {
      decision_.accel_scale = config_.accel_scale_override;
      accel_class = "accel_manual_override";
    } else if (!config_.auto_scale) {
      decision_.accel_scale = 1.0;
      accel_class = "accel_auto_scale_disabled";
    } else if (decision_.raw_accel_median >= 0.5 && decision_.raw_accel_median <= 1.5) {
      decision_.accel_scale = gravity_abs;
      accel_class = "accel_g";
    } else if (decision_.raw_accel_median >= 0.6 * gravity_abs &&
               decision_.raw_accel_median <= 1.4 * gravity_abs) {
      decision_.accel_scale = 1.0;
      accel_class = "accel_mps2";
    } else {
      decision_.accel_scale = 1.0;
    }

    std::string gyro_class = "gyro_radps_assumed";
    if (gyro_override) {
      decision_.gyro_scale = config_.gyro_scale_override;
      gyro_class = "gyro_manual_override";
    } else if (!config_.auto_scale) {
      decision_.gyro_scale = 1.0;
      gyro_class = "gyro_auto_scale_disabled";
    } else if (accel_class == "accel_g" && config_.assume_deg_per_sec_when_accel_g) {
      decision_.gyro_scale = 3.14159265358979323846 / 180.0;
      gyro_class = "gyro_degps_assumed_from_accel_g";
    } else {
      decision_.gyro_scale = 1.0;
    }

    decision_.classification = accel_class + "," + gyro_class;
  }

  ImuUnitScaleConfig config_;
  ImuUnitScaleDecision decision_;
  std::vector<double> raw_accel_norms_;
  std::vector<double> raw_gyro_norms_;
  bool first_stamp_valid_ = false;
  double first_stamp_sec_ = 0.0;
};

}  // namespace dlio
