#include "dlio/imu_unit_scaler.h"

#include <gtest/gtest.h>

#include <cmath>

namespace {

constexpr double kGravity = 9.80665;
constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

}  // namespace

TEST(ImuUnitScaler, RosStandardUnitsRemainIdentity) {
  dlio::ImuUnitScaler scaler;
  dlio::ImuUnitScaleConfig config;
  config.min_samples = 5;
  scaler.configure(config);

  for (int i = 0; i < 5; ++i) {
    const auto& decision = scaler.observe(9.81, 0.05, 0.01 * i, kGravity);
    EXPECT_DOUBLE_EQ(decision.accel_scale, 1.0);
    EXPECT_DOUBLE_EQ(decision.gyro_scale, 1.0);
  }

  const auto& decision = scaler.decision();
  EXPECT_TRUE(decision.locked);
  EXPECT_EQ(decision.classification, "accel_mps2,gyro_radps_assumed");
}

TEST(ImuUnitScaler, GAccelAndDegPerSecGyroScaleImmediately) {
  dlio::ImuUnitScaler scaler;
  dlio::ImuUnitScaleConfig config;
  config.min_samples = 50;
  scaler.configure(config);

  const auto& first = scaler.observe(0.99, 0.25, 10.0, kGravity);
  EXPECT_FALSE(first.locked);
  EXPECT_NEAR(first.accel_scale, kGravity, 1.0e-9);
  EXPECT_NEAR(first.gyro_scale, kDegToRad, 1.0e-12);
  EXPECT_EQ(first.classification, "accel_g,gyro_degps_assumed_from_accel_g");

  for (int i = 1; i < 50; ++i) {
    scaler.observe(1.0, 0.2, 10.0 + 0.005 * i, kGravity);
  }

  const auto& decision = scaler.decision();
  EXPECT_TRUE(decision.locked);
  EXPECT_NEAR(decision.accel_scale, kGravity, 1.0e-9);
  EXPECT_NEAR(decision.gyro_scale, kDegToRad, 1.0e-12);
}

TEST(ImuUnitScaler, OverridesForceScalesBeforeSampleWindowCompletes) {
  dlio::ImuUnitScaler scaler;
  dlio::ImuUnitScaleConfig config;
  config.min_samples = 100;
  config.accel_scale_override = 2.0;
  config.gyro_scale_override = 3.0;
  scaler.configure(config);

  const auto& decision = scaler.observe(9.81, 0.01, 0.0, kGravity);
  EXPECT_TRUE(decision.locked);
  EXPECT_DOUBLE_EQ(decision.accel_scale, 2.0);
  EXPECT_DOUBLE_EQ(decision.gyro_scale, 3.0);
  EXPECT_EQ(decision.classification, "accel_manual_override,gyro_manual_override");
}

TEST(ImuUnitScaler, MaxWaitLocksDecision) {
  dlio::ImuUnitScaler scaler;
  dlio::ImuUnitScaleConfig config;
  config.min_samples = 1000;
  config.max_wait_sec = 0.1;
  scaler.configure(config);

  scaler.observe(0.99, 0.2, 1.0, kGravity);
  const auto& decision = scaler.observe(1.01, 0.2, 1.11, kGravity);
  EXPECT_TRUE(decision.locked);
  EXPECT_NEAR(decision.accel_scale, kGravity, 1.0e-9);
  EXPECT_NEAR(decision.gyro_scale, kDegToRad, 1.0e-12);
}

TEST(ImuUnitScaler, LockedDecisionIgnoresLaterDifferentUnits) {
  dlio::ImuUnitScaler scaler;
  dlio::ImuUnitScaleConfig config;
  config.min_samples = 2;
  scaler.configure(config);

  scaler.observe(1.0, 0.2, 0.0, kGravity);
  scaler.observe(1.0, 0.2, 0.01, kGravity);
  ASSERT_TRUE(scaler.decision().locked);

  const auto& decision = scaler.observe(9.81, 0.01, 1.0, kGravity);
  EXPECT_NEAR(decision.accel_scale, kGravity, 1.0e-9);
  EXPECT_NEAR(decision.gyro_scale, kDegToRad, 1.0e-12);
}
