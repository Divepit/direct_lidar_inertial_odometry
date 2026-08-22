#include "dlio/imu_timestamp_tracker.h"

#include <gtest/gtest.h>

namespace {

TEST(ImuTimestampTracker, ComputesDeltaOnlyAfterFirstSample) {
  dlio::ImuTimestampTracker tracker;

  const auto first = tracker.observe(1'000'000'000LL);
  EXPECT_TRUE(first.accepted);
  EXPECT_FALSE(first.dt_seconds.has_value());

  const auto second = tracker.observe(1'002'500'000LL);
  ASSERT_TRUE(second.accepted);
  ASSERT_TRUE(second.dt_seconds.has_value());
  EXPECT_NEAR(*second.dt_seconds, 0.0025, 1e-12);
  EXPECT_EQ(second.rejected_count, 0U);
}

TEST(ImuTimestampTracker, RejectsDuplicateWithoutAdvancingSequence) {
  dlio::ImuTimestampTracker tracker;

  ASSERT_TRUE(tracker.observe(1'000'000'000LL).accepted);
  const auto duplicate = tracker.observe(1'000'000'000LL);
  EXPECT_FALSE(duplicate.accepted);
  EXPECT_EQ(duplicate.previous_stamp_ns, 1'000'000'000LL);
  EXPECT_EQ(duplicate.rejected_count, 1U);

  const auto next = tracker.observe(1'005'000'000LL);
  ASSERT_TRUE(next.accepted);
  ASSERT_TRUE(next.dt_seconds.has_value());
  EXPECT_NEAR(*next.dt_seconds, 0.005, 1e-12);
}

TEST(ImuTimestampTracker, RejectsBackwardSampleWithoutAdvancingSequence) {
  dlio::ImuTimestampTracker tracker;

  ASSERT_TRUE(tracker.observe(2'000'000'000LL).accepted);
  const auto backward = tracker.observe(1'900'000'000LL);
  EXPECT_FALSE(backward.accepted);
  EXPECT_EQ(backward.previous_stamp_ns, 2'000'000'000LL);
  EXPECT_EQ(backward.rejected_count, 1U);

  const auto next = tracker.observe(2'002'500'000LL);
  ASSERT_TRUE(next.accepted);
  ASSERT_TRUE(next.dt_seconds.has_value());
  EXPECT_NEAR(*next.dt_seconds, 0.0025, 1e-12);
}

TEST(ImuTimestampTracker, ResetAcceptsNewEpochWithoutFabricatedDelta) {
  dlio::ImuTimestampTracker tracker;

  ASSERT_TRUE(tracker.observe(10'000'000'000LL).accepted);
  EXPECT_FALSE(tracker.observe(9'000'000'000LL).accepted);

  tracker.resetSequence();
  const auto first_after_reset = tracker.observe(500'000'000LL);
  EXPECT_TRUE(first_after_reset.accepted);
  EXPECT_FALSE(first_after_reset.dt_seconds.has_value());
  EXPECT_EQ(first_after_reset.rejected_count, 1U);
  EXPECT_EQ(tracker.rejectedCount(), 1U);
}

TEST(ImuTimestampTracker, PreservesLegacyDeltaArithmeticAtEpochTimestamps) {
  dlio::ImuTimestampTracker tracker;
  constexpr std::int64_t kFirst = 1'760'705'399'999'729'000LL;
  constexpr std::int64_t kSecond = 1'760'705'400'002'229'000LL;

  ASSERT_TRUE(tracker.observe(kFirst).accepted);
  const auto second = tracker.observe(kSecond);
  ASSERT_TRUE(second.accepted);
  ASSERT_TRUE(second.dt_seconds.has_value());

  const double expected =
      static_cast<double>(kSecond) / 1e9 - static_cast<double>(kFirst) / 1e9;
  EXPECT_DOUBLE_EQ(*second.dt_seconds, expected);
}

}  // namespace
