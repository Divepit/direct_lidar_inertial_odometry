#include "dlio/scan_time_bounds.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

template <typename T>
sensor_msgs::msg::PointCloud2 makeTimingCloud(const std::string& field_name,
                                              const std::uint8_t datatype,
                                              const std::vector<T>& values,
                                              const double header_time) {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp.sec = static_cast<std::int32_t>(header_time);
  cloud.header.stamp.nanosec =
      static_cast<std::uint32_t>((header_time - static_cast<double>(cloud.header.stamp.sec)) * 1.0e9);
  cloud.height = 1U;
  cloud.width = static_cast<std::uint32_t>(values.size());
  cloud.is_dense = false;
  cloud.is_bigendian = false;
  cloud.point_step = sizeof(T);
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(static_cast<std::size_t>(cloud.row_step));

  sensor_msgs::msg::PointField field;
  field.name = field_name;
  field.offset = 0U;
  field.datatype = datatype;
  field.count = 1U;
  cloud.fields.push_back(field);

  for (std::size_t i = 0; i < values.size(); ++i) {
    std::memcpy(cloud.data.data() + i * sizeof(T), &values[i], sizeof(T));
  }
  return cloud;
}

sensor_msgs::msg::PointCloud2 makeUntimedCloud(const std::uint32_t point_count,
                                               const double header_time) {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp.sec = static_cast<std::int32_t>(header_time);
  cloud.header.stamp.nanosec =
      static_cast<std::uint32_t>((header_time - static_cast<double>(cloud.header.stamp.sec)) * 1.0e9);
  cloud.height = 1U;
  cloud.width = point_count;
  cloud.point_step = 16U;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(static_cast<std::size_t>(cloud.row_step));
  return cloud;
}

}  // namespace

TEST(ScanTimeBounds, AbsoluteTimestampSeconds) {
  const auto cloud = makeTimingCloud<double>(
      "timestamp",
      sensor_msgs::msg::PointField::FLOAT64,
      {101.20, 101.30, 101.25},
      101.20);

  const auto bounds = dlio::scanTimeBoundsFromPointCloud2(cloud);
  EXPECT_TRUE(bounds.valid);
  EXPECT_TRUE(bounds.used_point_timestamps);
  EXPECT_DOUBLE_EQ(bounds.start, 101.20);
  EXPECT_DOUBLE_EQ(bounds.end, 101.30);
}

TEST(ScanTimeBounds, AbsoluteTimestampNanoseconds) {
  const auto cloud = makeTimingCloud<double>(
      "timestamp",
      sensor_msgs::msg::PointField::FLOAT64,
      {1.780000000200e18, 1.780000000300e18, 1.780000000250e18},
      1780000000.20);

  const auto bounds = dlio::scanTimeBoundsFromPointCloud2(cloud);
  EXPECT_TRUE(bounds.valid);
  EXPECT_TRUE(bounds.used_point_timestamps);
  EXPECT_NEAR(bounds.start, 1780000000.20, 1.0e-6);
  EXPECT_NEAR(bounds.end, 1780000000.30, 1.0e-6);
}

TEST(ScanTimeBounds, RelativeNanosecondTField) {
  const auto cloud = makeTimingCloud<std::uint32_t>(
      "t",
      sensor_msgs::msg::PointField::UINT32,
      {0U, 50'000'000U, 100'000'000U},
      42.0);

  const auto bounds = dlio::scanTimeBoundsFromPointCloud2(cloud);
  EXPECT_TRUE(bounds.valid);
  EXPECT_TRUE(bounds.used_point_timestamps);
  EXPECT_DOUBLE_EQ(bounds.start, 42.0);
  EXPECT_DOUBLE_EQ(bounds.end, 42.1);
}

TEST(ScanTimeBounds, RelativeSecondTimeField) {
  const auto cloud = makeTimingCloud<float>(
      "time",
      sensor_msgs::msg::PointField::FLOAT32,
      {0.02F, 0.00F, 0.10F},
      12.0);

  const auto bounds = dlio::scanTimeBoundsFromPointCloud2(cloud);
  EXPECT_TRUE(bounds.valid);
  EXPECT_TRUE(bounds.used_point_timestamps);
  EXPECT_NEAR(bounds.start, 12.0, 1.0e-9);
  EXPECT_NEAR(bounds.end, 12.1, 1.0e-6);
}

TEST(ScanTimeBounds, MissingTimingFieldFallsBackToHeader) {
  const auto cloud = makeUntimedCloud(5U, 77.5);

  const auto bounds = dlio::scanTimeBoundsFromPointCloud2(cloud);
  EXPECT_TRUE(bounds.valid);
  EXPECT_FALSE(bounds.used_point_timestamps);
  EXPECT_DOUBLE_EQ(bounds.start, 77.5);
  EXPECT_DOUBLE_EQ(bounds.end, 77.5);
}

TEST(ScanTimeBounds, EmptyCloudFallsBackToHeaderButIsInvalid) {
  const auto cloud = makeUntimedCloud(0U, 88.25);

  const auto bounds = dlio::scanTimeBoundsFromPointCloud2(cloud);
  EXPECT_FALSE(bounds.valid);
  EXPECT_FALSE(bounds.used_point_timestamps);
  EXPECT_DOUBLE_EQ(bounds.start, 88.25);
  EXPECT_DOUBLE_EQ(bounds.end, 88.25);
}
