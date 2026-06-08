/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"

namespace dlio {

struct ScanTimeBounds {
  double start = 0.0;
  double end = 0.0;
  bool used_point_timestamps = false;
  bool valid = false;
};

inline double stampToSeconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<double>(stamp.sec) + 1.0e-9 * static_cast<double>(stamp.nanosec);
}

inline std::size_t pointFieldScalarSize(const std::uint8_t datatype) {
  using sensor_msgs::msg::PointField;
  switch (datatype) {
    case PointField::INT8:
    case PointField::UINT8:
      return 1U;
    case PointField::INT16:
    case PointField::UINT16:
      return 2U;
    case PointField::INT32:
    case PointField::UINT32:
    case PointField::FLOAT32:
      return 4U;
    case PointField::FLOAT64:
      return 8U;
    default:
      return 0U;
  }
}

template <typename T>
inline T readUnalignedScalar(const std::uint8_t* data) {
  T value{};
  std::memcpy(&value, data, sizeof(T));
  return value;
}

inline bool readPointFieldAsDouble(const std::uint8_t* data,
                                   const std::uint8_t datatype,
                                   double& value) {
  using sensor_msgs::msg::PointField;
  switch (datatype) {
    case PointField::INT8:
      value = static_cast<double>(readUnalignedScalar<std::int8_t>(data));
      return true;
    case PointField::UINT8:
      value = static_cast<double>(readUnalignedScalar<std::uint8_t>(data));
      return true;
    case PointField::INT16:
      value = static_cast<double>(readUnalignedScalar<std::int16_t>(data));
      return true;
    case PointField::UINT16:
      value = static_cast<double>(readUnalignedScalar<std::uint16_t>(data));
      return true;
    case PointField::INT32:
      value = static_cast<double>(readUnalignedScalar<std::int32_t>(data));
      return true;
    case PointField::UINT32:
      value = static_cast<double>(readUnalignedScalar<std::uint32_t>(data));
      return true;
    case PointField::FLOAT32:
      value = static_cast<double>(readUnalignedScalar<float>(data));
      return true;
    case PointField::FLOAT64:
      value = readUnalignedScalar<double>(data);
      return true;
    default:
      value = std::numeric_limits<double>::quiet_NaN();
      return false;
  }
}

inline const sensor_msgs::msg::PointField* findTimingField(
    const sensor_msgs::msg::PointCloud2& cloud) {
  const sensor_msgs::msg::PointField* timestamp_field = nullptr;
  const sensor_msgs::msg::PointField* t_field = nullptr;
  const sensor_msgs::msg::PointField* time_field = nullptr;

  for (const auto& field : cloud.fields) {
    if (field.name == "timestamp") {
      timestamp_field = &field;
    } else if (field.name == "t") {
      t_field = &field;
    } else if (field.name == "time") {
      time_field = &field;
    }
  }

  if (timestamp_field != nullptr) {
    return timestamp_field;
  }
  if (t_field != nullptr) {
    return t_field;
  }
  return time_field;
}

inline ScanTimeBounds scanTimeBoundsFromPointCloud2(
    const sensor_msgs::msg::PointCloud2& cloud) {
  ScanTimeBounds bounds;
  const double header_time = stampToSeconds(cloud.header.stamp);
  bounds.start = header_time;
  bounds.end = header_time;

  const auto* timing_field = findTimingField(cloud);
  if (timing_field == nullptr || cloud.width == 0U || cloud.height == 0U ||
      cloud.point_step == 0U) {
    bounds.valid = (cloud.width > 0U && cloud.height > 0U);
    return bounds;
  }

  const std::size_t scalar_size = pointFieldScalarSize(timing_field->datatype);
  if (scalar_size == 0U) {
    return bounds;
  }

  const std::size_t point_count =
      static_cast<std::size_t>(cloud.width) * static_cast<std::size_t>(cloud.height);
  bool have_time = false;
  bool timestamp_uses_nanoseconds = false;
  double min_time = std::numeric_limits<double>::infinity();
  double max_time = -std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i < point_count; ++i) {
    const std::size_t row = i / static_cast<std::size_t>(cloud.width);
    const std::size_t col = i % static_cast<std::size_t>(cloud.width);
    const std::size_t byte_offset =
        row * static_cast<std::size_t>(cloud.row_step) +
        col * static_cast<std::size_t>(cloud.point_step) +
        static_cast<std::size_t>(timing_field->offset);

    if (byte_offset + scalar_size > cloud.data.size()) {
      continue;
    }

    double raw_time = 0.0;
    if (!readPointFieldAsDouble(cloud.data.data() + byte_offset,
                                timing_field->datatype,
                                raw_time) ||
        !std::isfinite(raw_time)) {
      continue;
    }

    if (!have_time && timing_field->name == "timestamp") {
      timestamp_uses_nanoseconds = (raw_time > 1.0e14);
    }

    double point_time = raw_time;
    if (timing_field->name == "timestamp") {
      if (timestamp_uses_nanoseconds) {
        point_time *= 1.0e-9;
      }
    } else if (timing_field->name == "t") {
      point_time = header_time + raw_time * 1.0e-9;
    } else {
      point_time = header_time + raw_time;
    }

    have_time = true;
    min_time = std::min(min_time, point_time);
    max_time = std::max(max_time, point_time);
  }

  if (have_time && std::isfinite(min_time) && std::isfinite(max_time)) {
    bounds.start = min_time;
    bounds.end = max_time;
    bounds.used_point_timestamps = true;
    bounds.valid = true;
  }

  return bounds;
}

}  // namespace dlio
