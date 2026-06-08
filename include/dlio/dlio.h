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

// SYSTEM
#include <atomic>
#include <chrono>
#include <cstdint>

#ifdef HAS_CPUID
#include <cpuid.h>
#endif

#include <ctime>
#include <fstream>
#include <future>
#include <iomanip>
#include <ios>
#include <iostream>
#include <mutex>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/times.h>
#include <thread>
#include <malloc.h>
#include <memory>
#include <cmath>

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// BOOST
#include <boost/format.hpp>

// PCL
#define PCL_NO_PRECOMPILE

// DLIO
#include <nano_gicp/nano_gicp.h>

namespace dlio {
  enum class SensorType : std::uint8_t { OUSTER, VELODYNE, HESAI, LIVOX, UNKNOWN };

  class OdomNode;
  class MapNode;

  struct EIGEN_ALIGN16 Point {
    Point(): data{0.f, 0.f, 0.f, 1.f} { timestamp = 0.0; }

    PCL_ADD_POINT4D;
    float intensity; // intensity
    std::uint16_t ring = 0;
    union {
    std::uint32_t t;   // (Ouster) time since beginning of scan in nanoseconds
    float time;        // (Velodyne) time since beginning of scan in seconds
    double timestamp;  // (Hesai) absolute timestamp in seconds
                       // (Livox) absolute timestamp in (seconds * 10e9)
    };
    std::uint32_t raw_index = 0;
    std::uint16_t native_col = 0;
    std::uint8_t has_native_cell = 0;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}

POINT_CLOUD_REGISTER_POINT_STRUCT(dlio::Point,
                                 (float, x, x)
                                 (float, y, y)
                                 (float, z, z)
                                 (float, intensity, intensity)
                                 (std::uint16_t, ring, ring)
                                 (std::uint32_t, t, t)
                                 (float, time, time)
                                 (double, timestamp, timestamp)
                                 (std::uint32_t, raw_index, raw_index)
                                 (std::uint16_t, native_col, native_col)
                                 (std::uint8_t, has_native_cell, has_native_cell))

typedef dlio::Point PointType;
