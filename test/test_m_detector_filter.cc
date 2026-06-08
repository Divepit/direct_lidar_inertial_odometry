#include "dlio/m_detector_filter.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

PointType point(float x, float y, float z) {
  PointType p;
  p.x = x;
  p.y = y;
  p.z = z;
  p.intensity = 1.0f;
  p.ring = 0;
  p.timestamp = 1.0;
  p.raw_index = 0;
  p.native_col = 0;
  p.has_native_cell = 0U;
  return p;
}

PointType ringYawPoint(float range, float yaw_deg, std::uint16_t ring,
                       double timestamp = 1.0, std::uint32_t raw_index = 0) {
  const float yaw = yaw_deg * static_cast<float>(M_PI) / 180.0f;
  PointType p = point(range * std::sin(yaw), range * std::cos(yaw), 0.0f);
  p.ring = ring;
  p.timestamp = timestamp;
  p.raw_index = raw_index;
  p.native_col = static_cast<std::uint16_t>(raw_index / 128U);
  p.has_native_cell = 1U;
  return p;
}

PointType ringYawElevationPoint(float range, float yaw_deg, float elevation_deg,
                                std::uint16_t ring, double timestamp = 1.0,
                                std::uint32_t raw_index = 0) {
  const float yaw = yaw_deg * static_cast<float>(M_PI) / 180.0f;
  const float elevation = elevation_deg * static_cast<float>(M_PI) / 180.0f;
  const float horizontal = range * std::cos(elevation);
  PointType p = point(horizontal * std::sin(yaw),
                      horizontal * std::cos(yaw),
                      range * std::sin(elevation));
  p.ring = ring;
  p.timestamp = timestamp;
  p.raw_index = raw_index;
  p.native_col = static_cast<std::uint16_t>(raw_index / 128U);
  p.has_native_cell = 1U;
  return p;
}

pcl::PointCloud<PointType>::Ptr cloud(std::initializer_list<PointType> points) {
  auto out = std::make_shared<pcl::PointCloud<PointType>>();
  out->points.assign(points.begin(), points.end());
  out->width = static_cast<std::uint32_t>(out->points.size());
  out->height = 1U;
  out->is_dense = false;
  return out;
}

dlio::MDetectorFilter::Config config() {
  dlio::MDetectorFilter::Config cfg;
  cfg.enabled = true;
  cfg.voxel_size = 0.20;
  cfg.min_range = 0.0;
  cfg.max_range = 20.0;
  cfg.warmup_scans = 0;
  cfg.static_score_threshold = 2;
  cfg.static_window_scans = 5;
  cfg.max_age_scans = 50;
  cfg.local_radius = 30.0;
  cfg.projection_rows = 128;
  cfg.projection_cols = 900;
  cfg.projection_use_ring_field = true;
  cfg.projection_use_point_timestamp = true;
  cfg.history_duration = 2.0;
  cfg.max_history_frames = 5;
  cfg.frame_duration = 0.1;
  cfg.min_history_votes = 1;
  cfg.case_depth_margin = 0.10;
  cfg.map_consistency_depth = 0.20;
  cfg.min_cluster_points = 3;
  cfg.min_track_cluster_points = 3;
  cfg.max_cluster_extent = 3.0;
  cfg.max_assoc_distance = 1.0;
  cfg.track_confirm_hits = 1;
  cfg.track_ttl_scans = 2;
  cfg.static_veto_ratio = 0.70;
  return cfg;
}

pcl::PointCloud<PointType>::Ptr farStatic(float yaw = 0.0f, std::uint16_t ring = 8) {
  return cloud({
      ringYawPoint(5.0f, yaw, ring, 1.0, 128U * 10U),
      ringYawPoint(5.2f, yaw + 0.4f, ring, 1.001, 128U * 11U),
      ringYawPoint(5.4f, yaw - 0.4f, ring, 1.002, 128U * 12U),
  });
}

pcl::PointCloud<PointType>::Ptr foregroundCluster(float yaw = 0.0f, std::uint16_t ring = 8) {
  return cloud({
      ringYawPoint(2.0f, yaw, ring, 2.0, 128U * 700U),
      ringYawPoint(2.08f, yaw + 0.4f, ring, 2.001, 128U * 701U),
      ringYawPoint(2.16f, yaw - 0.4f, ring, 2.002, 128U * 702U),
      ringYawPoint(2.24f, yaw + 0.8f, ring, 2.003, 128U * 703U),
  });
}

/// Cluster with two history-supported foreground seeds and several physically
/// adjacent non-seed points. This catches accidental single-frame track
/// confirmation from a low seed ratio.
pcl::PointCloud<PointType>::Ptr lowSeedRatioCluster(std::uint16_t ring = 8) {
  return cloud({
      ringYawPoint(2.00f, 0.0f, ring, 2.0, 128U * 700U),
      ringYawPoint(2.05f, 0.4f, ring, 2.001, 128U * 701U),
      ringYawPoint(2.08f, 2.4f, ring, 2.002, 128U * 706U),
      ringYawPoint(2.10f, 2.8f, ring, 2.003, 128U * 707U),
      ringYawPoint(2.12f, 3.2f, ring, 2.004, 128U * 708U),
      ringYawPoint(2.14f, 3.6f, ring, 2.005, 128U * 709U),
  });
}

Eigen::Matrix4f translation(float x, float y, float z) {
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T(0, 3) = x;
  T(1, 3) = y;
  T(2, 3) = z;
  return T;
}

int projectionCol(float yaw_deg, int cols = 900) {
  constexpr float kTwoPi = 2.0f * static_cast<float>(M_PI);
  float yaw = yaw_deg * static_cast<float>(M_PI) / 180.0f;
  while (yaw < 0.0f) {
    yaw += kTwoPi;
  }
  while (yaw >= kTwoPi) {
    yaw -= kTwoPi;
  }
  int col = static_cast<int>(
      std::lround(yaw / kTwoPi * static_cast<float>(cols)));
  if (col >= cols) {
    col -= cols;
  }
  return std::clamp(col, 0, cols - 1);
}

void loadRingElevationModel(dlio::MDetectorFilter& filter,
                            const std::vector<float>& elevation_deg_by_ring,
                            double stamp = 1.0) {
  auto model_scan = std::make_shared<pcl::PointCloud<PointType>>();
  model_scan->points.reserve(elevation_deg_by_ring.size());
  for (std::size_t ring = 0; ring < elevation_deg_by_ring.size(); ++ring) {
    model_scan->points.push_back(ringYawElevationPoint(
        5.0f,
        static_cast<float>(ring) * 3.0f,
        elevation_deg_by_ring[ring],
        static_cast<std::uint16_t>(ring),
        stamp + 1.0e-4 * static_cast<double>(ring),
        128U * static_cast<std::uint32_t>(ring)));
  }
  model_scan->width = static_cast<std::uint32_t>(model_scan->points.size());
  model_scan->height = 1U;
  model_scan->is_dense = false;

  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  filter.update(model_scan, I, origin, I, stamp);
}

void expectCachedLookupMatchesLinear(
    const dlio::MDetectorFilter& filter,
    const std::vector<float>& elevation_deg_samples) {
  std::vector<float> elevation_rad_samples;
  elevation_rad_samples.reserve(elevation_deg_samples.size());
  for (const float elevation_deg : elevation_deg_samples) {
    elevation_rad_samples.push_back(
        elevation_deg * static_cast<float>(M_PI) / 180.0f);
  }
  ASSERT_TRUE(filter.debugRingLookupMatchesLinear(elevation_rad_samples));
  for (const float elevation_rad : elevation_rad_samples) {
    EXPECT_EQ(filter.debugNearestRingForElevationCached(elevation_rad),
              filter.debugNearestRingForElevationLinear(elevation_rad));
  }
}

}  // namespace

TEST(MDetectorFilter, RingLookupEmptyModelFallsBackToNoRing) {
  dlio::MDetectorFilter filter(config());

  EXPECT_EQ(filter.debugRingLookupCacheSize(), 0U);
  EXPECT_EQ(filter.debugNearestRingForElevationCached(0.0f), -1);
  EXPECT_EQ(filter.debugNearestRingForElevationLinear(0.0f), -1);
}

TEST(MDetectorFilter, RingLookupCacheMatchesLinearForMonotonicModel) {
  dlio::MDetectorFilter filter(config());
  loadRingElevationModel(filter, {-16.0f, -10.0f, -5.0f, 0.0f,
                                  4.0f, 8.0f, 12.0f, 16.0f});

  EXPECT_EQ(filter.debugRingLookupCacheSize(), 8U);
  expectCachedLookupMatchesLinear(
      filter, {-20.0f, -16.0f, -12.5f, -2.0f, 2.0f, 6.0f, 14.0f, 20.0f});
}

TEST(MDetectorFilter, RingLookupCacheMatchesLinearForNonMonotonicRingOrder) {
  dlio::MDetectorFilter filter(config());
  loadRingElevationModel(filter, {6.0f, -14.0f, 18.0f, -4.0f,
                                  10.0f, -9.0f, 2.0f, 14.0f});

  EXPECT_EQ(filter.debugRingLookupCacheSize(), 8U);
  expectCachedLookupMatchesLinear(
      filter, {-18.0f, -11.0f, -6.5f, -1.0f, 4.0f, 8.0f, 12.5f, 20.0f});
}

TEST(MDetectorFilter, RingLookupCacheMatchesLinearForDuplicateElevations) {
  dlio::MDetectorFilter filter(config());
  loadRingElevationModel(filter, {-10.0f, -10.0f, 0.0f, 0.0f,
                                  10.0f, 10.0f, 20.0f, 20.0f});

  EXPECT_EQ(filter.debugRingLookupCacheSize(), 8U);
  expectCachedLookupMatchesLinear(
      filter, {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f, 15.0f, 20.0f});
  EXPECT_EQ(filter.debugNearestRingForElevationCached(
                -10.0f * static_cast<float>(M_PI) / 180.0f),
            0);
  EXPECT_EQ(filter.debugNearestRingForElevationCached(
                0.0f * static_cast<float>(M_PI) / 180.0f),
            2);
}

TEST(MDetectorFilter, RingLookupCacheRebuildsAfterModelUpdate) {
  dlio::MDetectorFilter filter(config());
  loadRingElevationModel(filter, {-12.0f, -4.0f, 4.0f, 12.0f}, 1.0);
  expectCachedLookupMatchesLinear(filter, {-10.0f, -2.0f, 2.0f, 10.0f});

  loadRingElevationModel(filter, {-2.0f, 6.0f, -8.0f, 18.0f}, 1.1);

  EXPECT_EQ(filter.debugRingLookupCacheSize(), 4U);
  expectCachedLookupMatchesLinear(filter, {-10.0f, -4.0f, 0.0f, 8.0f, 16.0f});
}

TEST(MDetectorFilter, RingLookupCacheClearsOnReset) {
  dlio::MDetectorFilter filter(config());
  loadRingElevationModel(filter, {-12.0f, -4.0f, 4.0f, 12.0f}, 1.0);
  EXPECT_GT(filter.debugRingLookupCacheSize(), 0U);

  filter.reset();

  EXPECT_EQ(filter.debugRingLookupCacheSize(), 0U);
  EXPECT_EQ(filter.debugNearestRingForElevationCached(0.0f), -1);
  EXPECT_EQ(filter.debugNearestRingForElevationLinear(0.0f), -1);
}

TEST(MDetectorFilter, CachedRingLookupDoesNotChangeEndToEndStats) {
  dlio::MDetectorFilter cached_filter(config());
  dlio::MDetectorFilter lazy_filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  cached_filter.update(farStatic(0.0f, 10), I, origin, I, 1.0);
  lazy_filter.update(farStatic(0.0f, 10), I, origin, I, 1.0);
  expectCachedLookupMatchesLinear(cached_filter, {-3.0f, 0.0f, 3.0f});

  auto cached = cached_filter.update(foregroundCluster(0.0f, 10), I, origin, I, 1.1);
  auto lazy = lazy_filter.update(foregroundCluster(0.0f, 10), I, origin, I, 1.1);

  EXPECT_EQ(cached.stats.case1_points, lazy.stats.case1_points);
  EXPECT_EQ(cached.stats.case2_points, lazy.stats.case2_points);
  EXPECT_EQ(cached.stats.case3_points, lazy.stats.case3_points);
  EXPECT_EQ(cached.stats.seed_points, lazy.stats.seed_points);
  EXPECT_EQ(cached.stats.cluster_count, lazy.stats.cluster_count);
  EXPECT_EQ(cached.stats.dynamic_removed, lazy.stats.dynamic_removed);
  EXPECT_EQ(cached.stats.mapping_kept, lazy.stats.mapping_kept);
  EXPECT_EQ(cached.keyframe_cloud->size(), lazy.keyframe_cloud->size());
}

TEST(MDetectorFilter, DisabledModeIsPassThrough) {
  auto cfg = config();
  cfg.enabled = false;
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto scan = cloud({point(1.0f, 0.0f, 0.0f), point(2.0f, 0.0f, 0.0f)});

  auto reg = filter.filterRegistration(scan, origin, 100);
  EXPECT_EQ(reg.cloud->size(), scan->size());
  EXPECT_EQ(reg.dynamic_points->size(), 0U);
  auto mapped = filter.update(scan, I, origin, I, 0.0);
  EXPECT_EQ(mapped.keyframe_cloud->size(), scan->size());
  EXPECT_EQ(mapped.dynamic_points_map->size(), 0U);
  EXPECT_EQ(filter.voxelCount(), 0U);
}

TEST(MDetectorFilter, Jt128ProjectionUsesRingAndActualYawNotRawColumn) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto history = farStatic(0.0f, 9);
  history->push_back(ringYawPoint(6.0f, 40.0f, 10, 1.01, 128U * 200U));
  history->push_back(ringYawPoint(6.0f, 80.0f, 11, 1.02, 128U * 300U));
  history->push_back(ringYawPoint(6.0f, 120.0f, 12, 1.03, 128U * 400U));
  auto foreground = foregroundCluster(0.0f, 9);
  foreground->push_back(ringYawPoint(6.0f, 40.0f, 10, 2.01, 128U * 200U));
  foreground->push_back(ringYawPoint(6.0f, 80.0f, 11, 2.02, 128U * 300U));
  foreground->push_back(ringYawPoint(6.0f, 120.0f, 12, 2.03, 128U * 400U));

  filter.update(history, I, origin, I, 1.0);
  auto mapped = filter.update(foreground, I, origin, I, 1.1);

  EXPECT_TRUE(mapped.stats.projection_native);
  EXPECT_FALSE(mapped.stats.projection_ring_fallback);
  EXPECT_GT(mapped.stats.case1_points + mapped.stats.case2_points, 0U);
  EXPECT_GT(mapped.dynamic_points_map->size(), 0U);
}

TEST(MDetectorFilter, RemovedCloudEmissionCanBeDisabledWithoutChangingStats) {
  dlio::MDetectorFilter emitted_filter(config());
  dlio::MDetectorFilter counted_filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  emitted_filter.update(farStatic(), I, origin, I, 1.0);
  counted_filter.update(farStatic(), I, origin, I, 1.0);

  auto emitted = emitted_filter.update(foregroundCluster(), I, origin, I, 1.1, true);
  auto counted = counted_filter.update(foregroundCluster(), I, origin, I, 1.1, false);

  EXPECT_GT(emitted.stats.dynamic_removed, 0U);
  EXPECT_GT(emitted.dynamic_points_map->size(), 0U);
  EXPECT_EQ(counted.dynamic_points_map->size(), 0U);
  EXPECT_EQ(counted.stats.dynamic_removed, emitted.stats.dynamic_removed);
  EXPECT_EQ(counted.stats.mapping_kept, emitted.stats.mapping_kept);
  EXPECT_EQ(counted.keyframe_cloud->size(), emitted.keyframe_cloud->size());
}

TEST(MDetectorFilter, ValidIndicesAndPreScanStatsMatchValidityDecision) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  PointType nan_point = ringYawPoint(4.0f, 30.0f, 7, 1.0, 128U * 30U);
  nan_point.x = std::numeric_limits<float>::quiet_NaN();
  auto scan = cloud({
      ringYawPoint(4.0f, 0.0f, 1, 1.0, 128U * 10U),
      ringYawPoint(4.5f, 10.0f, 3, 1.001, 128U * 11U),
      ringYawPoint(25.0f, 20.0f, 5, 1.002, 128U * 12U),
      nan_point,
  });

  filter.update(scan, I, origin, I, 1.0);

  EXPECT_EQ(filter.debugValidIndexCount(), 2U);
  EXPECT_EQ(filter.debugPreScanValidPointCount(), 2);
  EXPECT_EQ(filter.debugPreScanValidRingCount(), 2);
  EXPECT_EQ(filter.debugPreScanUniqueRingCount(), 2);
  EXPECT_EQ(filter.debugPreScanTimestampCount(), 2);
  EXPECT_TRUE(filter.debugPreScanRingSeen(1));
  EXPECT_TRUE(filter.debugPreScanRingSeen(3));
  EXPECT_FALSE(filter.debugPreScanRingSeen(5));
}

TEST(MDetectorFilter, CollapsedRingMetadataUsesFallbackProjection) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto collapsed = cloud({
      ringYawPoint(5.0f, 0.0f, 0, 1.0, 0U),
      ringYawPoint(5.0f, 20.0f, 0, 1.001, 128U),
      ringYawPoint(5.0f, 40.0f, 0, 1.002, 256U),
      ringYawPoint(5.0f, 60.0f, 0, 1.003, 384U),
  });

  auto mapped = filter.update(collapsed, I, origin, I, 1.0);

  EXPECT_TRUE(mapped.stats.projection_ring_fallback);
  EXPECT_FALSE(mapped.stats.projection_native);
}

TEST(MDetectorFilter, ProjectionPreScanFlagsMatchMetadataAvailability) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto scan = cloud({
      ringYawPoint(5.0f, 0.0f, 0, 0.0, 0U),
      ringYawPoint(5.0f, 20.0f, 1, 0.0, 128U),
      ringYawPoint(5.0f, 40.0f, 2, 0.0, 256U),
      ringYawPoint(5.0f, 60.0f, 3, 0.0, 384U),
  });

  auto mapped = filter.update(scan, I, origin, I, 1.0);

  EXPECT_EQ(filter.debugValidIndexCount(), 4U);
  EXPECT_EQ(filter.debugPreScanValidPointCount(), 4);
  EXPECT_EQ(filter.debugPreScanValidRingCount(), 4);
  EXPECT_EQ(filter.debugPreScanUniqueRingCount(), 4);
  EXPECT_EQ(filter.debugPreScanTimestampCount(), 0);
  EXPECT_FALSE(mapped.stats.projection_ring_fallback);
  EXPECT_TRUE(mapped.stats.projection_timestamp_fallback);
  EXPECT_TRUE(mapped.stats.projection_native);
}

TEST(MDetectorFilter, RingElevationModelUpdatesOnlyWithEnoughUniqueRings) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto collapsed = cloud({
      ringYawElevationPoint(5.0f, 0.0f, -2.0f, 0, 1.0, 0U),
      ringYawElevationPoint(5.0f, 20.0f, -1.0f, 0, 1.001, 128U),
      ringYawElevationPoint(5.0f, 40.0f, 1.0f, 0, 1.002, 256U),
      ringYawElevationPoint(5.0f, 60.0f, 2.0f, 0, 1.003, 384U),
  });

  filter.update(collapsed, I, origin, I, 1.0);
  EXPECT_EQ(filter.debugPreScanUniqueRingCount(), 1);
  EXPECT_EQ(filter.debugRingLookupCacheSize(), 0U);

  auto enough_rings = cloud({
      ringYawElevationPoint(5.0f, 0.0f, -6.0f, 0, 1.1, 0U),
      ringYawElevationPoint(5.0f, 20.0f, -2.0f, 1, 1.101, 128U),
      ringYawElevationPoint(5.0f, 40.0f, 2.0f, 2, 1.102, 256U),
      ringYawElevationPoint(5.0f, 60.0f, 6.0f, 3, 1.103, 384U),
  });

  filter.update(enough_rings, I, origin, I, 1.1);
  EXPECT_EQ(filter.debugPreScanUniqueRingCount(), 4);
  EXPECT_EQ(filter.debugRingLookupCacheSize(), 4U);
}

TEST(MDetectorFilter, MixedInvalidScanMatchesValidOnlyBehavior) {
  dlio::MDetectorFilter valid_filter(config());
  dlio::MDetectorFilter mixed_filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  PointType nan_point = ringYawPoint(4.0f, 30.0f, 12, 1.0, 128U * 30U);
  nan_point.x = std::numeric_limits<float>::quiet_NaN();
  auto valid_history = farStatic(0.0f, 10);
  auto mixed_history = cloud({
      (*valid_history)[0],
      ringYawPoint(25.0f, 80.0f, 11, 1.01, 128U * 80U),
      (*valid_history)[1],
      nan_point,
      (*valid_history)[2],
  });
  auto valid_foreground = foregroundCluster(0.0f, 10);
  PointType foreground_nan = nan_point;
  foreground_nan.timestamp = 2.5;
  auto mixed_foreground = cloud({
      (*valid_foreground)[0],
      foreground_nan,
      (*valid_foreground)[1],
      ringYawPoint(30.0f, 120.0f, 13, 2.01, 128U * 120U),
      (*valid_foreground)[2],
      (*valid_foreground)[3],
  });

  valid_filter.update(valid_history, I, origin, I, 1.0);
  mixed_filter.update(mixed_history, I, origin, I, 1.0);
  auto valid = valid_filter.update(valid_foreground, I, origin, I, 1.1, false);
  auto mixed = mixed_filter.update(mixed_foreground, I, origin, I, 1.1, false);

  EXPECT_EQ(mixed_filter.debugValidIndexCount(), valid_filter.debugValidIndexCount());
  EXPECT_EQ(mixed_filter.debugPreScanValidPointCount(),
            valid_filter.debugPreScanValidPointCount());
  EXPECT_EQ(mixed_filter.debugPreScanValidRingCount(),
            valid_filter.debugPreScanValidRingCount());
  EXPECT_EQ(mixed_filter.debugPreScanUniqueRingCount(),
            valid_filter.debugPreScanUniqueRingCount());
  EXPECT_EQ(mixed_filter.debugPreScanTimestampCount(),
            valid_filter.debugPreScanTimestampCount());
  EXPECT_EQ(mixed.stats.projection_ring_fallback, valid.stats.projection_ring_fallback);
  EXPECT_EQ(mixed.stats.projection_timestamp_fallback,
            valid.stats.projection_timestamp_fallback);
  EXPECT_EQ(mixed.stats.projection_native, valid.stats.projection_native);
  EXPECT_EQ(mixed.stats.case1_points, valid.stats.case1_points);
  EXPECT_EQ(mixed.stats.case2_points, valid.stats.case2_points);
  EXPECT_EQ(mixed.stats.case3_points, valid.stats.case3_points);
  EXPECT_EQ(mixed.stats.seed_points, valid.stats.seed_points);
  EXPECT_EQ(mixed.stats.cluster_count, valid.stats.cluster_count);
  EXPECT_EQ(mixed.stats.cluster_points, valid.stats.cluster_points);
  EXPECT_EQ(mixed.stats.dynamic_removed, valid.stats.dynamic_removed);
  EXPECT_EQ(mixed.stats.mapping_kept, valid.stats.mapping_kept + 1U);
  EXPECT_EQ(mixed.keyframe_cloud->size(), valid.keyframe_cloud->size() + 1U);
  EXPECT_EQ(mixed.dynamic_points_map->size(), 0U);
  EXPECT_EQ(mixed_filter.debugPublishableIndexCount(),
            mixed_filter.debugValidIndexCount() + 1U);
  EXPECT_EQ(mixed_filter.debugLastProjectionTouchedCellCount(),
            valid_filter.debugLastProjectionTouchedCellCount());
  EXPECT_EQ(mixed_filter.debugLastHistoryValidCellCount(),
            valid_filter.debugLastHistoryValidCellCount());
}

TEST(MDetectorFilter, OutOfRangeFinitePointsDoNotAffectValidOnlyBehavior) {
  auto cfg = config();
  cfg.max_range = 6.0;
  dlio::MDetectorFilter valid_filter(cfg);
  dlio::MDetectorFilter mixed_filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  auto valid_history = farStatic(0.0f, 10);
  auto mixed_history = cloud({
      (*valid_history)[0],
      ringYawPoint(12.0f, 90.0f, 20, 1.01, 128U * 90U),
      (*valid_history)[1],
      ringYawPoint(14.0f, 130.0f, 21, 1.02, 128U * 130U),
      (*valid_history)[2],
  });
  auto valid_foreground = foregroundCluster(0.0f, 10);
  auto mixed_foreground = cloud({
      (*valid_foreground)[0],
      ringYawPoint(12.0f, 100.0f, 22, 2.01, 128U * 100U),
      (*valid_foreground)[1],
      (*valid_foreground)[2],
      ringYawPoint(15.0f, 140.0f, 23, 2.02, 128U * 140U),
      (*valid_foreground)[3],
  });

  valid_filter.update(valid_history, I, origin, I, 1.0);
  mixed_filter.update(mixed_history, I, origin, I, 1.0);
  auto valid = valid_filter.update(valid_foreground, I, origin, I, 1.1, false);
  auto mixed = mixed_filter.update(mixed_foreground, I, origin, I, 1.1, false);

  EXPECT_EQ(mixed_filter.debugValidIndexCount(), valid_filter.debugValidIndexCount());
  EXPECT_EQ(mixed_filter.debugPreScanValidPointCount(),
            valid_filter.debugPreScanValidPointCount());
  EXPECT_EQ(mixed_filter.debugPreScanValidRingCount(),
            valid_filter.debugPreScanValidRingCount());
  EXPECT_EQ(mixed_filter.debugPreScanUniqueRingCount(),
            valid_filter.debugPreScanUniqueRingCount());
  EXPECT_EQ(mixed_filter.debugPreScanTimestampCount(),
            valid_filter.debugPreScanTimestampCount());
  EXPECT_EQ(mixed.stats.case1_points, valid.stats.case1_points);
  EXPECT_EQ(mixed.stats.case2_points, valid.stats.case2_points);
  EXPECT_EQ(mixed.stats.case3_points, valid.stats.case3_points);
  EXPECT_EQ(mixed.stats.seed_points, valid.stats.seed_points);
  EXPECT_EQ(mixed.stats.cluster_count, valid.stats.cluster_count);
  EXPECT_EQ(mixed.stats.cluster_points, valid.stats.cluster_points);
  EXPECT_EQ(mixed.stats.dynamic_removed, valid.stats.dynamic_removed);
  EXPECT_EQ(mixed.stats.mapping_kept, valid.stats.mapping_kept + 2U);
  EXPECT_EQ(mixed.keyframe_cloud->size(), valid.keyframe_cloud->size() + 2U);
  EXPECT_EQ(mixed.dynamic_points_map->size(), valid.dynamic_points_map->size());
  EXPECT_EQ(mixed_filter.debugPublishableIndexCount(),
            mixed_filter.debugValidIndexCount() + 2U);
  EXPECT_EQ(mixed_filter.debugLastProjectionTouchedCellCount(),
            valid_filter.debugLastProjectionTouchedCellCount());
  EXPECT_EQ(mixed_filter.debugLastHistoryValidCellCount(),
            valid_filter.debugLastHistoryValidCellCount());
}

TEST(MDetectorFilter, OutOfRangeFiniteStaticPointsPassThroughCleanMapOnly) {
  auto cfg = config();
  cfg.max_range = 6.0;
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  auto scan = cloud({
      ringYawPoint(4.0f, 0.0f, 5, 1.0, 128U * 10U),
      ringYawPoint(12.0f, 45.0f, 6, 1.001, 128U * 45U),
      ringYawPoint(14.0f, 90.0f, 7, 1.002, 128U * 90U),
  });

  filter.update(scan, I, origin, I, 1.0, true);
  auto mapped = filter.update(scan, I, origin, I, 1.1, true);

  EXPECT_EQ(filter.debugValidIndexCount(), 1U);
  EXPECT_EQ(filter.debugPublishableIndexCount(), 3U);
  EXPECT_EQ(filter.debugPreScanValidPointCount(), 1);
  EXPECT_EQ(filter.debugLastProjectionTouchedCellCount(), 1U);
  EXPECT_EQ(filter.debugLastHistoryValidCellCount(), 1U);
  EXPECT_EQ(mapped.keyframe_cloud->size(), 3U);
  EXPECT_EQ(mapped.stats.mapping_kept, 3U);
  EXPECT_EQ(mapped.dynamic_points_map->size(), 0U);
  EXPECT_EQ(mapped.stats.dynamic_removed, 0U);
  EXPECT_EQ(filter.voxelCount(), 1U);
}

TEST(MDetectorFilter, OutOfRangeFinitePointsPassThroughDuringWarmup) {
  auto cfg = config();
  cfg.max_range = 6.0;
  cfg.warmup_scans = 5;
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  auto scan = cloud({
      ringYawPoint(4.0f, 0.0f, 5, 1.0, 128U * 10U),
      ringYawPoint(12.0f, 45.0f, 6, 1.001, 128U * 45U),
      ringYawPoint(14.0f, 90.0f, 7, 1.002, 128U * 90U),
  });

  auto mapped = filter.update(scan, I, origin, I, 1.0, true);

  EXPECT_TRUE(mapped.stats.warmup);
  EXPECT_EQ(filter.debugValidIndexCount(), 1U);
  EXPECT_EQ(filter.debugPublishableIndexCount(), 3U);
  EXPECT_EQ(mapped.keyframe_cloud->size(), 2U);
  EXPECT_EQ(mapped.stats.mapping_kept, 2U);
  EXPECT_EQ(mapped.dynamic_points_map->size(), 0U);
  EXPECT_EQ(mapped.stats.dynamic_removed, 0U);
  EXPECT_EQ(filter.voxelCount(), 1U);
}

TEST(MDetectorFilter, InRangeCandidatesAreNotRemovedDuringWarmup) {
  auto cfg = config();
  cfg.warmup_scans = 5;
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  filter.update(farStatic(), I, origin, I, 1.0);
  auto mapped = filter.update(foregroundCluster(), I, origin, I, 1.1, true);

  EXPECT_TRUE(mapped.stats.warmup);
  EXPECT_EQ(mapped.stats.dynamic_removed, 0U);
  EXPECT_EQ(mapped.dynamic_points_map->size(), 0U);
}

TEST(MDetectorFilter, UnsafeVoxelKeyPointPassesThroughWithoutDetectorState) {
  auto cfg = config();
  cfg.min_range = 0.0;
  cfg.max_range = 20.0;
  cfg.warmup_scans = 0;
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin(1.0e9f, 0.0f, 0.0f);
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto scan = cloud({point(1.0e9f, 0.0f, 0.0f)});

  auto mapped = filter.update(scan, I, origin, I, 1.0, true);

  EXPECT_EQ(filter.debugPublishableIndexCount(), 1U);
  EXPECT_EQ(filter.debugValidIndexCount(), 0U);
  EXPECT_EQ(filter.debugPreScanValidPointCount(), 0);
  EXPECT_EQ(filter.debugLastProjectionTouchedCellCount(), 0U);
  EXPECT_EQ(filter.debugLastHistoryValidCellCount(), 0U);
  EXPECT_EQ(filter.voxelCount(), 0U);
  EXPECT_EQ(mapped.keyframe_cloud->size(), 1U);
  EXPECT_EQ(mapped.dynamic_points_map->size(), 0U);
  EXPECT_EQ(mapped.stats.dynamic_removed, 0U);
}

TEST(MDetectorFilter, HugeProjectionDimensionsAreClampedAndDoNotOverflowIndexing) {
  auto cfg = config();
  cfg.projection_rows = std::numeric_limits<int>::max();
  cfg.projection_cols = std::numeric_limits<int>::max();
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  auto mapped = filter.update(farStatic(), I, origin, I, 1.0, true);

  EXPECT_LE(filter.config().projection_rows, 4096);
  EXPECT_LE(filter.config().projection_cols, 4096);
  EXPECT_EQ(filter.debugLastProjectionTouchedCellCount(), 0U);
  EXPECT_EQ(filter.debugLastHistoryValidCellCount(), 0U);
  EXPECT_FALSE(mapped.stats.projection_native);
  EXPECT_EQ(mapped.dynamic_points_map->size(), 0U);
}

TEST(MDetectorFilter, HistoryFrameStoresOnlyTouchedProjectionCells) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto scan = cloud({
      ringYawPoint(5.0f, 0.0f, 2, 1.0, 0U),
      ringYawPoint(4.0f, 0.0f, 2, 1.001, 128U),
      ringYawPoint(5.0f, 20.0f, 3, 1.002, 256U),
      ringYawPoint(5.0f, 40.0f, 4, 1.003, 384U),
      ringYawPoint(5.0f, 60.0f, 5, 1.004, 512U),
  });

  filter.update(scan, I, origin, I, 1.0);

  EXPECT_EQ(filter.debugValidIndexCount(), scan->size());
  EXPECT_EQ(filter.debugLastProjectionTouchedCellCount(), 4U);
  EXPECT_EQ(filter.debugLastHistoryValidCellCount(), 4U);
  EXPECT_TRUE(filter.debugLastHistoryCellValid(2, projectionCol(0.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(3, projectionCol(20.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(4, projectionCol(40.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(5, projectionCol(60.0f)));
}

TEST(MDetectorFilter, HistoryFrameDoesNotLeakCellsAcrossScans) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto first = cloud({
      ringYawPoint(5.0f, 0.0f, 2, 1.0, 0U),
      ringYawPoint(5.0f, 20.0f, 3, 1.001, 128U),
      ringYawPoint(5.0f, 40.0f, 4, 1.002, 256U),
      ringYawPoint(5.0f, 60.0f, 5, 1.003, 384U),
  });
  auto second = cloud({
      ringYawPoint(5.0f, 120.0f, 6, 1.1, 512U),
      ringYawPoint(5.0f, 140.0f, 7, 1.101, 640U),
      ringYawPoint(5.0f, 160.0f, 8, 1.102, 768U),
      ringYawPoint(5.0f, 180.0f, 9, 1.103, 896U),
  });

  filter.update(first, I, origin, I, 1.0);
  EXPECT_EQ(filter.debugLastProjectionTouchedCellCount(), 4U);
  EXPECT_EQ(filter.debugLastHistoryValidCellCount(), 4U);
  EXPECT_TRUE(filter.debugLastHistoryCellValid(2, projectionCol(0.0f)));

  filter.update(second, I, origin, I, 1.1);

  EXPECT_EQ(filter.debugLastProjectionTouchedCellCount(), 4U);
  EXPECT_EQ(filter.debugLastHistoryValidCellCount(), 4U);
  EXPECT_FALSE(filter.debugLastHistoryCellValid(2, projectionCol(0.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(6, projectionCol(120.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(7, projectionCol(140.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(8, projectionCol(160.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(9, projectionCol(180.0f)));
}

TEST(MDetectorFilter, ReusedHistorySlotDoesNotLeakOldCells) {
  auto cfg = config();
  cfg.max_history_frames = 2;
  cfg.history_duration = 10.0;
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto first = cloud({
      ringYawPoint(5.0f, 0.0f, 2, 1.0, 0U),
      ringYawPoint(5.0f, 20.0f, 3, 1.001, 128U),
      ringYawPoint(5.0f, 40.0f, 4, 1.002, 256U),
      ringYawPoint(5.0f, 60.0f, 5, 1.003, 384U),
  });
  auto second = cloud({
      ringYawPoint(5.0f, 120.0f, 6, 1.1, 512U),
      ringYawPoint(5.0f, 140.0f, 7, 1.101, 640U),
      ringYawPoint(5.0f, 160.0f, 8, 1.102, 768U),
      ringYawPoint(5.0f, 180.0f, 9, 1.103, 896U),
  });
  auto third = cloud({
      ringYawPoint(5.0f, 240.0f, 10, 1.2, 1024U),
      ringYawPoint(5.0f, 260.0f, 11, 1.201, 1152U),
      ringYawPoint(5.0f, 280.0f, 12, 1.202, 1280U),
      ringYawPoint(5.0f, 300.0f, 13, 1.203, 1408U),
  });

  filter.update(first, I, origin, I, 1.0);
  filter.update(second, I, origin, I, 1.1);
  filter.update(third, I, origin, I, 1.2);

  EXPECT_EQ(filter.debugHistorySlotCount(), 2U);
  EXPECT_EQ(filter.debugHistoryActiveFrameCount(), 2U);
  EXPECT_EQ(filter.debugLastHistoryValidCellCount(), 4U);
  EXPECT_FALSE(filter.debugLastHistoryCellValid(2, projectionCol(0.0f)));
  EXPECT_FALSE(filter.debugLastHistoryCellValid(6, projectionCol(120.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(10, projectionCol(240.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(11, projectionCol(260.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(12, projectionCol(280.0f)));
  EXPECT_TRUE(filter.debugLastHistoryCellValid(13, projectionCol(300.0f)));
}

TEST(MDetectorFilter, StaticSupportCacheReflectsPreUpdateVoxelState) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto wall = cloud({
      ringYawPoint(4.0f, 0.0f, 8, 1.0, 128U * 10U),
      ringYawPoint(4.2f, 0.4f, 8, 1.001, 128U * 11U),
      ringYawPoint(4.4f, -0.4f, 8, 1.002, 128U * 12U),
  });

  filter.update(wall, I, origin, I, 1.0);
  filter.update(wall, I, origin, I, 1.1);
  filter.update(wall, I, origin, I, 1.2);

  EXPECT_EQ(filter.debugValidIndexCount(), wall->size());
  EXPECT_TRUE(filter.debugStaticSupportedPoint(0));
  EXPECT_TRUE(filter.debugStaticSupportedPoint(1));
  EXPECT_TRUE(filter.debugStaticSupportedPoint(2));
}

TEST(MDetectorFilter, ActiveTrackCacheReflectsPreUpdateTrackMask) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  filter.update(farStatic(), I, origin, I, 1.0);
  auto first = filter.update(foregroundCluster(), I, origin, I, 1.1);
  EXPECT_GT(first.stats.confirmed_track_count, 0U);
  auto second = filter.update(foregroundCluster(), I, origin, I, 1.2);

  EXPECT_GT(second.stats.track_removed_points, 0U);
  EXPECT_EQ(filter.debugValidIndexCount(), foregroundCluster()->size());
  EXPECT_TRUE(filter.debugActiveTrackPoint(0));
  EXPECT_TRUE(filter.debugActiveTrackPoint(1));
}

TEST(MDetectorFilter, SeedIndicesMatchSeedMaskAndStats) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  filter.update(farStatic(), I, origin, I, 1.0);
  auto mapped = filter.update(foregroundCluster(), I, origin, I, 1.1);

  EXPECT_GT(mapped.stats.seed_points, 0U);
  EXPECT_EQ(filter.debugSeedIndexCount(), filter.debugSeedMaskCount());
  EXPECT_EQ(mapped.stats.seed_points, filter.debugSeedIndexCount());
  EXPECT_GE(mapped.stats.cluster_points, foregroundCluster()->size());
}

TEST(MDetectorFilter, MotionCompensatedHistoryDoesNotFlagStaticRobotMotion) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto static_point = cloud({point(5.0f, 0.0f, 0.0f),
                             point(5.2f, 0.1f, 0.0f),
                             point(5.4f, -0.1f, 0.0f)});

  filter.update(static_point, I, origin, I, 1.0);
  auto moved = filter.update(static_point, I, origin, translation(1.0f, 0.0f, 0.0f), 1.1);

  EXPECT_EQ(moved.dynamic_points_map->size(), 0U);
  EXPECT_EQ(moved.stats.case1_points, 0U);
  EXPECT_EQ(moved.stats.case2_points, 0U);
}

TEST(MDetectorFilter, RepeatedVotesRequiredBeforeMovingEventRemoval) {
  auto cfg = config();
  cfg.min_history_votes = 2;
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  filter.update(farStatic(), I, origin, I, 1.0);
  auto first = filter.update(foregroundCluster(), I, origin, I, 1.1);
  EXPECT_EQ(first.dynamic_points_map->size(), 0U);

  filter.update(farStatic(), I, origin, I, 1.2);
  auto second = filter.update(foregroundCluster(), I, origin, I, 1.3);
  EXPECT_GT(second.dynamic_points_map->size(), 0U);
}

TEST(MDetectorFilter, FullClusterRemovalIncludesNonSeedAdjacentPoints) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  filter.update(farStatic(0.0f, 10), I, origin, I, 1.0);

  auto object = foregroundCluster(0.0f, 10);
  object->push_back(ringYawPoint(2.12f, 1.2f, 10, 2.004, 128U * 704U));
  auto mapped = filter.update(object, I, origin, I, 1.1);

  EXPECT_GE(mapped.stats.cluster_points, object->size());
  EXPECT_GE(mapped.dynamic_points_map->size(), object->size());
}

TEST(MDetectorFilter, StaticSupportedWallVetoesDynamicRemoval) {
  auto cfg = config();
  cfg.static_veto_ratio = 0.10;
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto wall = cloud({
      point(3.0f, -0.4f, 0.0f),
      point(3.0f, 0.0f, 0.3f),
      point(3.0f, 0.4f, 0.6f),
      point(3.0f, 0.8f, 0.9f),
  });

  filter.update(wall, I, origin, I, 1.0);
  filter.update(wall, I, origin, I, 1.1);
  auto repeated = filter.update(wall, I, origin, I, 1.2);

  EXPECT_GT(repeated.keyframe_cloud->size(), 0U);
  EXPECT_EQ(repeated.dynamic_points_map->size(), 0U);
}

TEST(MDetectorFilter, LowSeedRatioClusterDoesNotRemoveOrCreateTrack) {
  auto cfg = config();
  cfg.min_history_votes = 2;
  cfg.min_cluster_points = 6;
  cfg.min_track_cluster_points = 6;
  cfg.track_confirm_hits = 2;
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
  auto sparse_history = cloud({
      ringYawPoint(5.0f, 0.0f, 8, 1.0, 128U * 10U),
      ringYawPoint(5.0f, 0.4f, 8, 1.001, 128U * 11U),
  });

  filter.update(sparse_history, I, origin, I, 1.0);
  filter.update(sparse_history, I, origin, I, 1.1);
  auto mapped = filter.update(lowSeedRatioCluster(8), I, origin, I, 1.2);

  EXPECT_EQ(mapped.stats.cluster_count, 0U);
  EXPECT_EQ(mapped.dynamic_points_map->size(), 0U);
  EXPECT_EQ(mapped.stats.confirmed_track_count, 0U);
  EXPECT_EQ(mapped.stats.tentative_track_count, 0U);
}

TEST(MDetectorFilter, ConfirmedTrackSuppressesFutureAndExpires) {
  auto cfg = config();
  cfg.track_ttl_scans = 1;
  dlio::MDetectorFilter filter(cfg);
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  filter.update(farStatic(), I, origin, I, 1.0);
  auto first = filter.update(foregroundCluster(), I, origin, I, 1.1);
  EXPECT_GT(first.stats.confirmed_track_count, 0U);

  auto stopped = filter.update(foregroundCluster(), I, origin, I, 1.2);
  EXPECT_GT(stopped.stats.track_removed_points, 0U);

  filter.update(farStatic(), I, origin, I, 1.3);
  filter.update(farStatic(), I, origin, I, 1.4);
  auto expired = filter.update(farStatic(), I, origin, I, 1.5);
  EXPECT_EQ(expired.stats.confirmed_track_count, 0U);
}

TEST(MDetectorFilter, RegistrationFallbackRestoresUnfilteredCloud) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  filter.update(farStatic(), I, origin, I, 1.0);
  filter.update(foregroundCluster(), I, origin, I, 1.1);

  auto reg = filter.filterRegistration(foregroundCluster(), origin, 100);
  EXPECT_TRUE(reg.stats.registration_fallback);
  EXPECT_EQ(reg.cloud->size(), foregroundCluster()->size());
  EXPECT_EQ(reg.kept_indices.size(), foregroundCluster()->size());
}

TEST(MDetectorFilter, ConfigureClampsThresholds) {
  auto cfg = config();
  cfg.min_track_cluster_points = 0;
  cfg.projection_cols = 1;
  dlio::MDetectorFilter filter(cfg);
  EXPECT_EQ(filter.config().min_track_cluster_points, 1);
  EXPECT_GE(filter.config().projection_cols, 16);
}

TEST(MDetectorFilter, ResetClearsState) {
  dlio::MDetectorFilter filter(config());
  const Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

  filter.update(farStatic(), I, origin, I, 1.0);
  filter.update(foregroundCluster(), I, origin, I, 1.1);
  EXPECT_GT(filter.voxelCount(), 0U);
  filter.reset();
  EXPECT_EQ(filter.voxelCount(), 0U);
  EXPECT_EQ(filter.debugHistorySlotCount(), 0U);
  EXPECT_EQ(filter.debugHistoryActiveFrameCount(), 0U);
  EXPECT_EQ(filter.debugSeedIndexCount(), 0U);
  EXPECT_EQ(filter.debugSeedMaskCount(), 0U);
  auto reg = filter.filterRegistration(foregroundCluster(), origin, 0);
  EXPECT_EQ(reg.dynamic_points->size(), 0U);
}
