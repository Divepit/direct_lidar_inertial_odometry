/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 ***********************************************************/

#include "dlio/m_detector_filter.h"

#include <cassert>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>

namespace {

constexpr int kMaxProjectionRows = 4096;
constexpr int kMaxProjectionCols = 4096;
constexpr std::size_t kMaxProjectionCells = 4'000'000U;

int popcount64(std::uint64_t value) {
#if defined(__GNUG__) || defined(__clang__)
  return __builtin_popcountll(value);
#else
  int count = 0;
  while (value != 0U) {
    value &= (value - 1U);
    ++count;
  }
  return count;
#endif
}

PointType makePointFromVector(const PointType& source, const Eigen::Vector3f& point) {
  PointType out = source;
  out.x = point.x();
  out.y = point.y();
  out.z = point.z();
  return out;
}

void finalizeCloud(const pcl::PointCloud<PointType>::Ptr& cloud) {
  cloud->width = static_cast<std::uint32_t>(cloud->points.size());
  cloud->height = 1U;
  cloud->is_dense = false;
}

float normalizeAnglePositive(float angle) {
  constexpr float kTwoPi = 2.0f * static_cast<float>(M_PI);
  while (angle < 0.0f) {
    angle += kTwoPi;
  }
  while (angle >= kTwoPi) {
    angle -= kTwoPi;
  }
  return angle;
}

}  // namespace

namespace dlio {

std::size_t MDetectorFilter::VoxelKeyHash::operator()(const VoxelKey& key) const {
  const std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::int64_t>(key.x));
  const std::uint64_t y = static_cast<std::uint64_t>(static_cast<std::int64_t>(key.y));
  const std::uint64_t z = static_cast<std::uint64_t>(static_cast<std::int64_t>(key.z));
  std::uint64_t h = x * 73856093ULL;
  h ^= y * 19349663ULL;
  h ^= z * 83492791ULL;
  return static_cast<std::size_t>(h);
}

MDetectorFilter::MDetectorFilter() = default;

MDetectorFilter::MDetectorFilter(const Config& config) {
  configure(config);
}

void MDetectorFilter::configure(const Config& config) {
  config_ = config;
  config_.voxel_size = std::max(config_.voxel_size, 0.02);
  config_.min_range = std::max(config_.min_range, 0.0);
  config_.max_range = std::max(config_.max_range, config_.min_range + config_.voxel_size);
  config_.warmup_scans = std::max(config_.warmup_scans, 0);
  config_.static_score_threshold = std::max(config_.static_score_threshold, 1);
  config_.static_window_scans = std::max(config_.static_window_scans,
                                         config_.static_score_threshold);
  config_.static_window_scans = std::min(config_.static_window_scans, 63);
  config_.max_age_scans = std::max(config_.max_age_scans, 1);
  config_.local_radius = std::max(config_.local_radius, config_.max_range);
  config_.projection_rows = std::clamp(config_.projection_rows, 1, kMaxProjectionRows);
  config_.projection_cols = std::clamp(config_.projection_cols, 16, kMaxProjectionCols);
  config_.history_duration = std::max(config_.history_duration, 0.0);
  config_.max_history_frames = std::max(config_.max_history_frames, 1);
  config_.frame_duration = std::max(config_.frame_duration, 1.0e-3);
  config_.min_history_votes = std::max(config_.min_history_votes, 1);
  config_.case_depth_margin = std::max(config_.case_depth_margin, 0.02);
  config_.map_consistency_depth = std::max(config_.map_consistency_depth, 0.02);
  config_.min_cluster_points = std::max(config_.min_cluster_points, 1);
  config_.min_track_cluster_points = std::max(config_.min_track_cluster_points, 1);
  config_.max_cluster_extent = std::max(config_.max_cluster_extent, config_.voxel_size);
  config_.max_assoc_distance = std::max(config_.max_assoc_distance, config_.voxel_size);
  config_.track_confirm_hits = std::max(config_.track_confirm_hits, 1);
  config_.track_ttl_scans = std::max(config_.track_ttl_scans, 1);
  config_.static_veto_ratio = std::clamp(config_.static_veto_ratio, 0.0, 1.0);
  ring_elevation_sum_.assign(static_cast<std::size_t>(config_.projection_rows), 0.0f);
  ring_elevation_count_.assign(static_cast<std::size_t>(config_.projection_rows), 0);
  invalidateRingElevationLookupCache();
}

const MDetectorFilter::Config& MDetectorFilter::config() const {
  return config_;
}

void MDetectorFilter::reset() {
  voxels_.clear();
  history_slots_.clear();
  history_order_.clear();
  tracks_.clear();
  active_track_mask_.clear();
  std::fill(ring_elevation_sum_.begin(), ring_elevation_sum_.end(), 0.0f);
  std::fill(ring_elevation_count_.begin(), ring_elevation_count_.end(), 0);
  invalidateRingElevationLookupCache();
  scratch_ = Scratch{};
  scan_index_ = 0;
  next_track_id_ = 1;
}

bool MDetectorFilter::enabled() const {
  return config_.enabled && validConfig();
}

bool MDetectorFilter::warmedUp() const {
  return scan_index_ >= config_.warmup_scans &&
         history_order_.size() >= static_cast<std::size_t>(
             std::min(config_.min_history_votes, config_.max_history_frames));
}

std::size_t MDetectorFilter::voxelCount() const {
  return voxels_.size();
}

bool MDetectorFilter::validConfig() const {
  return config_.voxel_size > 0.0 &&
         config_.max_range > config_.min_range &&
         config_.projection_rows > 0 &&
         config_.projection_cols > 0;
}

bool MDetectorFilter::pointFinite(const PointType& point) const {
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

bool MDetectorFilter::pointInRange(
    const Eigen::Vector3f& point_map,
    const Eigen::Vector3f& sensor_origin_map) const {
  if (!point_map.allFinite() || !sensor_origin_map.allFinite()) {
    return false;
  }
  const float range = (point_map - sensor_origin_map).norm();
  return range >= static_cast<float>(config_.min_range) &&
	         range <= static_cast<float>(config_.max_range);
}

bool MDetectorFilter::tryKeyFromPoint(
    const Eigen::Vector3f& point_map,
    VoxelKey& key) const {
  if (!point_map.allFinite() || config_.voxel_size <= 0.0) {
    return false;
  }

  const double inv = 1.0 / config_.voxel_size;
  const auto convert_axis = [inv](float value, int& out) {
    const double scaled = std::floor(static_cast<double>(value) * inv);
    if (!std::isfinite(scaled) ||
        scaled < static_cast<double>(std::numeric_limits<int>::min()) ||
        scaled > static_cast<double>(std::numeric_limits<int>::max())) {
      return false;
    }
    out = static_cast<int>(scaled);
    return true;
  };

  VoxelKey candidate;
  if (!convert_axis(point_map.x(), candidate.x) ||
      !convert_axis(point_map.y(), candidate.y) ||
      !convert_axis(point_map.z(), candidate.z)) {
    return false;
  }

  key = candidate;
  return true;
}

MDetectorFilter::VoxelKey MDetectorFilter::keyFromPoint(
    const Eigen::Vector3f& point_map) const {
  VoxelKey key;
  (void)tryKeyFromPoint(point_map, key);
  return key;
}

bool MDetectorFilter::projectionIndex(
    int row,
    int col,
    int cols,
    std::size_t cell_count,
    std::size_t& index) const {
  if (row < 0 || col < 0 || cols <= 0 || col >= cols) {
    return false;
  }

  const std::size_t row_u = static_cast<std::size_t>(row);
  const std::size_t col_u = static_cast<std::size_t>(col);
  const std::size_t cols_u = static_cast<std::size_t>(cols);
  if (row_u > (std::numeric_limits<std::size_t>::max() - col_u) / cols_u) {
    return false;
  }

  const std::size_t candidate = row_u * cols_u + col_u;
  if (candidate >= cell_count) {
    return false;
  }

  index = candidate;
  return true;
}

std::uint64_t MDetectorFilter::staticWindowMask() const {
  const int window = std::min(config_.static_window_scans, 63);
  return (1ULL << window) - 1ULL;
}

int MDetectorFilter::staticHitCount(const VoxelState& voxel) const {
  if (voxel.hit_bits_scan < 0) {
    return 0;
  }
  const int shift = std::max(0, scan_index_ - voxel.hit_bits_scan);
  if (shift >= 64) {
    return 0;
  }
  return popcount64((voxel.hit_bits << shift) & staticWindowMask());
}

bool MDetectorFilter::voxelIsStatic(const VoxelState& voxel) const {
  return staticHitCount(voxel) >= config_.static_score_threshold;
}

bool MDetectorFilter::voxelIsDynamic(const VoxelState& voxel) const {
  return voxel.dynamic_score > 0 &&
         voxel.last_dynamic_scan >= 0 &&
         scan_index_ - voxel.last_dynamic_scan <= config_.track_ttl_scans;
}

bool MDetectorFilter::isStaticVoxel(const Eigen::Vector3f& point_map) const {
  VoxelKey key;
  if (!tryKeyFromPoint(point_map, key)) {
    return false;
  }
  const auto it = voxels_.find(key);
  return it != voxels_.end() && voxelIsStatic(it->second);
}

bool MDetectorFilter::isDynamicVoxel(const Eigen::Vector3f& point_map) const {
  VoxelKey key;
  if (!tryKeyFromPoint(point_map, key)) {
    return false;
  }
  const auto it = voxels_.find(key);
  return it != voxels_.end() && voxelIsDynamic(it->second);
}

void MDetectorFilter::markStatic(const VoxelKey& key) {
  VoxelState& voxel = voxels_[key];
  const int shift = voxel.hit_bits_scan < 0 ? 0 : scan_index_ - voxel.hit_bits_scan;
  if (shift >= 64) {
    voxel.hit_bits = 0;
  } else if (shift > 0) {
    voxel.hit_bits <<= shift;
  }
  voxel.hit_bits |= 1ULL;
  voxel.hit_bits &= staticWindowMask();
  voxel.hit_bits_scan = scan_index_;
  voxel.last_seen_scan = scan_index_;
  voxel.dynamic_score = std::max(0, voxel.dynamic_score - 1);
}

void MDetectorFilter::markDynamic(const VoxelKey& key) {
  VoxelState& voxel = voxels_[key];
  voxel.dynamic_score = std::min(voxel.dynamic_score + 1, 32);
  voxel.last_dynamic_scan = scan_index_;
  voxel.last_seen_scan = scan_index_;
}

void MDetectorFilter::pruneVoxels(const Eigen::Vector3f& sensor_origin_map) {
  for (auto it = voxels_.begin(); it != voxels_.end();) {
    const VoxelState& voxel = it->second;
    const bool stale = voxel.last_seen_scan >= 0 &&
        scan_index_ - voxel.last_seen_scan > config_.max_age_scans;
    const Eigen::Vector3f center(
        (static_cast<float>(it->first.x) + 0.5f) * static_cast<float>(config_.voxel_size),
        (static_cast<float>(it->first.y) + 0.5f) * static_cast<float>(config_.voxel_size),
        (static_cast<float>(it->first.z) + 0.5f) * static_cast<float>(config_.voxel_size));
    const bool outside_local =
        (center - sensor_origin_map).norm() >
        static_cast<float>(config_.local_radius + config_.max_range);
    if (stale || outside_local) {
      it = voxels_.erase(it);
    } else {
      ++it;
    }
  }
}

void MDetectorFilter::updateRingElevationModel(
    const pcl::PointCloud<PointType>::ConstPtr& cloud,
    const std::vector<Eigen::Vector3f>& points_lidar,
    const std::vector<std::size_t>& valid_indices,
    const ProjectionPreScanStats& pre_scan) {
  if (!cloud || !config_.projection_use_ring_field) {
    return;
  }
  if (pre_scan.unique_ring_count < std::min(4, config_.projection_rows)) {
    return;
  }
  bool model_updated = false;
  for (const std::size_t i : valid_indices) {
    if (i >= cloud->size() || i >= points_lidar.size()) {
      continue;
    }
    const PointType& source = (*cloud)[i];
    if (source.ring >= static_cast<std::uint16_t>(config_.projection_rows)) {
      continue;
    }
    const Eigen::Vector3f& p = points_lidar[i];
    const float horizontal = std::hypot(p.x(), p.y());
    const float elevation = std::atan2(p.z(), horizontal);
    const std::size_t ring = static_cast<std::size_t>(source.ring);
    ring_elevation_sum_[ring] += elevation;
    ring_elevation_count_[ring] = std::min(ring_elevation_count_[ring] + 1, 20000);
    model_updated = true;
    if (ring_elevation_count_[ring] == 20000) {
      ring_elevation_sum_[ring] *= 0.5f;
      ring_elevation_count_[ring] = 10000;
    }
  }
  if (model_updated) {
    invalidateRingElevationLookupCache();
  }
}

void MDetectorFilter::invalidateRingElevationLookupCache() {
  ring_elevation_lookup_dirty_ = true;
}

void MDetectorFilter::rebuildRingElevationLookupCache() const {
  if (!ring_elevation_lookup_dirty_) {
    return;
  }

  ring_elevation_lookup_.clear();
  ring_elevation_lookup_.reserve(ring_elevation_count_.size());
  for (std::size_t i = 0; i < ring_elevation_count_.size(); ++i) {
    if (ring_elevation_count_[i] <= 0) {
      continue;
    }
    const float mean = ring_elevation_sum_[i] /
        static_cast<float>(ring_elevation_count_[i]);
    if (!std::isfinite(mean)) {
      continue;
    }
    ring_elevation_lookup_.push_back(
        RingElevationEntry{mean, static_cast<int>(i)});
  }

  std::sort(
      ring_elevation_lookup_.begin(),
      ring_elevation_lookup_.end(),
      [](const RingElevationEntry& a, const RingElevationEntry& b) {
        if (a.elevation == b.elevation) {
          return a.ring < b.ring;
        }
        return a.elevation < b.elevation;
      });
  ring_elevation_lookup_dirty_ = false;
}

int MDetectorFilter::nearestRingForElevationLinear(float elevation) const {
  int best = -1;
  float best_error = std::numeric_limits<float>::max();
  for (std::size_t i = 0; i < ring_elevation_count_.size(); ++i) {
    if (ring_elevation_count_[i] <= 0) {
      continue;
    }
    const float mean = ring_elevation_sum_[i] /
        static_cast<float>(ring_elevation_count_[i]);
    const float error = std::abs(mean - elevation);
    if (error < best_error) {
      best_error = error;
      best = static_cast<int>(i);
    }
  }
  return best;
}

int MDetectorFilter::nearestRingForElevation(float elevation) const {
  rebuildRingElevationLookupCache();
  if (ring_elevation_lookup_.empty() || !std::isfinite(elevation)) {
    return -1;
  }

  int best = -1;
  float best_error = std::numeric_limits<float>::max();
  const auto consider = [&](const RingElevationEntry& entry) {
    const float error = std::abs(entry.elevation - elevation);
    if (!std::isfinite(error)) {
      return;
    }
    if (best < 0 || error < best_error ||
        (error == best_error && entry.ring < best)) {
      best = entry.ring;
      best_error = error;
    }
  };

  const auto lower = std::lower_bound(
      ring_elevation_lookup_.begin(),
      ring_elevation_lookup_.end(),
      elevation,
      [](const RingElevationEntry& entry, float value) {
        return entry.elevation < value;
      });

  const auto consider_equal_elevation_group =
      [&](std::vector<RingElevationEntry>::const_iterator it) {
        if (it == ring_elevation_lookup_.end()) {
          return;
        }
        const float group_elevation = it->elevation;
        while (it != ring_elevation_lookup_.begin() &&
               std::prev(it)->elevation == group_elevation) {
          --it;
        }
        for (; it != ring_elevation_lookup_.end() &&
               it->elevation == group_elevation; ++it) {
          consider(*it);
        }
      };

  if (lower != ring_elevation_lookup_.end()) {
    consider_equal_elevation_group(lower);
  }
  if (lower != ring_elevation_lookup_.begin()) {
    consider_equal_elevation_group(std::prev(lower));
  }

#ifndef NDEBUG
  assert(best == nearestRingForElevationLinear(elevation));
#endif

  return best;
}

bool MDetectorFilter::projectPoint(
    const Eigen::Vector3f& point_lidar,
    int source_ring,
    bool prefer_source_ring,
    int& row,
    int& col,
    float& range,
    float& yaw,
    float& elevation,
    bool& ring_fallback) const {
  row = -1;
  col = -1;
  range = point_lidar.norm();
  ring_fallback = false;
  if (!std::isfinite(range) || range <= 1.0e-3f) {
    return false;
  }

  constexpr float kTwoPi = 2.0f * static_cast<float>(M_PI);
  yaw = normalizeAnglePositive(std::atan2(point_lidar.x(), point_lidar.y()));
  const float horizontal = std::hypot(point_lidar.x(), point_lidar.y());
  elevation = std::atan2(point_lidar.z(), horizontal);
  col = static_cast<int>(std::lround(
      yaw / kTwoPi * static_cast<float>(config_.projection_cols)));
  if (col >= config_.projection_cols) {
    col -= config_.projection_cols;
  }
  col = std::clamp(col, 0, config_.projection_cols - 1);

  if (prefer_source_ring &&
      config_.projection_use_ring_field &&
      source_ring >= 0 &&
      source_ring < config_.projection_rows) {
    row = source_ring;
  } else {
    row = nearestRingForElevation(elevation);
    if (row < 0) {
      ring_fallback = true;
      const float normalized =
          (elevation + 0.5f * static_cast<float>(M_PI)) /
          static_cast<float>(M_PI);
      row = static_cast<int>(std::floor(
          normalized * static_cast<float>(config_.projection_rows)));
    }
  }
  row = std::clamp(row, 0, config_.projection_rows - 1);
  return true;
}

void MDetectorFilter::buildProjection(
    const pcl::PointCloud<PointType>::ConstPtr& cloud_map_before_correction,
    const std::vector<Eigen::Vector3f>& corrected_points,
    const std::vector<Eigen::Vector3f>& points_lidar,
    const std::vector<std::size_t>& valid_indices,
    const ProjectionPreScanStats& pre_scan,
    std::vector<PointMeta>& point_meta,
    ProjectionFrame& projection) const {
  if (!projection.touched_cell_indices.empty()) {
    for (const int idx : projection.touched_cell_indices) {
      if (idx >= 0 && static_cast<std::size_t>(idx) < projection.cells.size()) {
        projection.cells[static_cast<std::size_t>(idx)] = ProjectionCell{};
      }
    }
    projection.touched_cell_indices.clear();
  }
  projection.rows = config_.projection_rows;
  projection.cols = config_.projection_cols;
	  projection.used_ring_fallback = false;
	  projection.used_timestamp_fallback = false;
	  projection.used_native_projection = false;
	  projection.point_cell_indices.resize(corrected_points.size());
	  std::fill(projection.point_cell_indices.begin(), projection.point_cell_indices.end(), -1);

	  const std::size_t rows_u = static_cast<std::size_t>(projection.rows);
	  const std::size_t cols_u = static_cast<std::size_t>(projection.cols);
	  if (rows_u == 0U ||
	      cols_u == 0U ||
	      rows_u > std::numeric_limits<std::size_t>::max() / cols_u) {
	    projection.cells.clear();
	    return;
	  }
	  const std::size_t cell_count = rows_u * cols_u;
	  if (cell_count > kMaxProjectionCells ||
	      cell_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
	    projection.cells.clear();
	    return;
	  }
	  if (projection.cells.size() != cell_count) {
	    projection.cells.assign(cell_count, ProjectionCell{});
	    projection.touched_cell_indices.clear();
	  }

	  if (!cloud_map_before_correction) {
	    return;
  }

  projection.used_ring_fallback =
      config_.projection_use_ring_field && pre_scan.valid_point_count > 0 &&
      (pre_scan.valid_ring_count == 0 ||
       pre_scan.unique_ring_count < std::min(4, projection.rows));
  projection.used_timestamp_fallback =
      config_.projection_use_point_timestamp && pre_scan.timestamp_count == 0;
  projection.used_native_projection =
      pre_scan.valid_point_count > 0 && !projection.used_ring_fallback;

  for (const std::size_t i : valid_indices) {
    if (i >= cloud_map_before_correction->size() ||
        i >= points_lidar.size() || i >= corrected_points.size()) {
      continue;
    }
    const PointType& source = (*cloud_map_before_correction)[i];
    bool ring_fallback = false;
    int row = -1;
    int col = -1;
    float range = 0.0f;
    float yaw = 0.0f;
    float elevation = 0.0f;
    const int source_ring = projection.used_ring_fallback ? -1 : static_cast<int>(source.ring);
    if (!projectPoint(points_lidar[i], source_ring, !projection.used_ring_fallback,
	                      row, col, range, yaw, elevation, ring_fallback)) {
	      continue;
	    }

	    std::size_t cell_index = 0U;
	    if (!projectionIndex(row, col, projection.cols, projection.cells.size(), cell_index)) {
	      continue;
	    }
	    ProjectionCell& cell = projection.cells[cell_index];
	    if (cell.valid && cell.range <= range) {
	      continue;
	    }
    if (cell.valid && cell.point_index < projection.point_cell_indices.size()) {
      projection.point_cell_indices[cell.point_index] = -1;
      if (cell.point_index < point_meta.size()) {
        point_meta[cell.point_index].projection_cell = -1;
      }
	    }
	    if (!cell.valid) {
	      projection.touched_cell_indices.push_back(static_cast<int>(cell_index));
	    }
	    cell.valid = true;
	    cell.point_index = i;
    cell.range = range;
    cell.row = row;
    cell.col = col;
    cell.ring = static_cast<int>(source.ring);
    cell.yaw = yaw;
    cell.elevation = elevation;
	    cell.timestamp =
	        projection.used_timestamp_fallback ? 0.0 : static_cast<double>(source.timestamp);
	    cell.point_lidar = points_lidar[i];
	    cell.point_map = corrected_points[i];
	    projection.point_cell_indices[i] = static_cast<int>(cell_index);
	    if (i < point_meta.size()) {
	      point_meta[i].projection_cell = static_cast<int>(cell_index);
	      point_meta[i].projection_range = range;
	      point_meta[i].projection_yaw = yaw;
      point_meta[i].projection_elevation = elevation;
    }
  }
}

void MDetectorFilter::appendDepthFrame(
    const ProjectionFrame& projection,
    const PointMask& removed_points,
    double stamp,
    const Eigen::Ref<const Eigen::Matrix4f>& T_map_lidar) {
  ensureHistorySlots(projection.cells.size());
  const std::size_t slot_index = nextHistorySlot();
  DepthFrame& frame = history_slots_[slot_index];
  resetDepthFrameCells(frame);
  frame.scan_index = scan_index_;
  frame.stamp = stamp;
  frame.T_map_lidar = T_map_lidar;
  frame.T_lidar_map = T_map_lidar.inverse();
  frame.valid_cell_indices.clear();

  for (const int touched_idx : projection.touched_cell_indices) {
    if (touched_idx < 0 ||
        static_cast<std::size_t>(touched_idx) >= projection.cells.size() ||
        static_cast<std::size_t>(touched_idx) >= frame.cells.size()) {
      continue;
    }
    const std::size_t i = static_cast<std::size_t>(touched_idx);
    const ProjectionCell& source = projection.cells[i];
    if (!source.valid) {
      continue;
    }
    DepthCell& cell = frame.cells[i];
    cell.valid = true;
    cell.min_range_all = source.range;
    cell.max_range_all = source.range;
    cell.all_points = 1;
    const bool removed =
        source.point_index < removed_points.size() && removed_points[source.point_index];
    if (!removed) {
      cell.min_range_static = source.range;
      cell.max_range_static = source.range;
      cell.static_points = 1;
    }
    frame.valid_cell_indices.push_back(touched_idx);
  }

  if (config_.history_duration > 0.0) {
    while (!history_order_.empty() &&
           stamp - history_slots_[history_order_.front()].stamp >
               config_.history_duration + 0.5 * config_.frame_duration) {
      history_order_.pop_front();
    }
  }
}

void MDetectorFilter::ensureHistorySlots(std::size_t cell_count) {
  const std::size_t max_frames =
      static_cast<std::size_t>(std::max(config_.max_history_frames, 1));
  if (history_slots_.size() != max_frames) {
    history_slots_.assign(max_frames, DepthFrame{});
    history_order_.clear();
  }
  bool resized_cells = false;
  for (DepthFrame& frame : history_slots_) {
    if (frame.cells.size() != cell_count) {
      frame.cells.assign(cell_count, DepthCell{});
      frame.valid_cell_indices.clear();
      resized_cells = true;
    }
  }
  if (resized_cells) {
    history_order_.clear();
  }
}

void MDetectorFilter::resetDepthFrameCells(DepthFrame& frame) {
  for (const int idx : frame.valid_cell_indices) {
    if (idx >= 0 && static_cast<std::size_t>(idx) < frame.cells.size()) {
      frame.cells[static_cast<std::size_t>(idx)] = DepthCell{};
    }
  }
  frame.valid_cell_indices.clear();
}

std::size_t MDetectorFilter::nextHistorySlot() {
  const std::size_t max_frames =
      static_cast<std::size_t>(std::max(config_.max_history_frames, 1));
  if (history_order_.size() >= max_frames) {
    const std::size_t slot = history_order_.front();
    history_order_.pop_front();
    history_order_.push_back(slot);
    return slot;
  }

  std::array<bool, 64> used{};
  std::vector<std::uint8_t> dynamic_used;
  const bool use_static_seen = max_frames <= used.size();
  if (!use_static_seen) {
    dynamic_used.assign(max_frames, 0U);
  }
  for (const std::size_t slot : history_order_) {
    if (slot >= max_frames) {
      continue;
    }
    if (use_static_seen) {
      used[slot] = true;
    } else {
      dynamic_used[slot] = 1U;
    }
  }
  for (std::size_t slot = 0; slot < max_frames; ++slot) {
    const bool occupied = use_static_seen ? used[slot] : dynamic_used[slot] != 0U;
    if (!occupied) {
      history_order_.push_back(slot);
      return slot;
    }
  }

  history_order_.push_back(0U);
  return 0U;
}

MDetectorFilter::PointEvidence MDetectorFilter::evaluatePointEvidence(
    const PointMeta& point,
    bool in_track_mask) const {
  PointEvidence evidence;
  const float margin = static_cast<float>(config_.case_depth_margin);
  const float static_margin = static_cast<float>(config_.map_consistency_depth);
  for (const std::size_t frame_slot : history_order_) {
    if (frame_slot >= history_slots_.size()) {
      continue;
    }
    const DepthFrame& frame = history_slots_[frame_slot];
    const Eigen::Vector4f p_hist_h = frame.T_lidar_map * point.map_h;
    const Eigen::Vector3f p_hist = p_hist_h.head<3>();
    int row = -1;
    int col = -1;
    float range = 0.0f;
    float yaw = 0.0f;
    float elevation = 0.0f;
    bool ring_fallback = false;
    if (!projectPoint(p_hist, point.source_ring, false,
                      row, col, range, yaw, elevation, ring_fallback)) {
      continue;
    }

    bool compared = false;
    bool case1 = false;
    bool case2 = false;
    bool case3 = false;
    bool static_consistent = false;
    for (int dc = -1; dc <= 1; ++dc) {
      int c = col + dc;
      if (c < 0) {
        c += config_.projection_cols;
	      } else if (c >= config_.projection_cols) {
	        c -= config_.projection_cols;
	      }
	      std::size_t cell_index = 0U;
	      if (!projectionIndex(row,
	                           c,
	                           config_.projection_cols,
	                           frame.cells.size(),
	                           cell_index)) {
	        continue;
	      }
	      const DepthCell& cell = frame.cells[cell_index];
      if (!cell.valid) {
        continue;
      }
      compared = true;
      if (cell.static_points > 0) {
        if (range + margin < cell.min_range_static) {
          case1 = true;
        }
        if (std::abs(range - cell.min_range_static) <= static_margin ||
            std::abs(range - cell.max_range_static) <= static_margin) {
          static_consistent = true;
        }
      }
      if (range + margin < cell.min_range_all) {
        case2 = true;
      }
      // Case 3 is continuity evidence for an already tracked object. It is not
      // used as standalone current-scan removal because the object may simply be
      // absent behind the current viewpoint or occluded by static structure.
      if (in_track_mask && range > cell.max_range_all + margin) {
        case3 = true;
      }
    }
    if (!compared) {
      continue;
    }
    ++evidence.compared_frames;
    if (case1) {
      ++evidence.case1_votes;
    }
    if (case2) {
      ++evidence.case2_votes;
    }
    if (case3) {
      ++evidence.case3_votes;
    }
    if (static_consistent) {
      ++evidence.static_votes;
    }
  }
  return evidence;
}

MDetectorFilter::RegistrationResult MDetectorFilter::filterRegistration(
    const pcl::PointCloud<PointType>::ConstPtr& cloud_map,
    const Eigen::Vector3f& sensor_origin_map,
    int min_points) const {
  RegistrationResult result;
  result.dynamic_points = std::make_shared<pcl::PointCloud<PointType>>();
  result.cloud = cloud_map ? cloud_map : std::make_shared<const pcl::PointCloud<PointType>>();
  result.stats.input_points = cloud_map ? cloud_map->size() : 0U;
  result.stats.registration_kept = result.stats.input_points;
  result.stats.voxel_count = voxels_.size();
  result.stats.scan_index = scan_index_;
  result.stats.warmup = !warmedUp();
  result.stats.track_count = tracks_.size();
  result.stats.confirmed_track_count = confirmedTrackCount();
  result.stats.tentative_track_count =
      result.stats.track_count >= result.stats.confirmed_track_count
          ? result.stats.track_count - result.stats.confirmed_track_count
          : 0U;

  if (!cloud_map || cloud_map->empty()) {
    result.stats.registration_kept = 0U;
    return result;
  }

  if (!enabled() || !warmedUp() || active_track_mask_.empty()) {
    result.kept_indices.reserve(cloud_map->size());
    for (std::size_t i = 0; i < cloud_map->size(); ++i) {
      result.kept_indices.push_back(static_cast<int>(i));
    }
    return result;
  }

  auto filtered = std::make_shared<pcl::PointCloud<PointType>>();
  filtered->points.reserve(cloud_map->size());
  result.kept_indices.reserve(cloud_map->size());

  for (std::size_t i = 0; i < cloud_map->size(); ++i) {
    const PointType& point = (*cloud_map)[i];
    const Eigen::Vector3f point_map(point.x, point.y, point.z);
    const bool valid = pointFinite(point) && pointInRange(point_map, sensor_origin_map);
    const bool tracked_dynamic = valid && pointInActiveTrackMask(point_map);
    if (tracked_dynamic) {
      ++result.stats.track_removed_points;
      continue;
    }
    filtered->push_back(point);
    result.kept_indices.push_back(static_cast<int>(i));
  }

  finalizeCloud(filtered);
  finalizeCloud(result.dynamic_points);
  result.stats.registration_kept = filtered->size();
  result.stats.registration_removed =
      result.stats.input_points >= result.stats.registration_kept
          ? result.stats.input_points - result.stats.registration_kept
          : 0U;

  if (static_cast<int>(filtered->size()) < min_points) {
    result.cloud = cloud_map;
    result.stats.registration_kept = cloud_map->size();
    result.stats.registration_fallback = true;
    result.kept_indices.clear();
    result.kept_indices.reserve(cloud_map->size());
    for (std::size_t i = 0; i < cloud_map->size(); ++i) {
      result.kept_indices.push_back(static_cast<int>(i));
    }
  } else {
    result.cloud = filtered;
  }

  return result;
}

std::vector<MDetectorFilter::Cluster> MDetectorFilter::buildClusters(
    const pcl::PointCloud<PointType>::ConstPtr& cloud_map_before_correction,
    const std::vector<Eigen::Vector3f>& corrected_points,
    const std::vector<VoxelKey>& point_keys,
    const PointMask& valid_points,
    const PointMask& static_supported_points,
    const PointMask& seed_points,
    const std::vector<std::size_t>& seed_indices,
    const PointMask& case1_points,
    const PointMask& case2_points,
    const PointMask& case3_points,
    const ScanVoxelMap& scan_voxels,
    const ProjectionFrame& projection) const {
  std::vector<Cluster> clusters;
  if (!cloud_map_before_correction || !warmedUp()) {
    return clusters;
  }

  std::vector<bool> visited(corrected_points.size(), false);
  const std::array<VoxelKey, 6> voxel_neighbors{{
      VoxelKey{1, 0, 0}, VoxelKey{-1, 0, 0},
      VoxelKey{0, 1, 0}, VoxelKey{0, -1, 0},
      VoxelKey{0, 0, 1}, VoxelKey{0, 0, -1}}};

  auto push_point = [&](std::queue<std::size_t>& q, std::size_t idx) {
    if (idx >= visited.size() || visited[idx] ||
        idx >= valid_points.size() || !valid_points[idx]) {
      return;
    }
    visited[idx] = true;
    q.push(idx);
  };

  auto allow_neighbor = [&](std::size_t idx, std::size_t neighbor_idx) {
    if (idx >= corrected_points.size() || neighbor_idx >= corrected_points.size() ||
        neighbor_idx >= seed_points.size()) {
      return false;
    }
    const bool neighbor_seed = seed_points[neighbor_idx] ||
        (neighbor_idx < case1_points.size() && case1_points[neighbor_idx]) ||
        (neighbor_idx < case2_points.size() && case2_points[neighbor_idx]);
    if (neighbor_idx < static_supported_points.size() &&
        static_supported_points[neighbor_idx] && !neighbor_seed) {
      return false;
    }
    const float max_link_distance =
        std::max(0.35f, 2.5f * static_cast<float>(config_.voxel_size));
    const float distance = (corrected_points[neighbor_idx] - corrected_points[idx]).norm();
    return distance <= max_link_distance || neighbor_seed;
  };

  // Same-ring range adjacency is intentionally conservative to avoid bridging
  // foreground objects into walls across depth jumps. This is not a public
  // parameter yet; expose it if M-detector segmentation needs dataset tuning.
  const float projection_range_gate =
      std::max(0.30f, 2.5f * static_cast<float>(config_.case_depth_margin));

  for (const std::size_t seed_idx : seed_indices) {
    if (seed_idx >= seed_points.size()) {
      continue;
    }
    if (!seed_points[seed_idx] || !valid_points[seed_idx] || visited[seed_idx]) {
      continue;
    }

    Cluster cluster;
    cluster.min = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    cluster.max = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
    cluster.min_key = VoxelKey{std::numeric_limits<int>::max(),
                               std::numeric_limits<int>::max(),
                               std::numeric_limits<int>::max()};
    cluster.max_key = VoxelKey{std::numeric_limits<int>::lowest(),
                               std::numeric_limits<int>::lowest(),
                               std::numeric_limits<int>::lowest()};
    std::unordered_set<VoxelKey, VoxelKeyHash> cluster_voxels;
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    std::queue<std::size_t> q;
    push_point(q, seed_idx);

    while (!q.empty()) {
      const std::size_t idx = q.front();
      q.pop();
	      if (idx >= corrected_points.size()) {
	        continue;
	      }
	      const Eigen::Vector3f& p = corrected_points[idx];
	      VoxelKey key;
	      if (idx < point_keys.size()) {
	        key = point_keys[idx];
	      } else if (!tryKeyFromPoint(p, key)) {
	        continue;
	      }
	      if (cluster_voxels.insert(key).second) {
	        cluster.voxels.push_back(key);
      }
      cluster.point_indices.push_back(idx);
      cluster.min = cluster.min.cwiseMin(p);
      cluster.max = cluster.max.cwiseMax(p);
      cluster.min_key.x = std::min(cluster.min_key.x, key.x);
      cluster.min_key.y = std::min(cluster.min_key.y, key.y);
      cluster.min_key.z = std::min(cluster.min_key.z, key.z);
      cluster.max_key.x = std::max(cluster.max_key.x, key.x);
      cluster.max_key.y = std::max(cluster.max_key.y, key.y);
      cluster.max_key.z = std::max(cluster.max_key.z, key.z);
      sum += p;
      if (idx < seed_points.size() && seed_points[idx]) {
        ++cluster.seed_count;
      }
      if (idx < case1_points.size() && case1_points[idx]) {
        ++cluster.case1_count;
      }
      if (idx < case2_points.size() && case2_points[idx]) {
        ++cluster.case2_count;
      }
      if (idx < case3_points.size() && case3_points[idx]) {
        ++cluster.case3_count;
      }
      if (idx < static_supported_points.size() && static_supported_points[idx]) {
        ++cluster.static_count;
      }

      const auto same_voxel_it = scan_voxels.find(key);
      if (same_voxel_it != scan_voxels.end()) {
        for (const std::size_t neighbor_idx : same_voxel_it->second) {
          if (allow_neighbor(idx, neighbor_idx)) {
            push_point(q, neighbor_idx);
          }
        }
      }
      for (const VoxelKey& offset : voxel_neighbors) {
        const VoxelKey next{key.x + offset.x, key.y + offset.y, key.z + offset.z};
        const auto voxel_it = scan_voxels.find(next);
        if (voxel_it == scan_voxels.end()) {
          continue;
        }
        for (const std::size_t neighbor_idx : voxel_it->second) {
          if (allow_neighbor(idx, neighbor_idx)) {
            push_point(q, neighbor_idx);
          }
        }
      }

      const int cell_index =
          idx < projection.point_cell_indices.size() ? projection.point_cell_indices[idx] : -1;
      if (cell_index < 0 ||
          static_cast<std::size_t>(cell_index) >= projection.cells.size()) {
        continue;
      }
      const ProjectionCell& cell = projection.cells[static_cast<std::size_t>(cell_index)];
      if (!cell.valid) {
        continue;
      }
      for (int dc = -1; dc <= 1; ++dc) {
        int col = cell.col + dc;
        if (col < 0) {
          col += projection.cols;
        } else if (col >= projection.cols) {
          col -= projection.cols;
        }
	        std::size_t neighbor_cell_index = 0U;
	        if (!projectionIndex(cell.row,
	                             col,
	                             projection.cols,
	                             projection.cells.size(),
	                             neighbor_cell_index)) {
	          continue;
	        }
	        const ProjectionCell& neighbor = projection.cells[neighbor_cell_index];
        if (!neighbor.valid) {
          continue;
        }
        const bool compatible_range =
            std::abs(neighbor.range - cell.range) <= projection_range_gate;
        const bool dynamic_neighbor =
            neighbor.point_index < seed_points.size() &&
            (seed_points[neighbor.point_index] || seed_points[idx]);
        if ((compatible_range || dynamic_neighbor) &&
            allow_neighbor(idx, neighbor.point_index)) {
          push_point(q, neighbor.point_index);
        }
      }
    }

    if (cluster.point_indices.empty()) {
      continue;
    }
    cluster.centroid = sum / static_cast<float>(cluster.point_indices.size());
    cluster.ground_like = clusterLooksGroundLike(cluster);
    cluster.edge_like = clusterLooksEdgeLike(cluster);
    cluster.wall_like = clusterLooksWallLike(cluster);
    const Eigen::Vector3f extent = cluster.max - cluster.min;
    const float max_extent = extent.maxCoeff();
    const float static_ratio =
        static_cast<float>(cluster.static_count) /
        static_cast<float>(std::max<std::size_t>(cluster.point_indices.size(), 1U));
    const float seed_ratio =
        static_cast<float>(cluster.seed_count) /
        static_cast<float>(std::max<std::size_t>(cluster.point_indices.size(), 1U));
    const bool static_veto =
        static_ratio > static_cast<float>(config_.static_veto_ratio) &&
        seed_ratio < 0.60f;
    // Accepted clusters need a substantial moving-event seed fraction. These
    // seed-ratio thresholds are not public parameters yet; expose them before
    // doing fine M-detector aggressiveness tuning across datasets.
    const bool enough_dynamic_support = seed_ratio >= 0.45f;

    cluster.accepted =
        static_cast<int>(cluster.point_indices.size()) >= config_.min_cluster_points &&
        max_extent <= static_cast<float>(config_.max_cluster_extent) &&
        enough_dynamic_support &&
        !cluster.ground_like &&
        !cluster.edge_like &&
        !cluster.wall_like &&
        !static_veto;
    // Strong clusters can confirm tracks immediately. The seed ratio is
    // deliberately hidden for now and should become a parameter if this backend
    // is tuned beyond the current dynamic dataset.
    cluster.strong = cluster.accepted && seed_ratio >= 0.70f;
    clusters.push_back(std::move(cluster));
  }

  return clusters;
}

bool MDetectorFilter::clusterLooksGroundLike(const Cluster& cluster) const {
  if (cluster.point_indices.empty()) {
    return false;
  }
  const Eigen::Vector3f extent = cluster.max - cluster.min;
  const float horizontal = std::max(extent.x(), extent.y());
  const float vertical = extent.z();
  return vertical <= std::max(0.25f, 2.0f * static_cast<float>(config_.voxel_size)) &&
         horizontal >= 0.8f &&
         cluster.seed_count * 2 < static_cast<int>(cluster.point_indices.size());
}

bool MDetectorFilter::clusterLooksEdgeLike(const Cluster& cluster) const {
  if (cluster.point_indices.empty()) {
    return false;
  }
  const Eigen::Vector3f extent = cluster.max - cluster.min;
  const float max_extent = extent.maxCoeff();
  const float min_extent = extent.minCoeff();
  const float static_ratio =
      static_cast<float>(cluster.static_count) /
      static_cast<float>(std::max<std::size_t>(cluster.point_indices.size(), 1U));
  const float seed_ratio =
      static_cast<float>(cluster.seed_count) /
      static_cast<float>(std::max<std::size_t>(cluster.point_indices.size(), 1U));
  return max_extent >= 0.8f &&
         min_extent <= std::max(0.15f, 1.25f * static_cast<float>(config_.voxel_size)) &&
         static_ratio >= 0.35f &&
         seed_ratio <= 0.35f;
}

bool MDetectorFilter::clusterLooksWallLike(const Cluster& cluster) const {
  if (cluster.point_indices.empty()) {
    return false;
  }
  const Eigen::Vector3f extent = cluster.max - cluster.min;
  const float max_extent = extent.maxCoeff();
  const float min_extent = extent.minCoeff();
  const float vertical = extent.z();
  const float static_ratio =
      static_cast<float>(cluster.static_count) /
      static_cast<float>(std::max<std::size_t>(cluster.point_indices.size(), 1U));
  const float seed_ratio =
      static_cast<float>(cluster.seed_count) /
      static_cast<float>(std::max<std::size_t>(cluster.point_indices.size(), 1U));
  return max_extent >= 1.5f &&
         min_extent <= std::max(0.18f, 1.5f * static_cast<float>(config_.voxel_size)) &&
         vertical >= 0.8f &&
         static_ratio >= 0.25f &&
         seed_ratio < 0.50f;
}

float MDetectorFilter::bboxOverlapRatio(const Cluster& cluster, const Track& track) const {
  const Eigen::Vector3f inter_min = cluster.min.cwiseMax(track.min);
  const Eigen::Vector3f inter_max = cluster.max.cwiseMin(track.max);
  const Eigen::Vector3f inter = (inter_max - inter_min).cwiseMax(Eigen::Vector3f::Zero());
  const float inter_vol = inter.x() * inter.y() * inter.z();
  const Eigen::Vector3f c_extent =
      (cluster.max - cluster.min).cwiseMax(Eigen::Vector3f::Constant(config_.voxel_size));
  const Eigen::Vector3f t_extent =
      (track.max - track.min).cwiseMax(Eigen::Vector3f::Constant(config_.voxel_size));
  const float c_vol = c_extent.x() * c_extent.y() * c_extent.z();
  const float t_vol = t_extent.x() * t_extent.y() * t_extent.z();
  return inter_vol / std::max(std::min(c_vol, t_vol), 1.0e-3f);
}

std::vector<MDetectorFilter::VoxelKey> MDetectorFilter::paddedClusterMaskKeys(
    const Cluster& cluster) const {
  std::vector<VoxelKey> keys;
  if (!cluster.accepted) {
    return keys;
  }
  // Track masks are padded only in XY to catch sparse returns around the object
  // footprint without carving vertical static structure. These padding values
  // are not public parameters yet.
  std::unordered_set<VoxelKey, VoxelKeyHash> seen;
  for (const VoxelKey& voxel : cluster.voxels) {
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = 0; dz <= 0; ++dz) {
          const VoxelKey key{voxel.x + dx, voxel.y + dy, voxel.z + dz};
          if (seen.insert(key).second) {
            keys.push_back(key);
          }
        }
      }
    }
  }
  return keys;
}

std::size_t MDetectorFilter::updateTracks(const std::vector<Cluster>& clusters) {
  for (Track& track : tracks_) {
    ++track.missed;
    if (track.confirmed && track.ttl_remaining > 0) {
      --track.ttl_remaining;
    }
  }

  std::vector<bool> matched(clusters.size(), false);
  for (Track& track : tracks_) {
    float best_score = std::numeric_limits<float>::max();
    int best_idx = -1;
    for (std::size_t i = 0; i < clusters.size(); ++i) {
      const Cluster& cluster = clusters[i];
      if (matched[i] || !cluster.accepted) {
        continue;
      }
      const Eigen::Vector3f predicted = track.centroid + track.velocity;
      const float distance = (cluster.centroid - predicted).norm();
      const float overlap = bboxOverlapRatio(cluster, track);
      const bool inside_gate =
          distance <= static_cast<float>(config_.max_assoc_distance) ||
          overlap >= 0.05f;
      const float score = distance - 0.5f * overlap;
      if (inside_gate && score < best_score) {
        best_score = score;
        best_idx = static_cast<int>(i);
      }
    }
    if (best_idx < 0) {
      continue;
    }

    const Cluster& cluster = clusters[static_cast<std::size_t>(best_idx)];
    matched[static_cast<std::size_t>(best_idx)] = true;
    track.velocity = cluster.centroid - track.centroid;
    track.centroid = cluster.centroid;
    track.min = cluster.min;
    track.max = cluster.max;
    track.min_key = cluster.min_key;
    track.max_key = cluster.max_key;
    track.mask_keys = paddedClusterMaskKeys(cluster);
    ++track.age;
    ++track.hits;
    track.missed = 0;
    track.last_seen_scan = scan_index_;
    track.stopped = track.confirmed && track.velocity.norm() <= 0.15f;
    track.confidence = std::min(track.confidence + (cluster.strong ? 1.0f : 0.5f), 100.0f);
    if (track.hits >= config_.track_confirm_hits || cluster.strong) {
      track.confirmed = true;
    }
    if (track.confirmed) {
      track.ttl_remaining = config_.track_ttl_scans;
    }
  }

  std::size_t rejected = 0U;
  for (std::size_t i = 0; i < clusters.size(); ++i) {
    const Cluster& cluster = clusters[i];
    if (matched[i] || !cluster.accepted) {
      continue;
    }
    if (static_cast<int>(cluster.point_indices.size()) < config_.min_track_cluster_points) {
      ++rejected;
      continue;
    }
    Track track;
    track.id = next_track_id_++;
    track.centroid = cluster.centroid;
    track.min = cluster.min;
    track.max = cluster.max;
    track.min_key = cluster.min_key;
    track.max_key = cluster.max_key;
    track.mask_keys = paddedClusterMaskKeys(cluster);
    track.age = 1;
    track.hits = 1;
    track.missed = 0;
    track.last_seen_scan = scan_index_;
    track.confidence = cluster.strong ? 1.0f : 0.5f;
    track.confirmed = cluster.strong || config_.track_confirm_hits <= 1;
    track.ttl_remaining = track.confirmed ? config_.track_ttl_scans : 0;
    tracks_.push_back(std::move(track));
  }

  pruneTracks();
  rebuildActiveTrackMask();
  return rejected;
}

void MDetectorFilter::pruneTracks() {
  tracks_.erase(
      std::remove_if(
          tracks_.begin(), tracks_.end(),
          [this](const Track& track) {
            if (!track.confirmed) {
              return track.missed > config_.track_confirm_hits;
            }
            return track.missed > config_.track_ttl_scans ||
                   track.ttl_remaining <= 0;
          }),
      tracks_.end());
}

void MDetectorFilter::rebuildActiveTrackMask() {
  active_track_mask_.clear();
  for (const Track& track : tracks_) {
    if (!track.confirmed || track.ttl_remaining <= 0) {
      continue;
    }
    active_track_mask_.insert(track.mask_keys.begin(), track.mask_keys.end());
  }
}

bool MDetectorFilter::pointInActiveTrackMask(const Eigen::Vector3f& point_map) const {
  VoxelKey key;
  if (!tryKeyFromPoint(point_map, key)) {
    return false;
  }
  return active_track_mask_.find(key) != active_track_mask_.end();
}

std::vector<MDetectorFilter::DebugBox> MDetectorFilter::activeTrackBoxes() const {
  std::vector<DebugBox> boxes;
  for (const Track& track : tracks_) {
    if (!track.confirmed || track.ttl_remaining <= 0) {
      continue;
    }
    DebugBox box;
    box.min = track.min;
    box.max = track.max;
    box.centroid = track.centroid;
    box.id = track.id;
    box.age = track.age;
    box.missed = track.missed;
    box.confirmed = track.confirmed;
    box.stopped = track.stopped;
    boxes.push_back(box);
  }
  return boxes;
}

std::size_t MDetectorFilter::confirmedTrackCount() const {
  return static_cast<std::size_t>(
      std::count_if(tracks_.begin(), tracks_.end(), [](const Track& track) {
        return track.confirmed && track.ttl_remaining > 0;
      }));
}

MDetectorFilter::MappingResult MDetectorFilter::update(
    const pcl::PointCloud<PointType>::ConstPtr& cloud_map_before_correction,
    const Eigen::Ref<const Eigen::Matrix4f>& T_correction,
    const Eigen::Vector3f& sensor_origin_map,
    const Eigen::Ref<const Eigen::Matrix4f>& T_map_lidar,
    double scan_stamp_sec,
    bool emit_removed_cloud) {
  MappingResult result;
  result.dynamic_points_map = std::make_shared<pcl::PointCloud<PointType>>();
  result.keyframe_cloud = cloud_map_before_correction
      ? cloud_map_before_correction
      : std::make_shared<const pcl::PointCloud<PointType>>();
  result.stats.input_points =
      cloud_map_before_correction ? cloud_map_before_correction->size() : 0U;
  result.stats.registration_kept = result.stats.input_points;
  result.stats.mapping_kept = result.stats.input_points;
  result.stats.voxel_count = voxels_.size();
  result.stats.scan_index = scan_index_;
  result.stats.warmup = !warmedUp();
  result.stats.track_count = tracks_.size();
  result.stats.confirmed_track_count = confirmedTrackCount();

  if (!cloud_map_before_correction || cloud_map_before_correction->empty()) {
    result.stats.mapping_kept = 0U;
    return result;
  }
  if (!enabled()) {
    return result;
  }

  ++scan_index_;
  const std::size_t point_count = cloud_map_before_correction->size();
  auto& corrected_points = scratch_.corrected_points;
  auto& points_lidar = scratch_.points_lidar;
  auto& publishable_points = scratch_.publishable_points;
  auto& valid_points = scratch_.valid_points;
  auto& publishable_indices = scratch_.publishable_indices;
  auto& valid_indices = scratch_.valid_indices;
  auto& seed_indices = scratch_.seed_indices;
  auto& point_keys = scratch_.point_keys;
  auto& point_meta = scratch_.point_meta;
  auto& scan_voxels = scratch_.scan_voxels;
  auto& projection_pre_scan = scratch_.projection_pre_scan;
  corrected_points.clear();
  points_lidar.clear();
  publishable_indices.clear();
  valid_indices.clear();
  seed_indices.clear();
  point_keys.clear();
  projection_pre_scan.reset();
  corrected_points.reserve(point_count);
  points_lidar.reserve(point_count);
  publishable_indices.reserve(point_count);
  valid_indices.reserve(point_count);
  seed_indices.reserve(point_count);
  point_keys.reserve(point_count);
  publishable_points.assign(point_count, false);
  valid_points.assign(point_count, false);
  point_meta.assign(point_count, PointMeta{});
  scan_voxels.clear();
  scan_voxels.reserve(point_count);

  const Eigen::Matrix4f T_lidar_map = T_map_lidar.inverse();
  for (std::size_t i = 0; i < point_count; ++i) {
    const PointType& point = (*cloud_map_before_correction)[i];
    if (!pointFinite(point)) {
      corrected_points.emplace_back(Eigen::Vector3f::Zero());
      points_lidar.emplace_back(Eigen::Vector3f::Zero());
      point_keys.push_back(VoxelKey{});
      continue;
    }
    const Eigen::Vector4f raw(point.x, point.y, point.z, 1.0f);
    const Eigen::Vector4f corrected_h = T_correction * raw;
    const Eigen::Vector3f corrected = corrected_h.head<3>();
    const bool valid = pointInRange(corrected, sensor_origin_map);
    corrected_points.emplace_back(corrected);
    publishable_points[i] = true;
    publishable_indices.push_back(i);
	    if (!valid) {
	      points_lidar.emplace_back(Eigen::Vector3f::Zero());
	      point_keys.push_back(VoxelKey{});
	      continue;
	    }
	
	    VoxelKey key;
	    if (!tryKeyFromPoint(corrected, key)) {
	      points_lidar.emplace_back(Eigen::Vector3f::Zero());
	      point_keys.push_back(VoxelKey{});
	      continue;
	    }

	    const Eigen::Vector4f map_h(corrected.x(), corrected.y(), corrected.z(), 1.0f);
	    const Eigen::Vector4f lidar_h = T_lidar_map * map_h;
	    const Eigen::Vector3f point_lidar = lidar_h.head<3>();
	    points_lidar.emplace_back(point_lidar);
	    point_keys.push_back(key);
    valid_points[i] = true;
    PointMeta& meta = point_meta[i];
    meta.corrected = corrected;
    meta.lidar = point_lidar;
    meta.map_h = map_h;
    meta.key = key;
    meta.source_ring = static_cast<int>(point.ring);
    meta.valid = true;
    valid_indices.push_back(i);
    ++projection_pre_scan.valid_point_count;
    if (point.ring < static_cast<std::uint16_t>(config_.projection_rows)) {
      ++projection_pre_scan.valid_ring_count;
      if (point.ring < projection_pre_scan.ring_seen.size() &&
          !projection_pre_scan.ring_seen[point.ring]) {
        projection_pre_scan.ring_seen[point.ring] = true;
        ++projection_pre_scan.unique_ring_count;
      }
    }
    if (std::isfinite(point.timestamp) && std::abs(point.timestamp) > 1.0e-9) {
      ++projection_pre_scan.timestamp_count;
    }
    scan_voxels[key].push_back(i);
  }

  updateRingElevationModel(
      cloud_map_before_correction, points_lidar, valid_indices, projection_pre_scan);
  auto& projection = scratch_.projection;
  buildProjection(
      cloud_map_before_correction, corrected_points, points_lidar, valid_indices,
      projection_pre_scan, point_meta, projection);
  result.stats.projection_ring_fallback = projection.used_ring_fallback;
  result.stats.projection_timestamp_fallback = projection.used_timestamp_fallback;
  result.stats.projection_native = projection.used_native_projection;

  auto& case1_points = scratch_.case1_points;
  auto& case2_points = scratch_.case2_points;
  auto& case3_points = scratch_.case3_points;
  auto& seed_points = scratch_.seed_points;
  auto& static_supported_points = scratch_.static_supported_points;
  auto& active_track_points = scratch_.active_track_points;
  auto& cluster_points = scratch_.cluster_points;
  auto& track_points = scratch_.track_points;
  auto& stopped_points = scratch_.stopped_points;
  auto& removed_points = scratch_.removed_points;
  case1_points.assign(point_count, false);
  case2_points.assign(point_count, false);
  case3_points.assign(point_count, false);
  seed_points.assign(point_count, false);
  static_supported_points.assign(point_count, false);
  active_track_points.assign(point_count, false);
  cluster_points.assign(point_count, false);
  track_points.assign(point_count, false);
  stopped_points.assign(point_count, false);
  removed_points.assign(point_count, false);

  for (const std::size_t i : valid_indices) {
    if (i >= point_keys.size()) {
      continue;
    }
    const VoxelKey& key = point_keys[i];
    const auto voxel_it = voxels_.find(key);
    static_supported_points[i] =
        voxel_it != voxels_.end() && voxelIsStatic(voxel_it->second);
    active_track_points[i] = active_track_mask_.find(key) != active_track_mask_.end();
  }

  if (warmedUp()) {
    std::size_t static_veto_count = 0U;
    std::size_t case1_count = 0U;
    std::size_t case2_count = 0U;
    std::size_t case3_count = 0U;
    auto evaluate_evidence_index = [&](const std::size_t i,
                                       std::size_t& local_static_veto_count,
                                       std::size_t& local_case1_count,
                                       std::size_t& local_case2_count,
                                       std::size_t& local_case3_count) {
      if (i >= point_count) {
        return;
      }
      const bool in_track = i < active_track_points.size() && active_track_points[i];
      const PointEvidence evidence = evaluatePointEvidence(
          point_meta[i],
          in_track);
      const int moving_votes = evidence.case1_votes + evidence.case2_votes;
      const float static_ratio =
          evidence.compared_frames > 0
              ? static_cast<float>(evidence.static_votes) /
                    static_cast<float>(evidence.compared_frames)
              : 0.0f;
      const bool strong_static_support =
          evidence.static_votes >= config_.min_history_votes ||
          static_ratio > static_cast<float>(config_.static_veto_ratio);
      const bool static_veto =
          strong_static_support &&
          !in_track;
      if (static_veto) {
        ++local_static_veto_count;
      }
      if (evidence.case1_votes >= config_.min_history_votes) {
        case1_points[i] = true;
        ++local_case1_count;
      }
      if (evidence.case2_votes >= config_.min_history_votes) {
        case2_points[i] = true;
        ++local_case2_count;
      }
      if (in_track && evidence.case3_votes > 0) {
        case3_points[i] = true;
        ++local_case3_count;
      }
      if (!static_veto && moving_votes >= config_.min_history_votes) {
        if (!seed_points[i]) {
          seed_points[i] = true;
          seed_indices.push_back(i);
        }
      }
    };

    for (const std::size_t i : valid_indices) {
      evaluate_evidence_index(
          i,
          static_veto_count,
          case1_count,
          case2_count,
          case3_count);
    }
    result.stats.static_veto_count = static_veto_count;
    result.stats.case1_points = case1_count;
    result.stats.case2_points = case2_count;
    result.stats.case3_points = case3_count;

    const std::vector<Cluster> clusters = buildClusters(
        cloud_map_before_correction,
        corrected_points,
        point_keys,
        valid_points,
        static_supported_points,
        seed_points,
        seed_indices,
        case1_points,
        case2_points,
        case3_points,
        scan_voxels,
        projection);
    for (const Cluster& cluster : clusters) {
      if (cluster.edge_like || cluster.wall_like) {
        ++result.stats.edge_reject_count;
      }
      if (cluster.ground_like) {
        ++result.stats.ground_reject_count;
      }
      if (!cluster.accepted) {
        continue;
      }
      ++result.stats.cluster_count;
      for (const std::size_t idx : cluster.point_indices) {
        if (idx >= cluster_points.size() || cluster_points[idx]) {
          continue;
        }
        cluster_points[idx] = true;
        ++result.stats.cluster_points;
      }
    }
    result.stats.track_cluster_reject_count = updateTracks(clusters);

    for (const VoxelKey& mask_key : active_track_mask_) {
      const auto it = scan_voxels.find(mask_key);
      if (it == scan_voxels.end()) {
        continue;
      }
      for (const std::size_t idx : it->second) {
        if (idx < track_points.size()) {
          track_points[idx] = true;
        }
      }
    }
    for (const Track& track : tracks_) {
      if (!track.confirmed || !track.stopped || track.ttl_remaining <= 0) {
        continue;
      }
      for (const VoxelKey& mask_key : track.mask_keys) {
        const auto it = scan_voxels.find(mask_key);
        if (it == scan_voxels.end()) {
          continue;
        }
        for (const std::size_t idx : it->second) {
          if (idx < stopped_points.size()) {
            stopped_points[idx] = true;
          }
        }
      }
    }
  }

  auto keyframe_cloud = std::make_shared<pcl::PointCloud<PointType>>();
  keyframe_cloud->points.reserve(cloud_map_before_correction->size());
  std::size_t dynamic_removed_count = 0U;
  for (const std::size_t i : publishable_indices) {
    if (i >= point_count) {
      continue;
    }
	    if (i >= valid_points.size() || !valid_points[i]) {
	      keyframe_cloud->push_back((*cloud_map_before_correction)[i]);
	      continue;
	    }
    const bool static_supported =
        i < static_supported_points.size() && static_supported_points[i];
    const bool current_foreground =
        (i < case1_points.size() && case1_points[i]) ||
        (i < case2_points.size() && case2_points[i]);
    bool in_track = i < track_points.size() && track_points[i];
    bool in_stopped = i < stopped_points.size() && stopped_points[i];
    bool in_cluster = i < cluster_points.size() && cluster_points[i];
    const bool case3_only_track_support =
        i < case3_points.size() && case3_points[i] && !current_foreground;
    if (case3_only_track_support) {
      in_track = false;
      in_stopped = false;
      in_cluster = false;
    }
    if (static_supported && !current_foreground) {
      in_track = false;
      in_stopped = false;
      in_cluster = false;
    }

    const bool dynamic = in_cluster || in_track || in_stopped;
    if (dynamic) {
      removed_points[i] = true;
      if (emit_removed_cloud) {
        const PointType removed_point =
            makePointFromVector((*cloud_map_before_correction)[i], corrected_points[i]);
        result.dynamic_points_map->push_back(removed_point);
      }
      ++dynamic_removed_count;
      markDynamic(point_keys[i]);
      if (in_track) {
        ++result.stats.track_removed_points;
        ++result.stats.track_mask_removed_points;
      }
      if (in_stopped) {
        ++result.stats.stopped_suppressed_points;
      }
      continue;
    }

    markStatic(point_keys[i]);
    if (warmedUp()) {
      keyframe_cloud->push_back((*cloud_map_before_correction)[i]);
    }
  }

  pruneVoxels(sensor_origin_map);
  appendDepthFrame(projection, removed_points, scan_stamp_sec, T_map_lidar);

  finalizeCloud(keyframe_cloud);
  finalizeCloud(result.dynamic_points_map);

  result.keyframe_cloud = keyframe_cloud;
  result.stats.dynamic_removed = dynamic_removed_count;
  result.stats.mapping_kept = keyframe_cloud->size();
  result.stats.seed_points = seed_indices.size();
  result.stats.voxel_count = voxels_.size();
  result.stats.scan_index = scan_index_;
  result.stats.warmup = !warmedUp();
  result.stats.track_count = tracks_.size();
  result.stats.confirmed_track_count = confirmedTrackCount();
  result.stats.tentative_track_count =
      result.stats.track_count >= result.stats.confirmed_track_count
          ? result.stats.track_count - result.stats.confirmed_track_count
          : 0U;

  return result;
}

}  // namespace dlio
