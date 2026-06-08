/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 ***********************************************************/

#pragma once

#include "dlio/dlio.h"

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pcl/point_cloud.h>

namespace dlio {

class MDetectorFilter {
public:
  struct Config {
    bool enabled = false;

    // Shared map-memory controls inherited from dynamic_filter/*.
    double voxel_size = 0.20;
    double min_range = 1.0;
    // Dynamic detection/removal range gate. Finite out-of-range points pass
    // through to the clean map unchanged; raise this, and local_radius with it,
    // for full-range dynamic detection.
    double max_range = 10.0;
    // Wait for enough temporal context before trusting dynamic labels.
    int warmup_scans = 5;
    // Static support in the voxel memory used to veto wall/floor/edge removal.
    int static_score_threshold = 2;
    int static_window_scans = 5;
    int max_age_scans = 300;
    // Local pruning radius for voxel memory; keep this at least max_range.
    double local_radius = 30.0;

    // JT-128 projection uses ring as image row and actual yaw as column.
    int projection_rows = 128;
    int projection_cols = 900;
    bool projection_use_ring_field = true;
    bool projection_use_point_timestamp = true;

    // Temporal depth-map history. Longer history and more votes are more
    // static-preserving; shorter history reacts faster to moving objects.
    double history_duration = 0.5;
    int max_history_frames = 5;
    double frame_duration = 0.1;
    int min_history_votes = 3;
    // Depth-event margins. Larger values reduce false positives on stable
    // structure; smaller values are more aggressive.
    double case_depth_margin = 0.15;
    double map_consistency_depth = 0.25;
    // Cluster gates. Larger minimums and smaller max extent preserve static
    // walls/floors/edges at the cost of sparse object sensitivity.
    int min_cluster_points = 60;
    int min_track_cluster_points = 120;
    double max_cluster_extent = 3.0;
    // Tracking persistence. Larger association/TTL follows objects longer but
    // can over-mask static structure after a bad association.
    double max_assoc_distance = 0.9;
    int track_confirm_hits = 2;
    int track_ttl_scans = 20;
    // Lower is more static-preserving; higher allows dynamic labels near
    // previously stable structure.
    double static_veto_ratio = 0.25;
  };

  struct Stats {
    std::size_t input_points = 0;
    std::size_t registration_kept = 0;
    std::size_t registration_removed = 0;
    std::size_t mapping_kept = 0;
    std::size_t dynamic_removed = 0;
    std::size_t case1_points = 0;
    std::size_t case2_points = 0;
    std::size_t case3_points = 0;
    std::size_t seed_points = 0;
    std::size_t cluster_points = 0;
    std::size_t cluster_count = 0;
    std::size_t track_removed_points = 0;
    std::size_t stopped_suppressed_points = 0;
    std::size_t track_mask_removed_points = 0;
    std::size_t track_cluster_reject_count = 0;
    std::size_t edge_reject_count = 0;
    std::size_t ground_reject_count = 0;
    std::size_t static_veto_count = 0;
    std::size_t tentative_track_count = 0;
    std::size_t confirmed_track_count = 0;
    std::size_t track_count = 0;
    std::size_t voxel_count = 0;
    int scan_index = 0;
    bool warmup = true;
    bool registration_fallback = false;
    bool projection_ring_fallback = false;
    bool projection_timestamp_fallback = false;
    bool projection_native = false;
  };

  struct DebugBox {
    Eigen::Vector3f min = Eigen::Vector3f::Zero();
    Eigen::Vector3f max = Eigen::Vector3f::Zero();
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    int id = -1;
    int age = 0;
    int missed = 0;
    bool confirmed = false;
    bool stopped = false;
  };

  struct RegistrationResult {
    pcl::PointCloud<PointType>::ConstPtr cloud;
    pcl::PointCloud<PointType>::Ptr dynamic_points;
    std::vector<int> kept_indices;
    Stats stats;
  };

  struct MappingResult {
    pcl::PointCloud<PointType>::ConstPtr keyframe_cloud;
    pcl::PointCloud<PointType>::Ptr dynamic_points_map;
    Stats stats;
  };

  MDetectorFilter();
  explicit MDetectorFilter(const Config& config);

  void configure(const Config& config);
  const Config& config() const;

  void reset();
  bool enabled() const;
  bool warmedUp() const;
  std::size_t voxelCount() const;

  RegistrationResult filterRegistration(
      const pcl::PointCloud<PointType>::ConstPtr& cloud_map,
      const Eigen::Vector3f& sensor_origin_map,
      int min_points) const;

  MappingResult update(
      const pcl::PointCloud<PointType>::ConstPtr& cloud_map_before_correction,
      const Eigen::Ref<const Eigen::Matrix4f>& T_correction,
      const Eigen::Vector3f& sensor_origin_map,
      const Eigen::Ref<const Eigen::Matrix4f>& T_map_lidar,
      double scan_stamp_sec = 0.0,
      bool emit_removed_cloud = true);

  bool isStaticVoxel(const Eigen::Vector3f& point_map) const;
  bool isDynamicVoxel(const Eigen::Vector3f& point_map) const;

  // Test-only accessors. These are compiled out of production builds and keep
  // the optimized scratch/history internals directly testable.
#ifdef DLIO_ENABLE_TEST_ACCESS
  int debugNearestRingForElevationLinear(float elevation) const {
    return nearestRingForElevationLinear(elevation);
  }
  int debugNearestRingForElevationCached(float elevation) const {
    return nearestRingForElevation(elevation);
  }
  bool debugRingLookupMatchesLinear(const std::vector<float>& elevations) const {
    for (const float elevation : elevations) {
      if (nearestRingForElevation(elevation) !=
          nearestRingForElevationLinear(elevation)) {
        return false;
      }
    }
    return true;
  }
  std::size_t debugRingLookupCacheSize() const {
    rebuildRingElevationLookupCache();
    return ring_elevation_lookup_.size();
  }
  std::size_t debugValidIndexCount() const {
    return scratch_.valid_indices.size();
  }
  std::size_t debugPublishableIndexCount() const {
    return scratch_.publishable_indices.size();
  }
  int debugPreScanValidPointCount() const {
    return scratch_.projection_pre_scan.valid_point_count;
  }
  int debugPreScanValidRingCount() const {
    return scratch_.projection_pre_scan.valid_ring_count;
  }
  int debugPreScanUniqueRingCount() const {
    return scratch_.projection_pre_scan.unique_ring_count;
  }
  int debugPreScanTimestampCount() const {
    return scratch_.projection_pre_scan.timestamp_count;
  }
  bool debugPreScanRingSeen(std::size_t ring) const {
    return ring < scratch_.projection_pre_scan.ring_seen.size() &&
           scratch_.projection_pre_scan.ring_seen[ring];
  }
  std::size_t debugLastProjectionTouchedCellCount() const {
    return scratch_.projection.touched_cell_indices.size();
  }
  std::size_t debugLastHistoryValidCellCount() const {
    if (history_order_.empty()) {
      return 0U;
    }
    return history_slots_[history_order_.back()].valid_cell_indices.size();
  }
	  bool debugLastHistoryCellValid(int row, int col) const {
	    std::size_t idx = 0U;
	    if (history_order_.empty() ||
	        !projectionIndex(row,
	                         col,
	                         config_.projection_cols,
	                         history_slots_[history_order_.back()].cells.size(),
	                         idx)) {
	      return false;
	    }
	    const DepthFrame& frame = history_slots_[history_order_.back()];
	    return idx < frame.cells.size() && frame.cells[idx].valid;
	  }
  std::size_t debugHistoryActiveFrameCount() const {
    return history_order_.size();
  }
  std::size_t debugHistorySlotCount() const {
    return history_slots_.size();
  }
  std::size_t debugSeedIndexCount() const {
    return scratch_.seed_indices.size();
  }
  std::size_t debugSeedMaskCount() const {
    std::size_t count = 0U;
    for (const auto value : scratch_.seed_points) {
      if (value) {
        ++count;
      }
    }
    return count;
  }
  bool debugStaticSupportedPoint(std::size_t index) const {
    return index < scratch_.static_supported_points.size() &&
           scratch_.static_supported_points[index];
  }
  bool debugActiveTrackPoint(std::size_t index) const {
    return index < scratch_.active_track_points.size() &&
           scratch_.active_track_points[index];
  }
#endif

private:
  struct VoxelKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const VoxelKey& other) const {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& key) const;
  };

  struct VoxelState {
    int last_seen_scan = -1;
    int last_dynamic_scan = -1;
    int hit_bits_scan = -1;
    std::uint64_t hit_bits = 0;
    int dynamic_score = 0;
  };

  struct ProjectionCell {
    bool valid = false;
    std::size_t point_index = 0;
    float range = 0.0f;
    int row = -1;
    int col = -1;
    int ring = -1;
    float yaw = 0.0f;
    float elevation = 0.0f;
    double timestamp = 0.0;
    Eigen::Vector3f point_lidar = Eigen::Vector3f::Zero();
    Eigen::Vector3f point_map = Eigen::Vector3f::Zero();
  };

  struct ProjectionFrame {
    int rows = 0;
    int cols = 0;
    bool used_ring_fallback = false;
    bool used_timestamp_fallback = false;
    bool used_native_projection = false;
    std::vector<ProjectionCell> cells;
    std::vector<int> point_cell_indices;
    std::vector<int> touched_cell_indices;
  };

  struct PointMeta {
    Eigen::Vector3f corrected = Eigen::Vector3f::Zero();
    Eigen::Vector3f lidar = Eigen::Vector3f::Zero();
    Eigen::Vector4f map_h = Eigen::Vector4f::Zero();
    VoxelKey key;
    int source_ring = -1;
    int projection_cell = -1;
    float projection_range = 0.0f;
    float projection_yaw = 0.0f;
    float projection_elevation = 0.0f;
    bool valid = false;
  };

  struct DepthCell {
    bool valid = false;
    float min_range_all = 0.0f;
    float max_range_all = 0.0f;
    float min_range_static = 0.0f;
    float max_range_static = 0.0f;
    int static_points = 0;
    int all_points = 0;
  };

  struct DepthFrame {
    int scan_index = 0;
    double stamp = 0.0;
    Eigen::Matrix4f T_map_lidar = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T_lidar_map = Eigen::Matrix4f::Identity();
    std::vector<DepthCell> cells;
    std::vector<int> valid_cell_indices;
  };

  struct PointEvidence {
    int case1_votes = 0;
    int case2_votes = 0;
    int case3_votes = 0;
    int static_votes = 0;
    int compared_frames = 0;
  };

  struct Cluster {
    std::vector<std::size_t> point_indices;
    std::vector<VoxelKey> voxels;
    Eigen::Vector3f min = Eigen::Vector3f::Zero();
    Eigen::Vector3f max = Eigen::Vector3f::Zero();
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    VoxelKey min_key;
    VoxelKey max_key;
    int seed_count = 0;
    int case1_count = 0;
    int case2_count = 0;
    int case3_count = 0;
    int static_count = 0;
    bool accepted = false;
    bool strong = false;
    bool ground_like = false;
    bool edge_like = false;
    bool wall_like = false;
  };

  struct Track {
    int id = -1;
    Eigen::Vector3f min = Eigen::Vector3f::Zero();
    Eigen::Vector3f max = Eigen::Vector3f::Zero();
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
    VoxelKey min_key;
    VoxelKey max_key;
    std::vector<VoxelKey> mask_keys;
    int age = 0;
    int hits = 0;
    int missed = 0;
    int ttl_remaining = 0;
    int last_seen_scan = -1;
    float confidence = 0.0f;
    bool confirmed = false;
    bool stopped = false;
  };

  struct RingElevationEntry {
    float elevation = 0.0f;
    int ring = -1;
  };

  struct ProjectionPreScanStats {
    int valid_point_count = 0;
    int valid_ring_count = 0;
    int unique_ring_count = 0;
    int timestamp_count = 0;
    std::array<bool, 128> ring_seen{};

    void reset() {
      valid_point_count = 0;
      valid_ring_count = 0;
      unique_ring_count = 0;
      timestamp_count = 0;
      ring_seen.fill(false);
    }
  };

  using VoxelMap = std::unordered_map<VoxelKey, VoxelState, VoxelKeyHash>;
  using ScanVoxelMap = std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelKeyHash>;
  using PointMask = std::vector<std::uint8_t>;

  struct Scratch {
    std::vector<Eigen::Vector3f> corrected_points;
    std::vector<Eigen::Vector3f> points_lidar;
    PointMask publishable_points;
    PointMask valid_points;
    std::vector<std::size_t> publishable_indices;
    std::vector<std::size_t> valid_indices;
    std::vector<std::size_t> seed_indices;
    std::vector<VoxelKey> point_keys;
    std::vector<PointMeta> point_meta;
    PointMask static_supported_points;
    PointMask active_track_points;
    PointMask case1_points;
    PointMask case2_points;
    PointMask case3_points;
    PointMask seed_points;
    PointMask cluster_points;
    PointMask track_points;
    PointMask stopped_points;
    PointMask removed_points;
    ScanVoxelMap scan_voxels;
    ProjectionFrame projection;
    ProjectionPreScanStats projection_pre_scan;
  };

  bool validConfig() const;
	  bool pointFinite(const PointType& point) const;
	  bool pointInRange(const Eigen::Vector3f& point_map,
	                    const Eigen::Vector3f& sensor_origin_map) const;
	  bool tryKeyFromPoint(const Eigen::Vector3f& point_map, VoxelKey& key) const;
	  VoxelKey keyFromPoint(const Eigen::Vector3f& point_map) const;
	  bool projectionIndex(int row,
	                       int col,
	                       int cols,
	                       std::size_t cell_count,
	                       std::size_t& index) const;
  std::uint64_t staticWindowMask() const;
  int staticHitCount(const VoxelState& voxel) const;
  bool voxelIsStatic(const VoxelState& voxel) const;
  bool voxelIsDynamic(const VoxelState& voxel) const;
  void markStatic(const VoxelKey& key);
  void markDynamic(const VoxelKey& key);
  void pruneVoxels(const Eigen::Vector3f& sensor_origin_map);
  void updateRingElevationModel(const pcl::PointCloud<PointType>::ConstPtr& cloud,
                                const std::vector<Eigen::Vector3f>& points_lidar,
                                const std::vector<std::size_t>& valid_indices,
                                const ProjectionPreScanStats& pre_scan);
  void invalidateRingElevationLookupCache();
  void rebuildRingElevationLookupCache() const;
  int nearestRingForElevationLinear(float elevation) const;
  int nearestRingForElevation(float elevation) const;
  bool projectPoint(const Eigen::Vector3f& point_lidar,
                    int source_ring,
                    bool prefer_source_ring,
                    int& row,
                    int& col,
                    float& range,
                    float& yaw,
                    float& elevation,
                    bool& ring_fallback) const;
  void buildProjection(
      const pcl::PointCloud<PointType>::ConstPtr& cloud_map_before_correction,
      const std::vector<Eigen::Vector3f>& corrected_points,
      const std::vector<Eigen::Vector3f>& points_lidar,
      const std::vector<std::size_t>& valid_indices,
      const ProjectionPreScanStats& pre_scan,
      std::vector<PointMeta>& point_meta,
      ProjectionFrame& projection) const;
  void appendDepthFrame(const ProjectionFrame& projection,
                        const PointMask& removed_points,
                        double stamp,
                        const Eigen::Ref<const Eigen::Matrix4f>& T_map_lidar);
  void ensureHistorySlots(std::size_t cell_count);
  void resetDepthFrameCells(DepthFrame& frame);
  std::size_t nextHistorySlot();
  PointEvidence evaluatePointEvidence(const PointMeta& point,
                                      bool in_track_mask) const;
  std::vector<Cluster> buildClusters(
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
      const ProjectionFrame& projection) const;
  bool clusterLooksGroundLike(const Cluster& cluster) const;
  bool clusterLooksEdgeLike(const Cluster& cluster) const;
  bool clusterLooksWallLike(const Cluster& cluster) const;
  float bboxOverlapRatio(const Cluster& cluster, const Track& track) const;
  std::size_t updateTracks(const std::vector<Cluster>& clusters);
  void pruneTracks();
  void rebuildActiveTrackMask();
  bool pointInActiveTrackMask(const Eigen::Vector3f& point_map) const;
  std::vector<VoxelKey> paddedClusterMaskKeys(const Cluster& cluster) const;
  std::vector<DebugBox> activeTrackBoxes() const;
  std::size_t confirmedTrackCount() const;

  Config config_;
  VoxelMap voxels_;
  std::vector<DepthFrame> history_slots_;
  std::deque<std::size_t> history_order_;
  std::vector<float> ring_elevation_sum_;
  std::vector<int> ring_elevation_count_;
  mutable std::vector<RingElevationEntry> ring_elevation_lookup_;
  mutable bool ring_elevation_lookup_dirty_ = true;
  std::vector<Track> tracks_;
  std::unordered_set<VoxelKey, VoxelKeyHash> active_track_mask_;
  Scratch scratch_;
  int scan_index_ = 0;
  int next_track_id_ = 1;
};

}  // namespace dlio
