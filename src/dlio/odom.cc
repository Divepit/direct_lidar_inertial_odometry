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

#include "dlio/odom.h"
#include "dlio/scan_time_bounds.h"
#include "dlio/utils.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>
#include <sstream>
#include <unordered_map>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rclcpp/qos.hpp"

namespace {

std::string shellQuote(const std::string& value) {
  std::string quoted = "'";
  for (const char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

Eigen::Vector3d quaternionToRollPitchYaw(const Eigen::Quaternionf& q_in) {
  Eigen::Quaterniond q(
      static_cast<double>(q_in.w()),
      static_cast<double>(q_in.x()),
      static_cast<double>(q_in.y()),
      static_cast<double>(q_in.z()));
  q.normalize();

  const double sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
  const double cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
  const double roll = std::atan2(sinr_cosp, cosr_cosp);

  const double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
  const double pitch = std::asin(std::clamp(sinp, -1.0, 1.0));

  const double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
  const double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
  const double yaw = std::atan2(siny_cosp, cosy_cosp);

  return Eigen::Vector3d(roll, pitch, yaw);
}

const std::array<const char*, 12>& runStatsPlotFiles() {
  static const std::array<const char*, 12> files{{
      "pose_position.pdf",
      "pose_position.png",
      "pose_orientation_rpy.pdf",
      "pose_orientation_rpy.png",
      "twist_linear_body.pdf",
      "twist_linear_body.png",
      "twist_angular_body.pdf",
      "twist_angular_body.png",
      "bias_accel.pdf",
      "bias_accel.png",
      "bias_gyro.pdf",
      "bias_gyro.png",
  }};
  return files;
}

template <typename PublisherPtrT>
inline bool hasSubscribers(const PublisherPtrT& pub) {
  return pub &&
         (pub->get_subscription_count() > 0 ||
          pub->get_intra_process_subscription_count() > 0);
}

struct NormalScatterStats {
  Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
  int valid_normals = 0;
  double total_weight = 0.0;
};

struct MetadataVoxelKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const MetadataVoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct MetadataVoxelKeyHash {
  std::size_t operator()(const MetadataVoxelKey& key) const {
    const std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::int64_t>(key.x));
    const std::uint64_t y = static_cast<std::uint64_t>(static_cast<std::int64_t>(key.y));
    const std::uint64_t z = static_cast<std::uint64_t>(static_cast<std::int64_t>(key.z));
    std::uint64_t h = x * 73856093ULL;
    h ^= y * 19349663ULL;
    h ^= z * 83492791ULL;
    return static_cast<std::size_t>(h);
  }
};

inline MetadataVoxelKey metadataKeyForPoint(const PointType& point, const double resolution) {
  const double inv = 1.0 / std::max(resolution, 1.0e-6);
  return MetadataVoxelKey{
      static_cast<int>(std::floor(static_cast<double>(point.x) * inv)),
      static_cast<int>(std::floor(static_cast<double>(point.y) * inv)),
      static_cast<int>(std::floor(static_cast<double>(point.z) * inv))};
}

inline void logTranslationSpectrumAlways(const rclcpp::Logger& logger,
                                         const Eigen::Vector3d& evals,
                                         const Eigen::Matrix3d& evecs) {
  RCLCPP_INFO(
      logger,
      "Translation observability from scan normal spread: eigvals(sorted asc) = [%.3f %.3f %.3f]",
      evals(0), evals(1), evals(2));

  for (int i = 0; i < 3; ++i) {
    const Eigen::Vector3d v = evecs.col(i).normalized();
    RCLCPP_INFO(
        logger,
        "  dir[%d] = [%.6f %.6f %.6f]",
        i,
        v.x(), v.y(), v.z());
  }
}

inline std::array<bool, 3> classifyWeakDirections(const Eigen::Vector3d& evals,
                                                  const double abs_threshold) {
  std::array<bool, 3> weak{{false, false, false}};
  for (int i = 0; i < 3; ++i) {
    const double lambda_i = evals(i);
    weak[i] = lambda_i <= abs_threshold;
  }
  return weak;
}

inline void sortEigenpairsAscending(const Eigen::Vector3d& evals_in,
                                    const Eigen::Matrix3d& evecs_in,
                                    Eigen::Vector3d& evals_out,
                                    Eigen::Matrix3d& evecs_out) {
  std::array<int, 3> idx{{0, 1, 2}};
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return evals_in(a) < evals_in(b);
  });

  for (int k = 0; k < 3; ++k) {
    evals_out(k) = evals_in(idx[k]);
    evecs_out.col(k) = evecs_in.col(idx[k]).normalized();
  }
}

inline double surfacePlanarityWeight(const Eigen::Vector3d& evals) {
  constexpr double kEps = 1e-12;
  const double lambda_max = std::max(evals(2), kEps);
  const double weight = (evals(1) - evals(0)) / lambda_max;
  if (!std::isfinite(weight)) {
    return 0.0;
  }
  return std::clamp(weight, 0.0, 1.0);
}

inline NormalScatterStats buildNormalScatterMatrix(const nano_gicp::CovarianceList& covs) {
  NormalScatterStats stats;

  for (const auto& cov4 : covs) {
    const Eigen::Matrix3d cov =
        0.5 * (cov4.block<3, 3>(0, 0) + cov4.block<3, 3>(0, 0).transpose());
    if (!cov.allFinite()) {
      continue;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    if (eig.info() != Eigen::Success) {
      continue;
    }

    const Eigen::Vector3d evals = eig.eigenvalues();
    const double weight = surfacePlanarityWeight(evals);
    if (weight <= 1e-6) {
      continue;
    }

    Eigen::Vector3d normal = eig.eigenvectors().col(0);
    if (!normal.allFinite()) {
      continue;
    }

    normal.normalize();
    stats.scatter.noalias() += weight * (normal * normal.transpose());
    stats.total_weight += weight;
    ++stats.valid_normals;
  }

  return stats;
}

}  // namespace

dlio::OdomNode::OdomNode() : Node("dlio_odom_node") {

  this->getParams();
  this->initializeRunStats();
  this->m_detector_filter_.configure(this->m_detector_filter_config_);

  this->dlio_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
  else {this->imu_calibrated = true;}
  this->deskew_status = false;
  this->deskew_size = 0;

  this->lidar_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto lidar_sub_opt = rclcpp::SubscriptionOptions();
  lidar_sub_opt.callback_group = this->lidar_cb_group;

  // Reliable transport with bounded history to absorb bursts before callback queuing.
  const size_t lidar_qos_depth = this->pointcloud_queue_size_;
  auto qosLiDAR = rclcpp::QoS(rclcpp::KeepLast(lidar_qos_depth))
              .reliability(rclcpp::ReliabilityPolicy::Reliable)
              .durability(rclcpp::DurabilityPolicy::Volatile);
  lidar_sub = create_subscription<sensor_msgs::msg::PointCloud2>(
      "pointcloud", qosLiDAR,
      std::bind(&dlio::OdomNode::callbackPointCloud, this, std::placeholders::_1),
      lidar_sub_opt);

  this->imu_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  // SensorDataQoS default expanded explicitly:
  // history=keep_last, depth=5, reliability=best_effort, durability=volatile.
  auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(200))
                    .reliability(rclcpp::ReliabilityPolicy::BestEffort)
                    .durability(rclcpp::DurabilityPolicy::Volatile);
  auto imu_sub_opt = rclcpp::SubscriptionOptions();
  // SubscriptionOptions default callback_group is nullptr (node default group).
  imu_sub_opt.callback_group = this->imu_cb_group;
  this->imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", imu_qos,
      std::bind(&dlio::OdomNode::callbackImu, this, std::placeholders::_1), imu_sub_opt);

  this->reset_srv_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->reset_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "dlio/reset",
      std::bind(&dlio::OdomNode::resetService, this, std::placeholders::_1, std::placeholders::_2),
      rclcpp::ServicesQoS(),
      this->reset_srv_cb_group_);

  this->map_reset_client_ = this->create_client<std_srvs::srv::Trigger>("dlio/reset_map");

  this->odom_pub     = this->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  this->pose_pub     = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 1);
  this->path_pub     = this->create_publisher<nav_msgs::msg::Path>("path_map", 1);
  this->path_odom_pub = this->create_publisher<nav_msgs::msg::Path>("path_odom", 1);
  this->path_map_prop_pub = this->create_publisher<nav_msgs::msg::Path>("path_map_prop", 1);
  this->kf_pose_pub  = this->create_publisher<geometry_msgs::msg::PoseArray>("kf_pose", 1);

  rclcpp::QoS reliable_qos(rclcpp::KeepLast(10));
  reliable_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  this->odom_map_pub = this->create_publisher<nav_msgs::msg::Odometry>("map_pose_inverted", reliable_qos);
  this->odom_baselink_pub = this->create_publisher<nav_msgs::msg::Odometry>("map_pose", reliable_qos);

  auto best_effort_qos = rclcpp::QoS(100)
                             .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                             .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE)
                             .history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);

  this->kf_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("kf_cloud", best_effort_qos);
  this->deskewed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed", best_effort_qos);
  this->deskewed_not_transformed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed_not_transformed", best_effort_qos);
  this->deskewed_map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed_and_transformed_to_map", best_effort_qos);
  this->dynamic_removed_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_removed", best_effort_qos);

  this->br = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

  // Markers use reliable transport to avoid stale RViz artifacts from dropped DELETE updates.
  rclcpp::QoS marker_qos(rclcpp::KeepLast(50));
  marker_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  marker_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
  marker_qos.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  this->pub_lin_vel_marker_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "markers/velocity_linear", marker_qos);
  this->pub_ang_vel_marker_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "markers/velocity_angular", marker_qos);
  this->pub_corr_marker_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "markers/correction", marker_qos);
  this->pub_degen_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "markers/degeneracy_directions", marker_qos);
  this->corr_marker_points_.reserve(
      2U * static_cast<std::size_t>(std::max(1, this->viz_corr_max_segments_)));

  this->pointcloud_worker_ = std::thread([this]{ pointCloudWorkerLoop(); });
  this->pub_worker_ = std::thread([this]{ workerLoop(); });

  {
    std::lock_guard<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    this->keyframes.reserve(this->kMaxKeyframes);
  }

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  if (!this->imu_calibrate_) {
    this->captureInitialImuBaseline(
        Eigen::Vector3f(0.f, 0.f, static_cast<float>(std::abs(this->gravity_))),
        this->state.q,
        this->state.b.accel,
        this->state.b.gyro);
  }

  this->degen_prev_trans_dirs_map_[0] = Eigen::Vector3d::UnitX();
  this->degen_prev_trans_dirs_map_[1] = Eigen::Vector3d::UnitY();
  this->degen_prev_trans_dirs_map_[2] = Eigen::Vector3d::UnitZ();

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->first_imu_stamp = 0.;
  this->prev_imu_stamp = 0.;

  this->original_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->current_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->registration_scan = this->current_scan;
  this->keyframe_mapping_scan = this->current_scan;
  this->dynamic_removed_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
  this->submap_cloud = std::make_shared<const pcl::PointCloud<PointType>>();

  this->num_processed_keyframes = 0;

  this->submap_hasChanged = true;
  this->submap_kf_idx_prev.clear();

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed = 0.0;

  this->convex_hull.setDimension(3);
  this->concave_hull.setDimension(3);
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
  this->concave_hull.setKeepInformation(true);

  this->gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);
  this->gicp.setNumThreads(this->num_threads_);

  this->gicp_temp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_temp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_temp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_temp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_temp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_temp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);
  this->gicp_temp.setNumThreads(this->num_threads_);

  this->gicp_self_.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_self_.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_self_.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_self_.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_self_.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_self_.setInitialLambdaFactor(this->gicp_init_lambda_factor_);
  this->gicp_self_.setNumThreads(this->num_threads_);

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(temp, true);
  this->gicp.setSearchMethodTarget(temp, true);
  this->gicp_temp.setSearchMethodSource(temp, true);
  this->gicp_temp.setSearchMethodTarget(temp, true);
  this->gicp_self_.setSearchMethodSource(temp, true);
  this->gicp_self_.setSearchMethodTarget(temp, true);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  const float crop_size = static_cast<float>(this->crop_size_);
  const float vf_res = static_cast<float>(this->vf_res_);
  this->crop.setMin(Eigen::Vector4f(-crop_size, -crop_size, -crop_size, 1.0f));
  this->crop.setMax(Eigen::Vector4f(crop_size, crop_size, crop_size, 1.0f));

  this->voxel.setLeafSize(vf_res, vf_res, vf_res);

  {
    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    this->metrics.spaciousness.push_back(0.);
    this->metrics.density.push_back(static_cast<float>(this->gicp_max_corr_dist_));
    // Start with a neutral model-deviation scale so adaptive gating
    // has a stable value before the first completed registration.
    this->metrics.motion_deviation.push_back(static_cast<float>(this->gicp_max_corr_dist_));
  }

  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while (file != nullptr && fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "processor", 9) == 0) this->numProcessors++;
  }
  if (file != nullptr) {
    fclose(file);
  }

}

inline bool hasWeakDirection(const std::array<bool, 3>& weak) {
  return weak[0] || weak[1] || weak[2];
}

inline void logTranslationDegeneracy(const rclcpp::Logger& logger,
                                     const Eigen::Vector3d& evals,
                                     const Eigen::Matrix3d& evecs,
                                     const std::array<bool, 3>& weak) {
RCLCPP_WARN(
    logger,
    "Translation degeneracy from scan normal spread detected. eigvals(sorted asc) = [%.3f %.3f %.3f]",
    evals(0), evals(1), evals(2));

  for (int i = 0; i < 3; ++i) {
    const Eigen::Vector3d v = evecs.col(i).normalized();
    RCLCPP_WARN(
        logger,
        "  dir[%d] = [%.6f %.6f %.6f]  weak=%s",
        i,
        v.x(), v.y(), v.z(),
        weak[i] ? "true" : "false");
  }
}

dlio::OdomNode::~OdomNode() {
  this->requestStop();

  if (pointcloud_worker_.joinable()) pointcloud_worker_.join();
  if (pub_worker_.joinable()) pub_worker_.join();
  if (submap_future.valid()) submap_future.wait();
  if (debug_future_.valid()) debug_future_.wait();

  this->generateRunStatsPlots();
  this->closeRunStats();

}

void dlio::OdomNode::requestStop() {
  const bool already_stopping = stop_.exchange(true, std::memory_order_relaxed);
  if (!already_stopping) {
    RCLCPP_INFO(this->get_logger(),
                "\033[38;5;214m[SHUTDOWN] Odom node stopping. Draining workers and exiting...\033[0m");
  }

  {
    std::lock_guard<std::mutex> lk(this->main_loop_running_mutex);
    this->main_loop_running = false;
  }
  {
    std::lock_guard<std::mutex> lk(this->q_mtx_);
    this->q_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(this->pc_q_mtx_);
    this->pc_q_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    this->reset_requested_ = false;
    if (this->reset_in_progress_.exchange(false)) {
      this->reset_succeeded_ = false;
      this->reset_status_message_ = "Shutdown requested.";
    } else if (this->reset_status_message_.empty()) {
      this->reset_status_message_ = "Shutdown requested.";
    }
  }

  pc_q_cv_.notify_all();
  q_cv_.notify_all();
  cv_imu_stamp.notify_all();
  submap_build_cv.notify_all();
  reset_done_cv_.notify_all();
}

bool dlio::OdomNode::shouldStop() {
  if (stop_.load(std::memory_order_relaxed)) {
    return true;
  }

  const auto context = this->get_node_base_interface()->get_context();
  return !context || !context->is_valid();
}

void dlio::OdomNode::captureInitialImuBaseline(
    const Eigen::Vector3f& gravity_vec,
    const Eigen::Quaternionf& gravity_align_q,
    const Eigen::Vector3f& accel_bias,
    const Eigen::Vector3f& gyro_bias) {
  if (this->initial_imu_baseline_.valid) {
    return;
  }

  Eigen::Vector3f gravity_vec_safe = gravity_vec;
  if (!gravity_vec_safe.allFinite() || gravity_vec_safe.norm() <= 1e-6f) {
    gravity_vec_safe = Eigen::Vector3f(0.f, 0.f, static_cast<float>(std::abs(this->gravity_)));
  }

  this->initial_imu_baseline_.gravity_vec = gravity_vec_safe;
  this->initial_imu_baseline_.gravity_norm = gravity_vec_safe.norm();
  this->initial_imu_baseline_.gravity_align_q = gravity_align_q.normalized();
  this->initial_imu_baseline_.accel_bias = accel_bias;
  this->initial_imu_baseline_.gyro_bias = gyro_bias;
  this->initial_imu_baseline_.valid = true;

  RCLCPP_INFO(
      this->get_logger(),
      "Captured initial IMU baseline: |g|=%.6f accel_bias=[%.6f %.6f %.6f] gyro_bias=[%.6f %.6f %.6f]",
      this->initial_imu_baseline_.gravity_norm,
      this->initial_imu_baseline_.accel_bias.x(),
      this->initial_imu_baseline_.accel_bias.y(),
      this->initial_imu_baseline_.accel_bias.z(),
      this->initial_imu_baseline_.gyro_bias.x(),
      this->initial_imu_baseline_.gyro_bias.y(),
      this->initial_imu_baseline_.gyro_bias.z());
}

bool dlio::OdomNode::beginPendingReset() {
  std::lock_guard<std::mutex> lock(this->reset_mutex_);
  if (this->shouldStop() || this->reset_in_progress_.load() || !this->reset_requested_) {
    return false;
  }

  this->reset_requested_ = false;
  this->reset_in_progress_.store(true);
  this->reset_succeeded_ = false;
  this->reset_status_message_.clear();
  return true;
}

void dlio::OdomNode::finishPendingReset(bool success, const std::string& message) {
  {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    if (this->shouldStop()) {
      success = false;
      this->reset_status_message_ = "Shutdown requested.";
    } else {
      this->reset_status_message_ = message;
    }
    this->reset_in_progress_.store(false);
    this->reset_succeeded_ = success;
  }
  this->reset_done_cv_.notify_all();
}

void dlio::OdomNode::requestMapReset(const std::string& origin) {
  if (this->shouldStop()) {
    return;
  }

  if (!this->map_reset_client_) {
    return;
  }

  if (!this->map_reset_client_->service_is_ready()) {
    RCLCPP_WARN(this->get_logger(),
                "[RESET:%s] dlio/reset_map service not ready — map was NOT cleared.",
                origin.c_str());
    return;
  }

  auto map_req = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto map_future = this->map_reset_client_->async_send_request(map_req);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!this->shouldStop()) {
    if (map_future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
      auto map_res = map_future.get();
      if (map_res->success) {
        RCLCPP_INFO(this->get_logger(),
                    "\033[33m[RESET:%s] Map reset confirmed: %s\033[0m",
                    origin.c_str(), map_res->message.c_str());
      } else {
        RCLCPP_WARN(this->get_logger(),
                    "[RESET:%s] Map reset returned failure: %s",
                    origin.c_str(), map_res->message.c_str());
      }
      return;
    }

    if (std::chrono::steady_clock::now() >= deadline) {
      RCLCPP_WARN(this->get_logger(),
                  "[RESET:%s] Map reset service call timed out after 3 s.",
                  origin.c_str());
      return;
    }
  }
}

bool dlio::OdomNode::triggerInternalReset(const std::string& reason) {
  bool owns_reset_request = false;
  {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    if (!this->initial_imu_baseline_.valid) {
      RCLCPP_ERROR(this->get_logger(),
                   "[RESET] Internal reset rejected: initial IMU baseline has not been captured yet.");
      return false;
    }

    if (this->reset_in_progress_.load()) {
      RCLCPP_WARN(this->get_logger(),
                  "[RESET] Internal reset requested while another reset is already in progress. "
                  "Executing worker-side reset immediately.");
    } else {
      owns_reset_request = true;
      this->reset_requested_ = false;
      this->reset_in_progress_.store(true);
      this->reset_succeeded_ = false;
      this->reset_status_message_.clear();
    }
  }

  this->degen_consecutive_hits_ = 0;

  RCLCPP_ERROR(this->get_logger(),
               "\033[31m[RESET] Internal self-reset triggered: %s\033[0m",
               reason.c_str());

  this->performReset();
  if (owns_reset_request) {
    this->requestMapReset("self");
  }
  return true;
}

void dlio::OdomNode::performReset() {
  // Called exclusively from pointCloudWorkerLoop() at a safe scan boundary.
  // The service thread is blocked on reset_done_cv_ for the duration of this function.
  RCLCPP_INFO(this->get_logger(),
              "\033[33m[RESET] ============ BEGIN SYSTEM RESET ============\033[0m");

  // -----------------------------------------------------------------------
  // Gap 3 fix (step 1/2): release main_loop_running BEFORE waiting on the
  // submap future, so pauseSubmapBuildIfNeeded() can unblock.
  // -----------------------------------------------------------------------
  {
    std::lock_guard<std::mutex> lk(this->main_loop_running_mutex);
    this->main_loop_running = false;
  }
  this->submap_build_cv.notify_all();
  RCLCPP_INFO(this->get_logger(), "[RESET] main_loop_running cleared, submap_build_cv notified.");

  // -----------------------------------------------------------------------
  // Gap 3 fix (step 2/2): now safe to wait for the async submap build.
  // -----------------------------------------------------------------------
  if (this->submap_future.valid() &&
      this->submap_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    RCLCPP_INFO(this->get_logger(), "[RESET] Waiting for in-flight submap build to finish...");
    this->submap_future.wait();
    RCLCPP_INFO(this->get_logger(), "[RESET] Submap build joined.");
  }

  // -----------------------------------------------------------------------
  // Gap 4 fix: flush the publish queue so the pub_worker_ doesn't emit
  // stale odometry after the state is cleared.
  // -----------------------------------------------------------------------
  {
    std::lock_guard<std::mutex> lk(this->q_mtx_);
    const std::size_t dropped = this->q_.size();
    this->q_.clear();
    if (dropped > 0) {
      RCLCPP_INFO(this->get_logger(),
                  "[RESET] Flushed %zu stale publish job(s) from publish queue.", dropped);
    }
  }
  this->q_cv_.notify_all();

  // -----------------------------------------------------------------------
  // IMU buffer and IMU-tracking state
  // -----------------------------------------------------------------------
  {
    std::lock_guard<std::mutex> lk(this->mtx_imu);
    this->imu_buffer.clear();
  }
  this->first_imu_stamp              = 0.0;
  this->prev_imu_stamp               = 0.0;
  this->imu_transform_prev_stamp_    = 0.0;
  this->imu_transform_prev_valid_    = false;
  this->imu_transform_ang_vel_prev_  = Eigen::Vector3f::Zero();
  this->first_imu_received           = false;
  RCLCPP_INFO(this->get_logger(), "[RESET] IMU buffer cleared, IMU stamps zeroed.");

  // -----------------------------------------------------------------------
  // Pose / velocity / bias — restored from the immutable startup baseline.
  // Mirror the IMU calibration initialization path, but reuse the saved
  // gravity vector and saved biases instead of estimating them again.
  // -----------------------------------------------------------------------
  const float gravity_norm = (this->initial_imu_baseline_.gravity_norm > 1e-6f)
      ? this->initial_imu_baseline_.gravity_norm
      : static_cast<float>(std::abs(this->gravity_));
  const float gravity_scalar =
      (this->gravity_ < 0.0) ? -gravity_norm : gravity_norm;

  Eigen::Vector3f reset_gravity_vec = this->initial_imu_baseline_.gravity_vec;
  if (!reset_gravity_vec.allFinite() || reset_gravity_vec.norm() <= 1e-6f) {
    reset_gravity_vec = Eigen::Vector3f(0.f, 0.f, std::abs(gravity_scalar));
  } else {
    reset_gravity_vec = reset_gravity_vec.normalized() * gravity_norm;
  }

  Eigen::Quaternionf reset_gravity_align_q = this->initial_imu_baseline_.gravity_align_q;
  if (this->gravity_align_) {
    reset_gravity_align_q = Eigen::Quaternionf::FromTwoVectors(
        reset_gravity_vec, Eigen::Vector3f(0.f, 0.f, gravity_scalar));
  }
  reset_gravity_align_q.normalize();
  this->gravity_ = gravity_scalar;

  {
    std::lock_guard<std::mutex> lk(this->geo.mtx);
    this->state.p               = Eigen::Vector3f::Zero();
    this->state.q               = reset_gravity_align_q;
    this->state.v.lin.b         = Eigen::Vector3f::Zero();
    this->state.v.lin.w         = Eigen::Vector3f::Zero();
    this->state.v.ang.b         = Eigen::Vector3f::Zero();
    this->state.v.ang.w         = Eigen::Vector3f::Zero();
    this->state.b.accel         = this->initial_imu_baseline_.accel_bias;
    this->state.b.gyro          = this->initial_imu_baseline_.gyro_bias;
    this->geo.first_opt_done    = false;
    this->geo.prev_p            = Eigen::Vector3f::Zero();
    this->geo.prev_q            = reset_gravity_align_q;
    this->geo.prev_vel          = Eigen::Vector3f::Zero();
    this->geo.dp                = 0.0;
    this->geo.dq_deg            = 0.0;
  }
  this->lidarPose.p = Eigen::Vector3f::Zero();
  this->lidarPose.q = reset_gravity_align_q;
  this->T       = Eigen::Matrix4f::Identity();
  this->T.block<3,3>(0,0) = reset_gravity_align_q.toRotationMatrix();
  this->T_prior = this->T;
  this->T_corr  = Eigen::Matrix4f::Identity();
  {
    std::lock_guard<std::mutex> lk(this->mtx_T_map_odom_latest);
    this->T_map_odom_latest    = Eigen::Matrix4f::Identity();
    this->has_T_map_odom_latest = false;
  }
  RCLCPP_INFO(this->get_logger(),
              "[RESET] State restored from baseline: |g|=%.4f  "
              "accel_bias=[%.4f %.4f %.4f]  gyro_bias=[%.4f %.4f %.4f]",
              this->initial_imu_baseline_.gravity_norm,
              this->initial_imu_baseline_.accel_bias.x(),
              this->initial_imu_baseline_.accel_bias.y(),
              this->initial_imu_baseline_.accel_bias.z(),
              this->initial_imu_baseline_.gyro_bias.x(),
              this->initial_imu_baseline_.gyro_bias.y(),
              this->initial_imu_baseline_.gyro_bias.z());

  // -----------------------------------------------------------------------
  // Keyframes and submap
  // -----------------------------------------------------------------------
  {
    std::lock_guard<decltype(this->keyframes_mutex)> lk(this->keyframes_mutex);
    this->keyframes.clear();
  }
  this->num_processed_keyframes = 0;
  this->keyframe_convex.clear();
  this->keyframe_concave.clear();
  this->submap_kf_idx_curr.clear();
  this->submap_kf_idx_prev.clear();
  this->submap_cloud   = std::make_shared<const pcl::PointCloud<PointType>>();
  this->submap_normals = nullptr;
  this->submap_kdtree  = nullptr;
  this->submap_hasChanged   = true;
  this->new_submap_is_ready = false;
  this->gicp.clearSource();
  this->gicp.clearTarget();
  this->gicp_temp.clearTarget();
  this->gicp_self_.clearSource();
  this->gicp_self_.clearTarget();
  RCLCPP_INFO(this->get_logger(), "[RESET] Keyframes, submap, and GICP caches cleared.");

  // -----------------------------------------------------------------------
  // Scan clouds
  // -----------------------------------------------------------------------
  this->original_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->current_scan  = std::make_shared<const pcl::PointCloud<PointType>>();
  this->registration_scan = this->current_scan;
  this->keyframe_mapping_scan = this->current_scan;
  this->dynamic_removed_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
  this->m_detector_filter_.reset();
  this->m_detector_filter_stats_ = MDetectorFilter::Stats{};

  // -----------------------------------------------------------------------
  // Timestamps, trajectory, and computation metrics
  // -----------------------------------------------------------------------
  this->first_scan_stamp = 0.0;
  this->prev_scan_stamp  = 0.0;
  this->scan_stamp       = 0.0;
  this->elapsed_time     = 0.0;
  this->length_traversed = 0.0;
  this->trajectory.clear();
  {
    std::lock_guard<std::mutex> lk(this->mtx_comp_times_);
    this->comp_times.clear();
  }
  {
    std::lock_guard<std::mutex> lk(this->g_metrics_mutex);
    this->metrics.spaciousness.clear();
    this->metrics.density.clear();
    this->metrics.motion_deviation.clear();
    this->metrics.spaciousness.push_back(0.f);
    this->metrics.density.push_back(static_cast<float>(this->gicp_max_corr_dist_));
    this->metrics.motion_deviation.push_back(static_cast<float>(this->gicp_max_corr_dist_));
  }
  this->spaciousness_lpf_initialized_ = false;
  this->density_lpf_initialized_      = false;
  RCLCPP_INFO(this->get_logger(), "[RESET] Timestamps, trajectory, and metrics cleared.");

  // -----------------------------------------------------------------------
  // Re-initialisation flags
  // Biases already restored above; skip re-calibration.
  // -----------------------------------------------------------------------
  this->dlio_initialized   = false;
  this->first_valid_scan   = false;
  this->imu_calibrated     = true;   // keep baseline biases, no re-calib
  RCLCPP_INFO(this->get_logger(),
              "[RESET] Flags reset: dlio_initialized=false, first_valid_scan=false, "
              "imu_calibrated=true (biases retained from baseline).");

  // -----------------------------------------------------------------------
  // Visualisation state
  // -----------------------------------------------------------------------
  this->path_poses_.clear();
  this->path_odom_poses_.clear();
  this->path_map_prop_poses_.clear();
  this->kf_pose_ros.poses.clear();
  this->corr_marker_points_.clear();
  this->degen_info_.valid              = false;
  this->degen_consecutive_hits_        = 0;
  this->degen_prev_dirs_initialized_   = false;
  this->degen_prev_trans_dirs_map_[0]  = Eigen::Vector3d::UnitX();
  this->degen_prev_trans_dirs_map_[1]  = Eigen::Vector3d::UnitY();
  this->degen_prev_trans_dirs_map_[2]  = Eigen::Vector3d::UnitZ();

  // Publish empty paths so RViz clears immediately.
  const rclcpp::Time now = this->now();
  {
    nav_msgs::msg::Path empty_path;
    empty_path.header.stamp    = now;
    empty_path.header.frame_id = this->odom_frame;
    if (hasSubscribers(this->path_pub))          this->path_pub->publish(empty_path);
    if (hasSubscribers(this->path_odom_pub))     this->path_odom_pub->publish(empty_path);
    empty_path.header.frame_id = "dlio_map";
    if (hasSubscribers(this->path_map_prop_pub)) this->path_map_prop_pub->publish(empty_path);
  }
  {
    geometry_msgs::msg::PoseArray empty_poses;
    empty_poses.header.stamp    = now;
    empty_poses.header.frame_id = "dlio_map";
    if (hasSubscribers(this->kf_pose_pub)) this->kf_pose_pub->publish(empty_poses);
  }
  // Delete correction line marker.
  if (hasSubscribers(this->pub_corr_marker_)) {
    visualization_msgs::msg::Marker del;
    del.header.stamp    = now;
    del.header.frame_id = "dlio_map";
    del.ns     = "correction_lines";
    del.id     = 2;
    del.action = visualization_msgs::msg::Marker::DELETE;
    this->pub_corr_marker_->publish(del);
  }
  // Delete degeneracy markers.
  if (hasSubscribers(this->pub_degen_marker_)) {
    visualization_msgs::msg::MarkerArray del_arr;
    for (int i = 0; i < 3; ++i) {
      visualization_msgs::msg::Marker m;
      m.header.stamp    = now;
      m.header.frame_id = "dlio_map";
      m.ns     = "degeneracy_translation";
      m.id     = i;
      m.action = visualization_msgs::msg::Marker::DELETE;
      del_arr.markers.push_back(m);
    }
    this->pub_degen_marker_->publish(del_arr);
  }
  RCLCPP_INFO(this->get_logger(),
              "[RESET] Visualisation state cleared and empty paths/markers published.");

  // -----------------------------------------------------------------------
  // Scan geometry gate: drop incoming scans until one has enough 3D structure
  // in its local surface normals, ensuring the first registration after reset
  // is not started from a nearly planar scene.
  // -----------------------------------------------------------------------
  if (this->restart_gate_enabled_) {
    RCLCPP_INFO(this->get_logger(),
                "\033[33m[RESET] Geometry gate active "
                "(min_normal_scatter_eigenvalue=%.1f). Waiting for a non-degenerate scan...\033[0m",
                this->restart_gate_min_eigenvalue_);

    int dropped = 0;
    while (!this->shouldStop()) {
      PointCloudJob gate_job;
      {
        std::unique_lock<std::mutex> lk(this->pc_q_mtx_);
        this->pc_q_cv_.wait(lk, [this]{
          return this->shouldStop() || !this->pc_q_.empty();
        });
        if (this->shouldStop()) break;
        gate_job = std::move(this->pc_q_.front());
        this->pc_q_.pop_front();
      }

      if (this->scanPassesGeometryGate(gate_job.cloud_msg)) {
        RCLCPP_INFO(this->get_logger(),
                    "\033[32m[RESET] Geometry gate passed after dropping %d scan(s). "
                    "Returning scan to queue for normal processing.\033[0m", dropped);
        {
          std::lock_guard<std::mutex> lk(this->pc_q_mtx_);
          this->pc_q_.push_front(std::move(gate_job));
        }
        this->pc_q_cv_.notify_one();
        break;
      }

      ++dropped;
      RCLCPP_WARN(this->get_logger(),
                  "[RESET] Geometry gate: scan #%d dropped (degenerate).", dropped);
    }

    if (this->shouldStop()) {
      this->finishPendingReset(false, "Node stopped during geometry gate.");
      return;
    }
  }

  // -----------------------------------------------------------------------
  // Signal completion to the waiting service thread.
  // -----------------------------------------------------------------------
  RCLCPP_INFO(this->get_logger(),
              "\033[32m[RESET] ============ ODOM RESET COMPLETE ============\033[0m");
  this->finishPendingReset(true, "Odom reset complete.");
}

bool dlio::OdomNode::scanPassesGeometryGate(
    const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  this->gicp_self_.setRegularizationMethod(nano_gicp::RegularizationMethod::NONE);
  const auto restore_regularization = [this]() {
    this->gicp_self_.setRegularizationMethod(nano_gicp::RegularizationMethod::PLANE);
  };


  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
  pcl::fromROSMsg(*pc, *cloud);
  if (cloud->empty()) {
    restore_regularization();
    return false;
  }

  // Remove points inside the ego-vehicle crop box.
  pcl::CropBox<PointType> crop_gate;
  const float crop_size = static_cast<float>(this->crop_size_);
  crop_gate.setNegative(true);
  crop_gate.setMin(Eigen::Vector4f(-crop_size, -crop_size, -crop_size, 1.0f));
  crop_gate.setMax(Eigen::Vector4f(crop_size, crop_size, crop_size, 1.0f));
  crop_gate.setInputCloud(cloud);
  crop_gate.filter(*cloud);

  // Voxel-downsample to keep the covariance computation fast.
  if (this->vf_use_) {
    pcl::VoxelGrid<PointType> vg;
    const float vf_res = static_cast<float>(this->vf_res_);
    vg.setLeafSize(vf_res, vf_res, vf_res);
    vg.setInputCloud(cloud);
    vg.filter(*cloud);
  }

  if (static_cast<int> (cloud->size()) < this->gicp_min_num_points_) {
    RCLCPP_WARN(this->get_logger(),
                "[GATE] Scan has only %zu points after filtering (min %d) — dropped.",
                cloud->size(), this->gicp_min_num_points_);
    restore_regularization();
    return false;
  }

  this->gicp_self_.setInputSource(cloud);
  if (!this->gicp_self_.calculateSourceCovariances()) {
    RCLCPP_WARN(this->get_logger(), "[GATE] Covariance computation failed — scan dropped.");
    restore_regularization();
    return false;
  }

  const auto covs = this->gicp_self_.getSourceCovariances();
  if (!covs || covs->size() != cloud->size()) {
    RCLCPP_WARN(this->get_logger(), "[GATE] Covariance cache is invalid — scan dropped.");
    restore_regularization();
    return false;
  }

  const NormalScatterStats normal_stats = buildNormalScatterMatrix(*covs);

  if (normal_stats.valid_normals < 3 ||
      normal_stats.total_weight <= 0.0 ||
      !normal_stats.scatter.allFinite()) {
    RCLCPP_WARN(this->get_logger(),
                "[GATE] Normal diversity is ill-defined (valid_normals=%d, total_weight=%.1f) — scan dropped.",
                normal_stats.valid_normals, normal_stats.total_weight);
    restore_regularization();
    return false;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(normal_stats.scatter);
  if (eig.info() != Eigen::Success) {
    RCLCPP_WARN(this->get_logger(), "[GATE] Normal-scatter eigendecomposition failed — scan dropped.");
    restore_regularization();
    return false;
  }

  const double min_eval = eig.eigenvalues().minCoeff();
  const bool passes = (min_eval >= this->restart_gate_min_eigenvalue_);

  RCLCPP_INFO(this->get_logger(),
              "[GATE] Scan normal diversity: eigvals=[%.1f %.1f %.1f]"
              "  valid_normals=%d"
              "  total_weight=%.1f"
              "  min=\033[38;5;214m%.1f\033[0m"
              "  threshold=\033[38;5;214m%.1f\033[0m"
              "  %s",
              eig.eigenvalues()(0), eig.eigenvalues()(1), eig.eigenvalues()(2),
              normal_stats.valid_normals, normal_stats.total_weight,
              min_eval, this->restart_gate_min_eigenvalue_,
              passes ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");

  restore_regularization();

  return passes;
}

void dlio::OdomNode::resetService(
    std::shared_ptr<std_srvs::srv::Trigger::Request>,  // NOLINT(performance-unnecessary-value-param)
    std::shared_ptr<std_srvs::srv::Trigger::Response> res) {  // NOLINT(performance-unnecessary-value-param)
  {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);

    if (this->shouldStop()) {
      res->success = false;
      res->message = "Reset rejected: node is shutting down.";
      return;
    }

    if (!this->initial_imu_baseline_.valid) {
      res->success = false;
      res->message = "Reset rejected: initial IMU baseline has not been captured yet.";
      return;
    }

    if (this->reset_in_progress_.load()) {
      res->success = false;
      res->message = "Reset rejected: another reset is already in progress.";
      return;
    }

    if (this->reset_requested_) {
      res->success = false;
      res->message = "Reset rejected: a reset request is already pending.";
      return;
    }

    this->reset_requested_ = true;
  }

  if (!this->beginPendingReset()) {
    res->success = false;
    res->message = this->shouldStop()
        ? "Reset rejected: node is shutting down."
        : "Reset rejected: failed to transition reset request into the pending state.";
    return;
  }

  RCLCPP_INFO(this->get_logger(), "\033[33m[RESET] Request accepted. Notifying worker thread...\033[0m");

  // Wake both worker waits so they see reset_in_progress_ immediately.
  this->pc_q_cv_.notify_all();
  this->cv_imu_stamp.notify_all();

  // Block until the worker thread calls finishPendingReset().
  {
    std::unique_lock<std::mutex> lock(this->reset_mutex_);
    this->reset_done_cv_.wait(lock, [this]{
      return this->shouldStop() || !this->reset_in_progress_.load();
    });
  }

  if (this->shouldStop()) {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    res->success = false;
    res->message = this->reset_status_message_.empty()
        ? "Reset aborted: node is shutting down."
        : this->reset_status_message_;
    return;
  }

  RCLCPP_INFO(this->get_logger(), "\033[33m[RESET] Odom reset done. Triggering map reset...\033[0m");
  this->requestMapReset("service");

  if (this->shouldStop()) {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    res->success = false;
    res->message = this->reset_status_message_.empty()
        ? "Reset aborted: node is shutting down."
        : this->reset_status_message_;
    return;
  }

  {
    std::lock_guard<std::mutex> lock(this->reset_mutex_);
    res->success = this->reset_succeeded_;
    res->message = this->reset_status_message_;
  }

  RCLCPP_INFO(this->get_logger(),
              "\033[32m[RESET] Full system reset complete. success=%s msg=\"%s\"\033[0m",
              res->success ? "true" : "false", res->message.c_str());
}

void dlio::OdomNode::enqueuePublish(
    pcl::PointCloud<PointType>::ConstPtr cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_all,
    double scanStamp,
    const Eigen::Vector3f& state_p_scan,
    const Eigen::Quaternionf& state_q_scan,
    const Eigen::Vector3f& state_vlin_b_scan,
    const Eigen::Vector3f& state_vang_b_scan) {

  PubJob job;
  job.cloud = std::move(cloud);
  job.T_cloud = T_cloud;
  job.T_all = T_all;
  job.scan_header_stamp = this->scan_header_stamp;
  job.scanStamp = scanStamp;

  // Store the odom-state snapshot from the same scan-time reference as T_all/T_cloud.
  job.state_p_scan = state_p_scan;
  job.state_q_scan = state_q_scan.normalized();
  job.state_vlin_b_scan = state_vlin_b_scan;
  job.state_vang_b_scan = state_vang_b_scan;

  {
    std::lock_guard<std::mutex> lk(q_mtx_);
    q_.push_back(std::move(job));
  }
  q_cv_.notify_one();
}

void dlio::OdomNode::workerLoop() {
  while (!this->shouldStop()) {
    PubJob job;
    {
      std::unique_lock<std::mutex> lk(q_mtx_);
      q_cv_.wait(lk, [this]{ return this->shouldStop() || !q_.empty(); });
      if (this->shouldStop()) break;
      job = std::move(q_.front());
      q_.pop_front();
    }

    publishToROS(job.cloud,
                 job.T_cloud,
                 job.T_all,
                 job.scanStamp,
                 job.state_p_scan,
                 job.state_q_scan,
                 job.state_vlin_b_scan,
                 job.state_vang_b_scan);
  }
}
void dlio::OdomNode::enqueuePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {
  {
    std::lock_guard<std::mutex> lk(pc_q_mtx_);

    // Keep queue bounded; if overloaded, drop the oldest scan and keep recent measurements.
    while (pc_q_.size() >= this->pointcloud_queue_size_) {
      pc_q_.pop_front();
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Pointcloud queue full. Dropping oldest scan.");
    }

    pc_q_.push_back(PointCloudJob{pc});
  }
  pc_q_cv_.notify_one();
}

std::size_t dlio::OdomNode::clearPointCloudQueue(const std::string& reason,
                                                 const bool log_if_dropped) {
  std::size_t dropped = 0U;
  {
    std::lock_guard<std::mutex> lk(this->pc_q_mtx_);
    dropped = this->pc_q_.size();
    this->pc_q_.clear();
  }

  if (log_if_dropped && dropped > 0U) {
    RCLCPP_WARN(this->get_logger(),
                "\033[38;5;214m[STARTUP] Cleared %zu queued pointcloud scan(s): %s\033[0m",
                dropped,
                reason.c_str());
  }

  return dropped;
}

bool dlio::OdomNode::imuBufferCoversRange(const double start_time,
                                          const double end_time) {
  std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);
  return !this->imu_buffer.empty() &&
         this->imu_buffer.front().stamp >= end_time &&
         this->imu_buffer.back().stamp <= start_time;
}

void dlio::OdomNode::pointCloudWorkerLoop() {

  while (!this->shouldStop()) {
    PointCloudJob job;
    {
      std::unique_lock<std::mutex> lk(pc_q_mtx_);
      // Gap 2 fix: predicate now also wakes on reset_in_progress_ so the worker
      // is not left blocked while the service thread waits for it.
      pc_q_cv_.wait(lk, [this]{
        return this->shouldStop()
            || this->reset_in_progress_.load()
            || !pc_q_.empty();
      });

      if (this->shouldStop()) {
        break;
      }

      // Gap 2 fix: if a reset arrived, drain the queue (while we hold the lock)
      // and jump to performReset() before touching any scan data.
      if (this->reset_in_progress_.load()) {
        const std::size_t dropped = pc_q_.size();
        pc_q_.clear();
        lk.unlock();
        if (dropped > 0) {
          RCLCPP_INFO(this->get_logger(),
                      "\033[33m[RESET] Worker: drained %zu queued scan(s) before reset.\033[0m",
                      dropped);
        }
        this->performReset();
        continue;
      }

      job = std::move(pc_q_.front());
      pc_q_.pop_front();
    }

    if (!job.cloud_msg) {
      continue;
    }

    // Wait here, before any pointcloud processing begins.
    const dlio::ScanTimeBounds scan_time_bounds =
        dlio::scanTimeBoundsFromPointCloud2(*job.cloud_msg);
    const double scan_start_time = scan_time_bounds.start;
    const double scan_end_time = scan_time_bounds.end;

    {
      std::unique_lock<decltype(this->mtx_imu)> imu_lock(this->mtx_imu);
      // Gap 2 fix: also wake on reset_in_progress_ so clearing imu_buffer
      // during reset doesn't leave this wait stuck forever.
      this->cv_imu_stamp.wait(imu_lock, [this, scan_end_time]{
        return this->shouldStop()
            || this->reset_in_progress_.load()
            || (!this->imu_buffer.empty() &&
                this->imu_buffer.front().stamp >= scan_end_time);
      });
    }

    if (this->shouldStop()) {
      break;
    }

    // Reset arrived while we were waiting for IMU: drop the dequeued scan and reset.
    if (this->reset_in_progress_.load()) {
      RCLCPP_INFO(this->get_logger(),
                  "\033[33m[RESET] Worker: reset detected after IMU wait — dropping current scan.\033[0m");
      this->performReset();
      continue;
    }

    if (!this->first_valid_scan.load() &&
        !this->imuBufferCoversRange(scan_start_time, scan_end_time)) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "\033[38;5;214m[STARTUP] Dropping first post-calibration scan without full IMU coverage "
          "(scan_start=%.6f, scan_end=%.6f).\033[0m",
          scan_start_time, scan_end_time);
      continue;
    }

    this->processPointCloud(job.cloud_msg);
  }
}

void dlio::OdomNode::loadRunStatsParams() {
  dlio::declare_param(this, "run_stats/enabled", this->run_stats_enabled_, false);
  dlio::declare_param(this, "run_stats/output_dir", this->run_stats_output_dir_,
                      std::string("/tmp/dlio_run_stats"));
  dlio::declare_param(this, "run_stats/overwrite", this->run_stats_overwrite_, true);
  dlio::declare_param(this, "run_stats/plot_on_shutdown",
                      this->run_stats_plot_on_shutdown_, true);
  dlio::declare_param(this, "run_stats/plot_dpi", this->run_stats_plot_dpi_, 600);
  dlio::declare_param(this, "run_stats/plot_script", this->run_stats_plot_script_,
                      std::string(""));

  if (this->run_stats_plot_dpi_ < 72) {
    RCLCPP_WARN(this->get_logger(),
                "run_stats/plot_dpi must be >= 72. Clamping to 72.");
    this->run_stats_plot_dpi_ = 72;
  }
}

void dlio::OdomNode::loadDynamicFilterCommonParams() {
  dlio::declare_param(this, "dynamic_filter/enabled",
                      this->m_detector_filter_config_.enabled, false);
  dlio::declare_param(this, "dynamic_filter/force_removed_cloud_output",
                      this->dynamic_filter_force_removed_cloud_output_, false);
  dlio::declare_param(this, "dynamic_filter/voxel_size",
                      this->m_detector_filter_config_.voxel_size, 0.20);
  dlio::declare_param(this, "dynamic_filter/min_range",
                      this->m_detector_filter_config_.min_range, 1.0);
  dlio::declare_param(this, "dynamic_filter/max_range",
                      this->m_detector_filter_config_.max_range, 10.0);
  dlio::declare_param(this, "dynamic_filter/warmup_scans",
                      this->m_detector_filter_config_.warmup_scans, 5);
  dlio::declare_param(this, "dynamic_filter/static_score_threshold",
                      this->m_detector_filter_config_.static_score_threshold, 2);
  dlio::declare_param(this, "dynamic_filter/static_window_scans",
                      this->m_detector_filter_config_.static_window_scans, 5);
  dlio::declare_param(this, "dynamic_filter/max_age_scans",
                      this->m_detector_filter_config_.max_age_scans, 300);
  dlio::declare_param(this, "dynamic_filter/local_radius",
                      this->m_detector_filter_config_.local_radius, 30.0);
}

void dlio::OdomNode::loadMDetectorFilterParams() {
  dlio::declare_param(this, "dynamic_filter/m_detector/projection/rows",
                      this->m_detector_filter_config_.projection_rows, 128);
  dlio::declare_param(this, "dynamic_filter/m_detector/projection/cols",
                      this->m_detector_filter_config_.projection_cols, 900);
  dlio::declare_param(this, "dynamic_filter/m_detector/projection/use_ring_field",
                      this->m_detector_filter_config_.projection_use_ring_field, true);
  dlio::declare_param(this, "dynamic_filter/m_detector/projection/use_point_timestamp",
                      this->m_detector_filter_config_.projection_use_point_timestamp, true);
  dlio::declare_param(this, "dynamic_filter/m_detector/history_duration",
                      this->m_detector_filter_config_.history_duration, 0.5);
  dlio::declare_param(this, "dynamic_filter/m_detector/max_history_frames",
                      this->m_detector_filter_config_.max_history_frames, 5);
  dlio::declare_param(this, "dynamic_filter/m_detector/frame_duration",
                      this->m_detector_filter_config_.frame_duration, 0.1);
  dlio::declare_param(this, "dynamic_filter/m_detector/min_history_votes",
                      this->m_detector_filter_config_.min_history_votes, 3);
  dlio::declare_param(this, "dynamic_filter/m_detector/case_depth_margin",
                      this->m_detector_filter_config_.case_depth_margin, 0.15);
  dlio::declare_param(this, "dynamic_filter/m_detector/map_consistency_depth",
                      this->m_detector_filter_config_.map_consistency_depth, 0.25);
  dlio::declare_param(this, "dynamic_filter/m_detector/min_cluster_points",
                      this->m_detector_filter_config_.min_cluster_points, 60);
  dlio::declare_param(this, "dynamic_filter/m_detector/min_track_cluster_points",
                      this->m_detector_filter_config_.min_track_cluster_points, 120);
  dlio::declare_param(this, "dynamic_filter/m_detector/max_cluster_extent",
                      this->m_detector_filter_config_.max_cluster_extent, 3.0);
  dlio::declare_param(this, "dynamic_filter/m_detector/max_assoc_distance",
                      this->m_detector_filter_config_.max_assoc_distance, 0.9);
  dlio::declare_param(this, "dynamic_filter/m_detector/track_confirm_hits",
                      this->m_detector_filter_config_.track_confirm_hits, 2);
  dlio::declare_param(this, "dynamic_filter/m_detector/track_ttl_scans",
                      this->m_detector_filter_config_.track_ttl_scans, 20);
  dlio::declare_param(this, "dynamic_filter/m_detector/static_veto_ratio",
                      this->m_detector_filter_config_.static_veto_ratio, 0.25);
}

void dlio::OdomNode::getParams() {

  // Version
  dlio::declare_param(this, "version", this->version_, "0.0.0");

  // Frames
  dlio::declare_param(this, "frames/odom", this->odom_frame, "dlio_odom");
  dlio::declare_param(this, "frames/baselink", this->baselink_frame, "base_link");
  dlio::declare_param(this, "frames/lidar", this->lidar_frame, "lidar");
  dlio::declare_param(this, "frames/imu", this->imu_frame, "imu");

  // Deskew Flag
  dlio::declare_param(this, "pointcloud/deskew", this->deskew_, true);
  dlio::declare_param(this, "pointcloud/queueSize", this->pointcloud_queue_size_, 5);
  if (this->pointcloud_queue_size_ < 1) {
    RCLCPP_WARN(this->get_logger(), "pointcloud/queueSize must be >= 1. Falling back to 1.");
    this->pointcloud_queue_size_ = 1;
  }

  // Gravity
  dlio::declare_param(this, "odom/gravity", this->gravity_, 9.80665);

  // Compute time offset between lidar and imu
  dlio::declare_param(this, "odom/computeTimeOffset", this->time_offset_, false);

  // Keep OpenMP work bounded for live-robot jitter. This controls GICP and the
  // existing deskew loop; raise only after measuring CPU headroom on the robot.
  dlio::declare_param(this, "odom/num_threads", this->num_threads_, 4);
#ifdef _OPENMP
  const int max_openmp_threads = std::max(1, omp_get_max_threads());
#else
  const int max_openmp_threads = 1;
#endif
  if (this->num_threads_ < 1) {
    RCLCPP_WARN(this->get_logger(), "odom/num_threads must be >= 1. Clamping to 1.");
    this->num_threads_ = 1;
  }
  if (this->num_threads_ > max_openmp_threads) {
    RCLCPP_WARN(this->get_logger(),
                "odom/num_threads=%d exceeds available OpenMP threads=%d. Clamping.",
                this->num_threads_, max_openmp_threads);
    this->num_threads_ = max_openmp_threads;
  }

  // Keyframe Threshold
  dlio::declare_param(this, "odom/keyframe/threshD", this->keyframe_thresh_dist_, 0.1);
  dlio::declare_param(this, "odom/keyframe/threshR", this->keyframe_thresh_rot_, 1.0);

  // Submap
  dlio::declare_param(this, "odom/submap/keyframe/knn", this->submap_knn_, 10);
  dlio::declare_param(this, "odom/submap/keyframe/kcv", this->submap_kcv_, 10);
  dlio::declare_param(this, "odom/submap/keyframe/kcc", this->submap_kcc_, 10);

  // Dense map resolution
  dlio::declare_param(this, "map/dense/filtered", this->densemap_filtered_, true);

  // Wait until movement to publish map
  dlio::declare_param(this, "map/waitUntilMove", this->wait_until_move_, false);

  // Crop Box Filter
  dlio::declare_param(this, "odom/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Voxel Grid Filter
  dlio::declare_param(this, "pointcloud/voxelize", this->vf_use_, true);
  dlio::declare_param(this, "odom/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // Adaptive Parameters
  dlio::declare_param(this, "adaptive", this->adaptive_params_, true);
  dlio::declare_param(this, "adaptive/spaciousness/min",    this->adaptive_sp_min_,          0.5f);
  dlio::declare_param(this, "adaptive/spaciousness/max",    this->adaptive_sp_max_,          5.0f);
  dlio::declare_param(this, "adaptive/density/factor_min",  this->adaptive_den_factor_min_,  0.5f);
  dlio::declare_param(this, "adaptive/density/factor_max",  this->adaptive_den_factor_max_,  2.0f);

  // Extrinsics
  std::vector<double> t_default{0., 0., 0.};
  std::vector<double> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<double> baselink2imu_t, baselink2imu_R;
  dlio::declare_param(this, "extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(static_cast<float>(baselink2imu_t[0]),
                    static_cast<float>(baselink2imu_t[1]),
                    static_cast<float>(baselink2imu_t[2]));
  this->extrinsics.baselink2imu.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2imu_R.begin(), baselink2imu_R.end()).data(), 3, 3);
  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<double> baselink2lidar_t, baselink2lidar_R;
  dlio::declare_param(this, "extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
    Eigen::Vector3f(static_cast<float>(baselink2lidar_t[0]),
                    static_cast<float>(baselink2lidar_t[1]),
                    static_cast<float>(baselink2lidar_t[2]));
  this->extrinsics.baselink2lidar.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2lidar_R.begin(), baselink2lidar_R.end()).data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // IMU
  dlio::declare_param(this, "odom/imu/calibration/accel", this->calibrate_accel_, true);
  dlio::declare_param(this, "odom/imu/calibration/gyro", this->calibrate_gyro_, true);
  dlio::declare_param(this, "odom/imu/calibration/time", this->imu_calib_time_, 3.0);
  dlio::declare_param(this, "odom/imu/bufferSize", this->imu_buffer_size_, 2000);
  dlio::declare_param(this, "odom/imu/unit_check/enabled",
                      this->imu_unit_scale_config_.enabled, true);
  dlio::declare_param(this, "odom/imu/unit_check/auto_scale",
                      this->imu_unit_scale_config_.auto_scale, true);
  dlio::declare_param(this, "odom/imu/unit_check/min_samples",
                      this->imu_unit_scale_config_.min_samples, 50);
  dlio::declare_param(this, "odom/imu/unit_check/max_wait_sec",
                      this->imu_unit_scale_config_.max_wait_sec, 0.5);
  dlio::declare_param(this, "odom/imu/unit_check/warn_period_sec",
                      this->imu_unit_scale_config_.warn_period_sec, 2.0);
  dlio::declare_param(this, "odom/imu/unit_check/assume_deg_per_sec_when_accel_g",
                      this->imu_unit_scale_config_.assume_deg_per_sec_when_accel_g, true);
  dlio::declare_param(this, "odom/imu/unit_check/accel_scale_override",
                      this->imu_unit_scale_config_.accel_scale_override, 0.0);
  dlio::declare_param(this, "odom/imu/unit_check/gyro_scale_override",
                      this->imu_unit_scale_config_.gyro_scale_override, 0.0);
  this->imu_unit_scaler_.configure(this->imu_unit_scale_config_);

  std::vector<double> accel_default{0., 0., 0.}; std::vector<double> prior_accel_bias;
  std::vector<double> gyro_default{0., 0., 0.}; std::vector<double> prior_gyro_bias;

  dlio::declare_param(this, "odom/imu/approximateGravity", this->gravity_align_, true);
  dlio::declare_param(this, "imu/calibration", this->imu_calibrate_, true);
  dlio::declare_param(this, "imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  dlio::declare_param(this, "imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<double> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<double> imu_sm;

  dlio::declare_param(this, "imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel = Eigen::Vector3f(static_cast<float>(prior_accel_bias[0]),
                                          static_cast<float>(prior_accel_bias[1]),
                                          static_cast<float>(prior_accel_bias[2]));
    this->state.b.gyro = Eigen::Vector3f(static_cast<float>(prior_gyro_bias[0]),
                                         static_cast<float>(prior_gyro_bias[1]),
                                         static_cast<float>(prior_gyro_bias[2]));
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(imu_sm.begin(), imu_sm.end()).data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  dlio::declare_param(this, "odom/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  dlio::declare_param(this, "odom/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  dlio::declare_param(this, "odom/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
      std::sqrt(std::numeric_limits<double>::max()));
  dlio::declare_param(this, "odom/gicp/maxIterations", this->gicp_max_iter_, 64);
  dlio::declare_param(this, "odom/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  dlio::declare_param(this, "odom/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  dlio::declare_param(this, "odom/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);

  // Geometric Observer
  // --- Orientation layer (Eq. 3): Kq ≈ 2*c1 (attitude rate), Kgb ≈ c2 (gyro-bias rate) ---
  dlio::declare_param(this, "odom/geo/Kq",              this->geo_Kq_,              6.7);  // ~2/τq, τq≈0.30 s
  dlio::declare_param(this, "odom/geo/Kgb",             this->geo_Kgb_,             2.0);  // ~0.3–1.0 * c1; pick ~0.5*c1

  // --- Translation layer (Eq. 15): Kp≈ω_n^2, Kv≈2ζω_n, Kab≈K1 ---
  dlio::declare_param(this, "odom/geo/Kp",              this->geo_Kp_,              2.25); // ω_n≈1.5 s^-1  => ω_n^2
  dlio::declare_param(this, "odom/geo/Kv",              this->geo_Kv_,              3.0);  // 2 ζ ω_n with ζ≈1
  dlio::declare_param(this, "odom/geo/Kab",             this->geo_Kab_,             0.10); // conservative accel-bias adaption

  // --- Bias anti-windup clamps (pick from your IMU datasheet ranges) ---
  dlio::declare_param(this, "odom/geo/abias_max",       this->geo_abias_max_,       1.5);  // [m/s^2]
  dlio::declare_param(this, "odom/geo/gbias_max",       this->geo_gbias_max_,       0.30); // [rad/s]

  // Visualization (velocity markers)
  dlio::declare_param(this, "odom/debug/enabled",            this->debug_enabled_,        false);

  dlio::declare_param(this, "viz/vel_marker/enabled",        this->viz_vel_markers_,      true);
  dlio::declare_param(this, "viz/vel_marker/scale_lin",      this->viz_lin_gain_,         0.5);   // arrow length gain
  dlio::declare_param(this, "viz/vel_marker/ang/radius_gain",this->viz_ang_radius_gain_,  0.20);
  dlio::declare_param(this, "viz/vel_marker/ang/r_min",      this->viz_ang_radius_min_,   0.10);
  dlio::declare_param(this, "viz/vel_marker/ang/r_max",      this->viz_ang_radius_max_,   0.50);
  dlio::declare_param(this, "viz/vel_marker/thickness",      this->viz_disc_thickness_,   0.03);
  dlio::declare_param(this, "viz/vel_marker/lifetime",       this->viz_marker_lifetime_,  0.10);
  dlio::declare_param(this, "viz/corr_marker/enabled",       this->viz_corr_marker_,      true);
  dlio::declare_param(this, "viz/corr_marker/scale",         this->viz_corr_gain_,        1.0);
  dlio::declare_param(this, "viz/corr_marker/line_width",    this->viz_corr_line_width_,  0.008);
  dlio::declare_param(this, "viz/corr_marker/max_segments",  this->viz_corr_max_segments_,2000);
  dlio::declare_param(this, "viz/corr_marker/lifetime",      this->viz_corr_lifetime_,    0.0);

  // Translation-only degeneracy analysis from the raw scan normal-spread
  // matrix. Weak directions are flagged by absolute thresholds on its
  // eigenvalues.
  dlio::declare_param(this, "odom/gicp/degeneracy/enabled",
                      this->use_degeneracy_, true);
  dlio::declare_param(this, "odom/gicp/degeneracy/trans_eig_abs_threshold",
                      this->degen_trans_eig_abs_thresh_, 200.0);
  dlio::declare_param(this, "odom/gicp/degeneracy/reset_consecutive_count",
                      this->degen_reset_consecutive_count_, 5);

  // Restart geometry gate: after a reset, incoming scans are checked via
  // local-normal diversity before the system re-initializes. Scans that fail
  // (weakest normal-scatter eigenvalue below threshold) are dropped.
  dlio::declare_param(this, "odom/restart/geometry_gate/enabled",
                      this->restart_gate_enabled_, true);
  dlio::declare_param(this, "odom/restart/geometry_gate/min_eigenvalue",
                      this->restart_gate_min_eigenvalue_, 50.0);

  dlio::declare_param(this, "viz/degeneracy_marker/enabled", this->viz_degen_marker_, true);
  dlio::declare_param(this, "viz/degeneracy_marker/trans_scale", this->viz_degen_trans_scale_, 0.75);
  dlio::declare_param(this, "viz/degeneracy_marker/shaft_diameter", this->viz_degen_shaft_diam_, 0.03);
  dlio::declare_param(this, "viz/degeneracy_marker/head_diameter", this->viz_degen_head_diam_, 0.06);
  dlio::declare_param(this, "viz/degeneracy_marker/head_length", this->viz_degen_head_len_, 0.10);
  dlio::declare_param(this, "viz/degeneracy_marker/lifetime", this->viz_degen_lifetime_, 0.0);

  this->loadRunStatsParams();
  this->loadDynamicFilterCommonParams();
  this->loadMDetectorFilterParams();

  if (this->viz_corr_gain_ < 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/corr_marker/scale must be >= 0. Clamping to 0.");
    this->viz_corr_gain_ = 0.0;
  }
  if (this->viz_corr_line_width_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/corr_marker/line_width must be > 0. Clamping to 0.01.");
    this->viz_corr_line_width_ = 0.01;
  }
  if (this->viz_corr_max_segments_ < 1) {
    RCLCPP_WARN(this->get_logger(), "viz/corr_marker/max_segments must be >= 1. Clamping to 1.");
    this->viz_corr_max_segments_ = 1;
  }
  if (this->viz_corr_lifetime_ < 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/corr_marker/lifetime must be >= 0. Clamping to 0.");
    this->viz_corr_lifetime_ = 0.0;
  }

  if (this->degen_trans_eig_abs_thresh_ < 0.0) {
    RCLCPP_WARN(this->get_logger(),
                "odom/gicp/degeneracy/trans_eig_abs_threshold must be >= 0. Clamping to 0.");
    this->degen_trans_eig_abs_thresh_ = 0.0;
  }
  if (this->degen_reset_consecutive_count_ < 0) {
    RCLCPP_WARN(this->get_logger(),
                "odom/gicp/degeneracy/reset_consecutive_count must be >= 0. Clamping to 0.");
    this->degen_reset_consecutive_count_ = 0;
  }
  if (this->viz_degen_trans_scale_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/degeneracy_marker/trans_scale must be > 0. Clamping to 0.75.");
    this->viz_degen_trans_scale_ = 0.75;
  }
  if (this->viz_degen_shaft_diam_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/degeneracy_marker/shaft_diameter must be > 0. Clamping to 0.03.");
    this->viz_degen_shaft_diam_ = 0.03;
  }
  if (this->viz_degen_head_diam_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/degeneracy_marker/head_diameter must be > 0. Clamping to 0.06.");
    this->viz_degen_head_diam_ = 0.06;
  }
  if (this->viz_degen_head_len_ <= 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/degeneracy_marker/head_length must be > 0. Clamping to 0.10.");
    this->viz_degen_head_len_ = 0.10;
  }
  if (this->viz_degen_lifetime_ < 0.0) {
    RCLCPP_WARN(this->get_logger(), "viz/degeneracy_marker/lifetime must be >= 0. Clamping to 0.");
    this->viz_degen_lifetime_ = 0.0;
  }

}

void dlio::OdomNode::initializeRunStats() {
  if (!this->run_stats_enabled_) {
    return;
  }

  namespace fs = std::filesystem;
  if (this->run_stats_output_dir_.empty()) {
    this->run_stats_output_dir_ = "/tmp/dlio_run_stats";
  }

  if (this->run_stats_plot_script_.empty()) {
    try {
      this->run_stats_plot_script_ =
          ament_index_cpp::get_package_share_directory("direct_lidar_inertial_odometry") +
          "/scripts/plot_run_stats.py";
    } catch (const std::exception& ex) {
      RCLCPP_WARN(this->get_logger(),
                  "Could not resolve installed run-stats plotter: %s", ex.what());
      this->run_stats_plot_script_ = "scripts/plot_run_stats.py";
    }
  }

  std::error_code ec;
  fs::create_directories(this->run_stats_output_dir_, ec);
  if (ec) {
    RCLCPP_ERROR(this->get_logger(),
                 "Failed to create run_stats/output_dir '%s': %s",
                 this->run_stats_output_dir_.c_str(), ec.message().c_str());
    this->run_stats_enabled_ = false;
    return;
  }

  if (this->run_stats_overwrite_) {
    const fs::path out_dir(this->run_stats_output_dir_);
    fs::remove(out_dir / "run_stats.csv", ec);
    for (const char* name : runStatsPlotFiles()) {
      fs::remove(out_dir / name, ec);
    }
    fs::remove(out_dir / "run_stats_summary.txt", ec);
  }

  const fs::path csv_path = fs::path(this->run_stats_output_dir_) / "run_stats.csv";
  const std::ios_base::openmode mode =
      std::ios::out | (this->run_stats_overwrite_ ? std::ios::trunc : std::ios::app);

  std::lock_guard<std::mutex> lock(this->run_stats_mtx_);
  this->run_stats_csv_.open(csv_path, mode);
  if (!this->run_stats_csv_.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open run stats CSV '%s'.",
                 csv_path.string().c_str());
    this->run_stats_enabled_ = false;
    return;
  }

  this->run_stats_csv_ << std::setprecision(12);
  if (this->run_stats_overwrite_ || fs::file_size(csv_path, ec) == 0U) {
    this->run_stats_csv_
        << "time_s,stamp_sec,"
        << "p_x_m,p_y_m,p_z_m,"
        << "q_w,q_x,q_y,q_z,"
        << "roll_rad,pitch_rad,yaw_rad,"
        << "vlin_b_x_mps,vlin_b_y_mps,vlin_b_z_mps,"
        << "vang_b_x_radps,vang_b_y_radps,vang_b_z_radps,"
        << "accel_bias_x_mps2,accel_bias_y_mps2,accel_bias_z_mps2,"
        << "gyro_bias_x_radps,gyro_bias_y_radps,gyro_bias_z_radps\n";
  }

  this->run_stats_rows_ = 0U;
  this->run_stats_have_first_stamp_ = false;
  this->run_stats_plot_generated_ = false;

  RCLCPP_INFO(this->get_logger(), "Run-state CSV export enabled: %s",
              csv_path.string().c_str());
}

void dlio::OdomNode::recordRunStats(
    const double stamp_sec,
    const Eigen::Ref<const Eigen::Matrix4f>& T_map_base,
    const Eigen::Vector3f& vlin_b,
    const Eigen::Vector3f& vang_b,
    const Eigen::Vector3f& accel_bias,
    const Eigen::Vector3f& gyro_bias) {
  if (!this->run_stats_enabled_) {
    return;
  }

  Eigen::Quaternionf q_map_base(T_map_base.block<3, 3>(0, 0));
  q_map_base.normalize();
  const Eigen::Vector3f p_map_base = T_map_base.block<3, 1>(0, 3);
  const Eigen::Vector3d rpy = quaternionToRollPitchYaw(q_map_base);

  std::lock_guard<std::mutex> lock(this->run_stats_mtx_);
  if (!this->run_stats_csv_.is_open()) {
    return;
  }

  if (!this->run_stats_have_first_stamp_) {
    this->run_stats_first_stamp_ = stamp_sec;
    this->run_stats_have_first_stamp_ = true;
  }
  const double time_s = stamp_sec - this->run_stats_first_stamp_;

  this->run_stats_csv_
      << time_s << ','
      << stamp_sec << ','
      << p_map_base.x() << ','
      << p_map_base.y() << ','
      << p_map_base.z() << ','
      << q_map_base.w() << ','
      << q_map_base.x() << ','
      << q_map_base.y() << ','
      << q_map_base.z() << ','
      << rpy.x() << ','
      << rpy.y() << ','
      << rpy.z() << ','
      << vlin_b.x() << ','
      << vlin_b.y() << ','
      << vlin_b.z() << ','
      << vang_b.x() << ','
      << vang_b.y() << ','
      << vang_b.z() << ','
      << accel_bias.x() << ','
      << accel_bias.y() << ','
      << accel_bias.z() << ','
      << gyro_bias.x() << ','
      << gyro_bias.y() << ','
      << gyro_bias.z() << '\n';
  ++this->run_stats_rows_;
}

void dlio::OdomNode::closeRunStats() {
  std::lock_guard<std::mutex> lock(this->run_stats_mtx_);
  if (this->run_stats_csv_.is_open()) {
    this->run_stats_csv_.flush();
    this->run_stats_csv_.close();
  }
}

void dlio::OdomNode::generateRunStatsPlots() {
  if (!this->run_stats_enabled_ || !this->run_stats_plot_on_shutdown_ ||
      this->run_stats_plot_generated_) {
    return;
  }

  const std::size_t rows = this->run_stats_rows_;
  this->closeRunStats();

  if (rows == 0U) {
    RCLCPP_WARN(this->get_logger(),
                "run_stats/plot_on_shutdown requested, but no run-state rows were written.");
    return;
  }

  namespace fs = std::filesystem;
  const fs::path csv_path = fs::path(this->run_stats_output_dir_) / "run_stats.csv";
  if (!fs::exists(csv_path)) {
    RCLCPP_WARN(this->get_logger(), "Run-state CSV does not exist: %s",
                csv_path.string().c_str());
    return;
  }

  const fs::path script_path(this->run_stats_plot_script_);
  if (!fs::exists(script_path)) {
    RCLCPP_WARN(this->get_logger(), "Run-state plotter script does not exist: %s",
                script_path.string().c_str());
    return;
  }

  std::ostringstream cmd;
  cmd << "python3 "
      << shellQuote(script_path.string())
      << " --csv " << shellQuote(csv_path.string())
      << " --out-dir " << shellQuote(this->run_stats_output_dir_)
      << " --dpi " << this->run_stats_plot_dpi_;

  const int rc = std::system(cmd.str().c_str());
  if (rc != 0) {
    RCLCPP_WARN(this->get_logger(), "Run-state plotter failed with code %d.", rc);
    return;
  }

  this->run_stats_plot_generated_ = true;
  RCLCPP_INFO(this->get_logger(), "Run-state plots generated in %s",
              this->run_stats_output_dir_.c_str());
}

void dlio::OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout << '\n'
            << "+-------------------------------------------------------------------+\n";
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << '\n';
  std::cout << "+-------------------------------------------------------------------+\n";

}

void dlio::OdomNode::publishToROS(
    const pcl::PointCloud<PointType>::ConstPtr& cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_all,
    const double scanStamp,
    const Eigen::Vector3f& state_p_scan,
    const Eigen::Quaternionf& state_q_scan,
    const Eigen::Vector3f& state_vlin_b_scan,
    const Eigen::Vector3f& state_vang_b_scan)
{
  // Build an exact scan-time timestamp once and use it everywhere below.
  const uint64_t nsec = static_cast<uint64_t>(scanStamp * 1e9);
  builtin_interfaces::msg::Time scan_stamp_msg;
  scan_stamp_msg.sec = static_cast<int32_t>(nsec / 1000000000ULL);
  scan_stamp_msg.nanosec = static_cast<uint32_t>(nsec % 1000000000ULL);
  const rclcpp::Time scan_time(scan_stamp_msg);

  // ---------------------------------------------------------------------------
  // dlio_map <-> base_link at scan time
  //
  // T_all is the scan-time transform dlio_map -> base_link.
  // Your TF convention in this node publishes the inverse direction:
  //   parent = base_link, child = dlio_map
  // ---------------------------------------------------------------------------
  const Eigen::Vector3f p_mb = T_all.block<3,1>(0,3);
  Eigen::Quaternionf q_mb(T_all.block<3,3>(0,0));
  q_mb.normalize();

  const Eigen::Quaternionf q_bm = q_mb.conjugate();
  const Eigen::Vector3f p_bm = -(q_bm._transformVector(p_mb));

  // map_pose: pose of dlio_map in base_link, time-aligned to the scan.
  if (hasSubscribers(this->odom_map_pub)) {
    nav_msgs::msg::Odometry odom_map;
    odom_map.header.stamp = scan_stamp_msg;
    odom_map.header.frame_id = this->baselink_frame;
    odom_map.child_frame_id = "dlio_map";

    odom_map.pose.pose.position.x = p_bm.x();
    odom_map.pose.pose.position.y = p_bm.y();
    odom_map.pose.pose.position.z = p_bm.z();
    odom_map.pose.pose.orientation.w = q_bm.w();
    odom_map.pose.pose.orientation.x = q_bm.x();
    odom_map.pose.pose.orientation.y = q_bm.y();
    odom_map.pose.pose.orientation.z = q_bm.z();

    // Twist of dlio_map w.r.t. base_link, expressed in child frame (dlio_map).
    // Keep this derived from the same scan-time odom snapshot used below so the
    // published pose/twist/cloud/TF are self-consistent.
    const Eigen::Vector3f v_mb_m = q_mb._transformVector(state_vlin_b_scan);
    const Eigen::Vector3f w_mb_m = q_mb._transformVector(state_vang_b_scan);
    const Eigen::Vector3f v_bm_m = -(v_mb_m + p_mb.cross(w_mb_m));
    const Eigen::Vector3f w_bm_m = -w_mb_m;

    odom_map.twist.twist.linear.x  = v_bm_m.x();
    odom_map.twist.twist.linear.y  = v_bm_m.y();
    odom_map.twist.twist.linear.z  = v_bm_m.z();
    odom_map.twist.twist.angular.x = w_bm_m.x();
    odom_map.twist.twist.angular.y = w_bm_m.y();
    odom_map.twist.twist.angular.z = w_bm_m.z();

    this->odom_map_pub->publish(odom_map);
  }

  // base_pose: pose of base_link in dlio_map, time-aligned to the scan.
  if (hasSubscribers(this->odom_baselink_pub)) {
    nav_msgs::msg::Odometry odom_baselink;
    odom_baselink.header.stamp    = scan_stamp_msg;
    odom_baselink.header.frame_id = "dlio_map";
    odom_baselink.child_frame_id  = this->baselink_frame;

    odom_baselink.pose.pose.position.x    = p_mb.x();
    odom_baselink.pose.pose.position.y    = p_mb.y();
    odom_baselink.pose.pose.position.z    = p_mb.z();
    odom_baselink.pose.pose.orientation.w = q_mb.w();
    odom_baselink.pose.pose.orientation.x = q_mb.x();
    odom_baselink.pose.pose.orientation.y = q_mb.y();
    odom_baselink.pose.pose.orientation.z = q_mb.z();

    // Twist of base_link w.r.t. dlio_map, expressed in child frame (base_link).
    odom_baselink.twist.twist.linear.x  = state_vlin_b_scan.x();
    odom_baselink.twist.twist.linear.y  = state_vlin_b_scan.y();
    odom_baselink.twist.twist.linear.z  = state_vlin_b_scan.z();
    odom_baselink.twist.twist.angular.x = state_vang_b_scan.x();
    odom_baselink.twist.twist.angular.y = state_vang_b_scan.y();
    odom_baselink.twist.twist.angular.z = state_vang_b_scan.z();

    this->odom_baselink_pub->publish(odom_baselink);
  }

  // ---------------------------------------------------------------------------
  // Scan-time path in dlio_map
  // ---------------------------------------------------------------------------
  this->path_ros.header.stamp = scan_stamp_msg;
  this->path_ros.header.frame_id = "dlio_map";

  geometry_msgs::msg::PoseStamped path_pose;
  path_pose.header.stamp = scan_stamp_msg;
  path_pose.header.frame_id = "dlio_map";
  path_pose.pose.position.x = p_mb.x();
  path_pose.pose.position.y = p_mb.y();
  path_pose.pose.position.z = p_mb.z();
  path_pose.pose.orientation.w = q_mb.w();
  path_pose.pose.orientation.x = q_mb.x();
  path_pose.pose.orientation.y = q_mb.y();
  path_pose.pose.orientation.z = q_mb.z();

  constexpr size_t kMaxPath = 1500;
  if (this->path_poses_.size() >= kMaxPath) {
    this->path_poses_.pop_front();
  }
  this->path_poses_.push_back(std::move(path_pose));
  if (hasSubscribers(this->path_pub)) {
    this->path_ros.poses.assign(this->path_poses_.begin(), this->path_poses_.end());
    this->path_pub->publish(this->path_ros);
  }

  if (this->viz_corr_marker_ && this->pub_corr_marker_ &&
      hasSubscribers(this->pub_corr_marker_)) {
    this->publishCorrectionMarker(scan_time, T_cloud, T_all);
  }

  // ---------------------------------------------------------------------------
  // dlio_odom <-> base_link at the SAME scan-time corrected state snapshot
  //
  // state_q_scan/state_p_scan represent dlio_odom -> base_link at scan time.
  // The cloud published in dlio_odom must use this exact same snapshot.
  // ---------------------------------------------------------------------------
  const Eigen::Quaternionf q_ob = state_q_scan.normalized();
  const Eigen::Quaternionf q_bo = q_ob.conjugate();
  const Eigen::Vector3f p_bo = -(q_bo._transformVector(state_p_scan));

  Eigen::Matrix4f T_bl_odom = Eigen::Matrix4f::Identity();
  T_bl_odom.block<3,3>(0,0) = q_bo.toRotationMatrix();
  T_bl_odom.block<3,1>(0,3) = p_bo;

  // map -> odom at the exact scan-time corrected snapshot.
  const Eigen::Matrix4f T_map_odom = T_all * T_bl_odom;

  // Keep the latest map->odom for IMU-rate propagated map visualization.
  {
    std::lock_guard<std::mutex> map_odom_lock(this->mtx_T_map_odom_latest);
    this->T_map_odom_latest = T_map_odom;
    this->has_T_map_odom_latest = true;
  }

  // ---------------------------------------------------------------------------
  // TFs at scan time
  //
  // Publish only base_link->dlio_map here.
  // base_link->dlio_odom is published exclusively in publishPoseSnapshot().
  // ---------------------------------------------------------------------------
  std::vector<geometry_msgs::msg::TransformStamped> tfs;
  tfs.reserve(1);

  geometry_msgs::msg::TransformStamped tf_bl_map;
  tf_bl_map.header.stamp = scan_stamp_msg;
  tf_bl_map.header.frame_id = this->baselink_frame;
  tf_bl_map.child_frame_id = "dlio_map";
  tf_bl_map.transform.translation.x = p_bm.x();
  tf_bl_map.transform.translation.y = p_bm.y();
  tf_bl_map.transform.translation.z = p_bm.z();
  tf_bl_map.transform.rotation.w = q_bm.w();
  tf_bl_map.transform.rotation.x = q_bm.x();
  tf_bl_map.transform.rotation.y = q_bm.y();
  tf_bl_map.transform.rotation.z = q_bm.z();
  tfs.emplace_back(std::move(tf_bl_map));

  this->br->sendTransform(tfs);

  // Publish clouds using the same scan-time map<->odom relation and the exact
  // same timestamp used for TF lookup.
  this->publishCloud(cloud, T_cloud, T_all, T_map_odom, scan_time);
}

static inline void prepare_xyz_msg(sensor_msgs::msg::PointCloud2& msg,
                                   const std::string& frame_id,
                                   const rclcpp::Time& stamp,
                                   size_t n) {
  msg.header.frame_id = frame_id;

  // precise conversion rclcpp::Time -> builtin_interfaces::msg::Time
  const int64_t nsec = stamp.nanoseconds();
  msg.header.stamp.sec     = static_cast<int32_t>(nsec / 1000000000LL);
  msg.header.stamp.nanosec = static_cast<uint32_t>(nsec % 1000000000LL);

  msg.height = 1;
  msg.width  = static_cast<uint32_t>(n);
  msg.is_bigendian = false;
  msg.is_dense = true;

  sensor_msgs::PointCloud2Modifier mod(msg);
  mod.setPointCloud2FieldsByString(1, "xyz");
  mod.resize(n);
}

void dlio::OdomNode::publishCloud(
    const pcl::PointCloud<PointType>::ConstPtr& cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
    const Eigen::Ref<const Eigen::Matrix4f>& T_all,
    const Eigen::Ref<const Eigen::Matrix4f>& T_map_odom,
    const rclcpp::Time& cloud_stamp)
{
  if (this->wait_until_move_ && this->length_traversed < 0.1) {
    return;
  }

  if (!cloud) {
    return;
  }

  const size_t n = cloud->size();
  if (n == 0) {
    return;
  }

  // Cloud input is already in dlio_map after deskew / registration pipeline.
  // We publish three views of the same scan:
  //   1) deskewed                 : in dlio_odom
  //   2) deskewed_not_transformed : in base_link
  //   3) deskewed_and_transformed_to_map : in dlio_map
  // Each view is only computed and published when its topic has a subscriber.
  const bool want_odom = hasSubscribers(this->deskewed_pub);
  const bool want_base = hasSubscribers(this->deskewed_not_transformed_pub);
  const bool want_map  = hasSubscribers(this->deskewed_map_pub);

  if (!want_odom && !want_base && !want_map) {
    return;
  }

  if (want_odom) {
    const Eigen::Matrix4f T_odom_map = T_map_odom.inverse();
    sensor_msgs::msg::PointCloud2 msg;
    prepare_xyz_msg(msg, this->odom_frame, cloud_stamp, n);
    sensor_msgs::PointCloud2Iterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
    bool all_finite = true;
    for (size_t i = 0; i < n; ++i, ++x, ++y, ++z) {
      const auto& p = (*cloud)[i];
      const Eigen::Vector4f v_odom = T_odom_map * (T_cloud * Eigen::Vector4f(p.x, p.y, p.z, 1.f));
      *x = v_odom.x(); *y = v_odom.y(); *z = v_odom.z();
      all_finite = all_finite && std::isfinite(v_odom.x()) && std::isfinite(v_odom.y()) && std::isfinite(v_odom.z());
    }
    msg.is_dense = all_finite;
    this->deskewed_pub->publish(msg);
  }

  if (want_base) {
    const Eigen::Matrix4f T_revert = T_all.inverse() * T_cloud;
    sensor_msgs::msg::PointCloud2 msg;
    prepare_xyz_msg(msg, this->baselink_frame, cloud_stamp, n);
    sensor_msgs::PointCloud2Iterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
    bool all_finite = true;
    for (size_t i = 0; i < n; ++i, ++x, ++y, ++z) {
      const auto& p = (*cloud)[i];
      const Eigen::Vector4f v_base = T_revert * Eigen::Vector4f(p.x, p.y, p.z, 1.f);
      *x = v_base.x(); *y = v_base.y(); *z = v_base.z();
      all_finite = all_finite && std::isfinite(v_base.x()) && std::isfinite(v_base.y()) && std::isfinite(v_base.z());
    }
    msg.is_dense = all_finite;
    this->deskewed_not_transformed_pub->publish(msg);
  }

  if (want_map) {
    sensor_msgs::msg::PointCloud2 msg;
    prepare_xyz_msg(msg, "dlio_map", cloud_stamp, n);
    sensor_msgs::PointCloud2Iterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
    bool all_finite = true;
    for (size_t i = 0; i < n; ++i, ++x, ++y, ++z) {
      const auto& p = (*cloud)[i];
      const Eigen::Vector4f v_map = T_cloud * Eigen::Vector4f(p.x, p.y, p.z, 1.f);
      *x = v_map.x(); *y = v_map.y(); *z = v_map.z();
      all_finite = all_finite && std::isfinite(v_map.x()) && std::isfinite(v_map.y()) && std::isfinite(v_map.z());
    }
    msg.is_dense = all_finite;
    this->deskewed_map_pub->publish(msg);
  }
}

void dlio::OdomNode::publishKeyframe(const KeyframeData& kf) {

  // Push back
  geometry_msgs::msg::Pose p;
  p.position.x = kf.position[0];
  p.position.y = kf.position[1];
  p.position.z = kf.position[2];
  p.orientation.w = kf.orientation.w();
  p.orientation.x = kf.orientation.x();
  p.orientation.y = kf.orientation.y();
  p.orientation.z = kf.orientation.z();
  this->kf_pose_ros.poses.push_back(p);

  // Trim PoseArray to avoid unbounded RViz payload
  if (this->kf_pose_ros.poses.size() > 30) {
    const auto trim_count =
        static_cast<decltype(this->kf_pose_ros.poses)::difference_type>(
            this->kf_pose_ros.poses.size() - 30U);
    this->kf_pose_ros.poses.erase(
      this->kf_pose_ros.poses.begin(),
      this->kf_pose_ros.poses.begin() + trim_count
    );
  }

  // Keyframes are stored/published after being transformed into the map frame.
  this->kf_pose_ros.header.stamp = kf.timestamp;
  this->kf_pose_ros.header.frame_id = "dlio_map";
  if (hasSubscribers(this->kf_pose_pub)) {
    this->kf_pose_pub->publish(this->kf_pose_ros);
  }

  if (hasSubscribers(this->kf_cloud_pub) && kf.mapping_cloud) {
    auto publish_kf_cloud = [&]() {
      sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*kf.mapping_cloud, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = kf.timestamp;
      keyframe_cloud_ros.header.frame_id = "dlio_map";
      this->kf_cloud_pub->publish(keyframe_cloud_ros);
    };
    if (this->vf_use_) {
      if (kf.mapping_cloud->points.size() ==
          static_cast<decltype(kf.mapping_cloud->points.size())>(kf.mapping_cloud->width) *
              static_cast<decltype(kf.mapping_cloud->points.size())>(kf.mapping_cloud->height)) {
        publish_kf_cloud();
      }
    } else {
      publish_kf_cloud();
    }
  }
}

void dlio::OdomNode::getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  pcl::PointCloud<PointType>::Ptr original_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*pc, *original_scan_);

  bool has_ring_field = false;
  bool has_timestamp_field = false;
  for (const auto& field : pc->fields) {
    has_ring_field = has_ring_field || field.name == "ring";
    has_timestamp_field = has_timestamp_field || field.name == "timestamp";
  }

  const std::size_t jt128_rows = 128U;
  bool native_jt128_order =
      has_ring_field &&
      !original_scan_->empty() &&
      original_scan_->size() % jt128_rows == 0U;
  if (native_jt128_order) {
    for (std::size_t i = 0; i < original_scan_->size(); ++i) {
      if (original_scan_->points[i].ring != static_cast<std::uint16_t>(i % jt128_rows)) {
        native_jt128_order = false;
        break;
      }
    }
  }

  for (std::size_t i = 0; i < original_scan_->size(); ++i) {
    PointType& pt = original_scan_->points[i];
    pt.raw_index = static_cast<std::uint32_t>(std::min<std::size_t>(
        i, static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
    if (native_jt128_order) {
      pt.native_col = static_cast<std::uint16_t>(i / jt128_rows);
      pt.has_native_cell = 1U;
    } else {
      pt.native_col = 0U;
      pt.has_native_cell = 0U;
    }
    if (!has_timestamp_field) {
      pt.timestamp = 0.0;
    }
  }

  // Remove NaNs
  // std::vector<int> idx;
  // original_scan_->is_dense = false;
  // pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // Crop Box Filter
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;

  // automatically detect sensor type
  if (this->sensor == dlio::SensorType::UNKNOWN) {
    for (auto &field : pc->fields) {
      if (field.name == "t") {
        this->sensor = dlio::SensorType::OUSTER;
        break;
      } else if (field.name == "time") {
        this->sensor = dlio::SensorType::VELODYNE;
        break;
      } else if (field.name == "timestamp" && !original_scan_->points.empty() && original_scan_->points[0].timestamp < 1e14) {
        this->sensor = dlio::SensorType::HESAI;
        // ROBOSENSE IS ALSO HERE
        break;
      } else if (field.name == "timestamp" && !original_scan_->points.empty() && original_scan_->points[0].timestamp > 1e14) {
        this->sensor = dlio::SensorType::LIVOX;
        break;
      }
    }
  }

  if (this->sensor == dlio::SensorType::UNKNOWN) {
    this->deskew_ = false;
  }

}

void dlio::OdomNode::preprocessPoints() {

  if (!this->original_scan || this->original_scan->empty()) {
    this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
    this->current_scan = this->deskewed_scan;
    this->deskew_status = false;
    this->deskew_size = 0;
    return;
  }

  // Deskew the original dlio-type scan
  if (this->deskew_) {

    this->deskewPointcloud();

    if (!this->first_valid_scan) {
      return;
    }

  } else {

    this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();

    // don't process scans until IMU data is present
    if (!this->first_valid_scan) {
      bool imu_ready = false;
{
  std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);
  imu_ready = !this->imu_buffer.empty() && this->imu_buffer.front().stamp >= this->scan_stamp;
}

      if (!imu_ready) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan

    } else {

      // IMU prior for second scan onwards
      std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
      frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                this->geo.prev_vel.cast<float>(), {this->scan_stamp});

      if (frames.size() > 0) {
        this->T_prior = frames.back();
      } else {
        this->T_prior = this->T;
      }

    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*this->original_scan, *deskewed_scan_,
                              this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_) {
    pcl::PointCloud<PointType>::Ptr current_scan_ = std::make_shared<pcl::PointCloud<PointType>>(*this->deskewed_scan);
    std::unordered_map<MetadataVoxelKey, PointType, MetadataVoxelKeyHash> voxel_metadata;
    if (this->deskewed_scan && !this->deskewed_scan->empty()) {
      voxel_metadata.reserve(this->deskewed_scan->size());
      for (const PointType& source : this->deskewed_scan->points) {
        voxel_metadata.emplace(metadataKeyForPoint(source, this->vf_res_), source);
      }
    }
    this->voxel.setInputCloud(current_scan_);
    this->voxel.filter(*current_scan_);
    if (!voxel_metadata.empty() && !current_scan_->empty()) {
      for (PointType& point : current_scan_->points) {
        const auto it = voxel_metadata.find(metadataKeyForPoint(point, this->vf_res_));
        if (it == voxel_metadata.end()) {
          continue;
        }
        const PointType& source = it->second;
        point.ring = source.ring;
        point.timestamp = source.timestamp;
        point.raw_index = source.raw_index;
        point.native_col = source.native_col;
        point.has_native_cell = source.has_native_cell;
      }
    }
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

}

void dlio::OdomNode::applyDynamicFilterBeforeRegistration() {
  this->registration_scan = this->current_scan;

  if (!this->current_scan || this->current_scan->empty()) {
    this->m_detector_filter_stats_ = MDetectorFilter::Stats{};
    return;
  }

  Eigen::Matrix4f T_map_lidar_prior = this->T_prior * this->extrinsics.baselink2lidar_T;
  Eigen::Vector3f sensor_origin_map = T_map_lidar_prior.block<3, 1>(0, 3);

  const auto result = this->m_detector_filter_.filterRegistration(
      this->current_scan,
      sensor_origin_map,
      this->gicp_min_num_points_);
  this->registration_scan = result.cloud;
  this->m_detector_filter_stats_ = result.stats;
}

void dlio::OdomNode::updateDynamicFilterAfterCorrection() {
  this->keyframe_mapping_scan = this->current_scan;

  const bool dynamic_filter_active = this->m_detector_filter_.enabled();
  const pcl::PointCloud<PointType>::ConstPtr dynamic_update_scan =
      dynamic_filter_active && this->deskewed_scan && !this->deskewed_scan->empty()
          ? this->deskewed_scan
          : this->current_scan;

  if (!dynamic_update_scan || dynamic_update_scan->empty()) {
    this->dynamic_removed_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
    return;
  }

  Eigen::Matrix4f T_map_lidar = this->T * this->extrinsics.baselink2lidar_T;
  Eigen::Vector3f sensor_origin_map = T_map_lidar.block<3, 1>(0, 3);

  const std::size_t registration_kept = this->m_detector_filter_stats_.registration_kept;
  const std::size_t registration_removed = this->m_detector_filter_stats_.registration_removed;
  const bool registration_fallback = this->m_detector_filter_stats_.registration_fallback;
  const bool emit_removed_cloud =
      this->dynamic_filter_force_removed_cloud_output_ ||
      hasSubscribers(this->dynamic_removed_pub_);

  auto result = this->m_detector_filter_.update(
      dynamic_update_scan,
      this->T_corr,
      sensor_origin_map,
      T_map_lidar,
      rclcpp::Time(this->scan_header_stamp).seconds(),
      emit_removed_cloud);

  if (this->m_detector_filter_.enabled() && this->vf_use_ &&
      result.keyframe_cloud && !result.keyframe_cloud->empty()) {
    auto filtered_mapping = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::VoxelGrid<PointType> mapping_voxel;
    const float vf_res = static_cast<float>(this->vf_res_);
    mapping_voxel.setLeafSize(vf_res, vf_res, vf_res);
    mapping_voxel.setInputCloud(result.keyframe_cloud);
    mapping_voxel.filter(*filtered_mapping);
    this->keyframe_mapping_scan = filtered_mapping;
  } else {
    this->keyframe_mapping_scan = result.keyframe_cloud;
  }
  this->dynamic_removed_cloud_ = result.dynamic_points_map;

  this->m_detector_filter_stats_ = result.stats;
  this->m_detector_filter_stats_.registration_kept = registration_kept;
  this->m_detector_filter_stats_.registration_removed = registration_removed;
  this->m_detector_filter_stats_.registration_fallback = registration_fallback;

  if (this->m_detector_filter_.enabled()) {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "[MDET] input=%zu reg_keep=%zu reg_removed=%zu map_keep=%zu dynamic=%zu case1=%zu case2=%zu case3=%zu seeds=%zu clusters=%zu cluster_pts=%zu track_rm=%zu stopped_rm=%zu track_cluster_rejects=%zu static_veto=%zu edge_rejects=%zu ground_rejects=%zu tracks=%zu/%zu tentative=%zu voxels=%zu warmup=%s fallback=%s projection=%s proj_fallback=%s/%s",
        this->m_detector_filter_stats_.input_points,
        this->m_detector_filter_stats_.registration_kept,
        this->m_detector_filter_stats_.registration_removed,
        this->m_detector_filter_stats_.mapping_kept,
        this->m_detector_filter_stats_.dynamic_removed,
        this->m_detector_filter_stats_.case1_points,
        this->m_detector_filter_stats_.case2_points,
        this->m_detector_filter_stats_.case3_points,
        this->m_detector_filter_stats_.seed_points,
        this->m_detector_filter_stats_.cluster_count,
        this->m_detector_filter_stats_.cluster_points,
        this->m_detector_filter_stats_.track_removed_points,
        this->m_detector_filter_stats_.stopped_suppressed_points,
        this->m_detector_filter_stats_.track_cluster_reject_count,
        this->m_detector_filter_stats_.static_veto_count,
        this->m_detector_filter_stats_.edge_reject_count,
        this->m_detector_filter_stats_.ground_reject_count,
        this->m_detector_filter_stats_.confirmed_track_count,
        this->m_detector_filter_stats_.track_count,
        this->m_detector_filter_stats_.tentative_track_count,
        this->m_detector_filter_stats_.voxel_count,
        this->m_detector_filter_stats_.warmup ? "true" : "false",
        this->m_detector_filter_stats_.registration_fallback ? "true" : "false",
        this->m_detector_filter_stats_.projection_native ? "jt128_ring_yaw" : "fallback",
        this->m_detector_filter_stats_.projection_ring_fallback ? "ring" : "none",
        this->m_detector_filter_stats_.projection_timestamp_fallback ? "timestamp" : "none");
  }

  this->publishDynamicRemovedCloud();
}

void dlio::OdomNode::publishDynamicRemovedCloud() {
  auto publish_cloud = [this](const pcl::PointCloud<PointType>::ConstPtr& cloud,
                              const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub) {
    if (!pub || !hasSubscribers(pub) || !cloud || cloud->empty()) {
      return;
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.stamp = this->scan_header_stamp;
    msg.header.frame_id = "dlio_map";
    pub->publish(msg);
  };

  // The map node accumulates this topic to save dynamic_points.pcd. Keep it
  // alive even when visual/debug topics are disabled.
  publish_cloud(this->dynamic_removed_cloud_, this->dynamic_removed_pub_);
}

void dlio::OdomNode::deskewPointcloud() {

  if (!this->original_scan || this->original_scan->empty()) {
    this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
    this->deskew_status = false;
    this->deskew_size = 0;
    return;
  }

  auto deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
  deskewed_scan_->points.resize(this->original_scan->points.size());
  deskewed_scan_->width  = static_cast<uint32_t>(deskewed_scan_->points.size());
  deskewed_scan_->height = 1;

  // individual point timestamps should be relative to this time
  double sweep_ref_time = rclcpp::Time(this->scan_header_stamp).seconds();

  // sort points by timestamp and build list of timestamps
  std::function<bool(const PointType&, const PointType&)> point_time_cmp;
  std::function<bool(boost::range::index_value<PointType&, long>,
                     boost::range::index_value<PointType&, long>)> point_time_neq;
  std::function<double(boost::range::index_value<PointType&, long>)> extract_point_time;

  if (this->sensor == dlio::SensorType::OUSTER) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.t < p2.t; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().t != p2.value().t; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + static_cast<double>(pt.value().t) * 1e-9; };

  } else if (this->sensor == dlio::SensorType::VELODYNE) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.time < p2.time; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().time != p2.value().time; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().time; };

  } else if (this->sensor == dlio::SensorType::HESAI) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp; };

  } else if (this->sensor == dlio::SensorType::LIVOX) {
    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [](boost::range::index_value<PointType&, long> pt)
      { return static_cast<double>(pt.value().timestamp) * 1e-9; };
  
    } else {
    this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>(*this->original_scan);
    this->deskew_status = false;
    this->deskew_size = 0;
    return;
  }

  // Copy points into deskewed_scan_ in timestamp order while preserving the
  // native JT-128 channel order for points with identical timestamps.
  deskewed_scan_->points = this->original_scan->points;
  std::stable_sort(deskewed_scan_->points.begin(), deskewed_scan_->points.end(),
                   [&](const PointType& p1, const PointType& p2) {
                     if (point_time_cmp(p1, p2)) {
                       return true;
                     }
                     if (point_time_cmp(p2, p1)) {
                       return false;
                     }
                     return p1.raw_index < p2.raw_index;
                   });

  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points
                                  | boost::adaptors::indexed()
                                  | boost::adaptors::adjacent_filtered(point_time_neq);

  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;

  // compute offset between sweep reference time and first point timestamp
  double offset = 0.0;
  if (this->time_offset_) {
    const auto begin_it = points_unique_timestamps.begin();
    if (begin_it == points_unique_timestamps.end()) {
      this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>(*this->original_scan);
      this->deskew_status = false;
      this->deskew_size = 0;
      return;
    }
    offset = sweep_ref_time - extract_point_time(*begin_it);
  }

  // build list of unique timestamps and indices of first point with each timestamp
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++) {
    timestamps.push_back(extract_point_time(*it) + offset);
    unique_time_indices.push_back(static_cast<int>(it->index()));
  }

  if (timestamps.empty()) {
    this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>(*this->original_scan);
    this->deskew_status = false;
    this->deskew_size = 0;
    return;
  }

  unique_time_indices.push_back(static_cast<int>(deskewed_scan_->points.size()));

  // RCLCPP_INFO_THROTTLE(
  //     this->get_logger(), *this->get_clock(), 1000,
  //     "[deskew dbg] header=%.9f first_raw=%.9f last_raw=%.9f first_adj=%.9f last_adj=%.9f "
  //     "header-first_raw=%.3f ms header-last_raw=%.3f ms span=%.3f ms offset=%.3f ms n_unique=%zu",
  //     sweep_ref_time,
  //     first_point_time_raw,
  //     last_point_time_raw,
  //     timestamps.front(),
  //     timestamps.back(),
  //     1e3 * (sweep_ref_time - first_point_time_raw),
  //     1e3 * (sweep_ref_time - last_point_time_raw),
  //     1e3 * (timestamps.back() - timestamps.front()),
  //     1e3 * offset,
  //     timestamps.size());

  // int median_pt_index = timestamps.size() / 2;
  // this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point
  this->scan_stamp = timestamps[0];

  // if (this->prev_scan_stamp > 0.0) {
  //   RCLCPP_INFO_THROTTLE(
  //       this->get_logger(), *this->get_clock(), 1000,
  //       "[deskew interval dbg] prev_first=%.9f curr_first=%.9f curr_last=%.9f "
  //       "scan_span=%.3f ms query_span=%.3f ms",
  //       this->prev_scan_stamp,
  //       timestamps.front(),
  //       timestamps.back(),
  //       1e3 * (timestamps.back() - timestamps.front()),
  //       1e3 * (timestamps.back() - this->prev_scan_stamp));
  // }

  // The first accepted scan must have calibrated IMU coverage for the whole sweep.
  if (!this->first_valid_scan) {
    if (!this->imuBufferCoversRange(timestamps.front(), timestamps.back())) {
      this->deskew_status = false;
      this->deskew_size = 0;
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "\033[38;5;214m[STARTUP] Deskew refused first scan without full IMU coverage "
          "(scan_start=%.6f, scan_end=%.6f).\033[0m",
          timestamps.front(), timestamps.back());
      return;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }

  // IMU prior & deskewing for second scan onwards
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
  frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                              this->geo.prev_vel.cast<float>(), timestamps);
  this->deskew_size = static_cast<int>(frames.size()); // if integration successful, equal to timestamps.size()

  // if there are no frames between the start and end of the sweep
  // that probably means that there's a sync issue
  if (frames.size() != timestamps.size()) {
    // clang-format off
    std::cerr
      << "\033[1;41m\033[1;37m"
      << "\n"
      << "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n"
      << "  !!                                                                            !!  \n"
      << "  !!   DESKEW FAILED: integrateImu returned " << std::setw(5) << frames.size()
                                    << " frames for " << std::setw(5) << timestamps.size() << " points   !!  \n"
      << "  !!   Scan will be published WITHOUT per-point motion compensation.            !!  \n"
      << "  !!   Likely cause: IMU buffer gap or bad LiDAR/IMU time sync.                !!  \n"
      << "  !!   prev_scan_stamp=" << std::fixed << std::setprecision(6) << this->prev_scan_stamp
                         << "  scan_end=" << timestamps.back() << "                          !!  \n"
      << "  !!                                                                            !!  \n"
      << "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n"
      << "\033[0m\n";
    // clang-format on
    RCLCPP_FATAL(this->get_logger(),
      "DESKEW FAILED: integrateImu got %zu frames for %zu point timestamps "
      "(prev_scan_stamp=%.6f scan_end=%.6f). Scan published at T_prior without deskewing.",
      frames.size(), timestamps.size(), this->prev_scan_stamp, timestamps.back());

    this->T_prior = this->T;
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }

  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  // this->T_prior = frames[median_pt_index];
  this->T_prior = frames[0];

#pragma omp parallel for num_threads(this->num_threads_)
  for (int i = 0; i < timestamps.size(); i++) {

    Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;

    // transform point to world frame
    for (int k = unique_time_indices[i]; k < unique_time_indices[i+1]; k++) {
      auto &pt = deskewed_scan_->points[k];
      pt.getVector4fMap()[3] = 1.;
      pt.getVector4fMap() = T * pt.getVector4fMap();
    }
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;

}

void dlio::OdomNode::initializeInputTarget() {

  this->prev_scan_stamp = this->scan_stamp;

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  // keep history of keyframes
  KeyframeData kf;
  kf.position = this->lidarPose.p;
  kf.orientation = this->lidarPose.q;
  kf.registration_cloud = this->registration_scan;
  kf.mapping_cloud = this->keyframe_mapping_scan;
  kf.covariances = this->gicp.getSourceCovariances();
  kf.timestamp = this->scan_header_stamp;
  kf.transform = this->T_corr;
  this->keyframes.push_back(std::move(kf));

}

void dlio::OdomNode::setInputSource() {
  // Source = current deskewed/filtered scan in world frame.
  // NanoGICP builds a source k-d tree and source covariances from this cloud.
  this->gicp.setInputSource(this->registration_scan);
  this->gicp.calculateSourceCovariances();
}

void dlio::OdomNode::initializeDLIO() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }

  this->dlio_initialized = true;
  std::cout << '\n' << " DLIO initialized!" << '\n';

}

// ROS 2 Jazzy subscription callbacks accept SharedPtr by value; const ref is not a supported callback signature here.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
void dlio::OdomNode::callbackPointCloud(sensor_msgs::msg::PointCloud2::SharedPtr pc) {
  auto record_pointcloud_rate = [this]() {
    constexpr std::size_t kRateWindowSize = 20;
    this->pc_rate_window_.push_back(std::chrono::steady_clock::now());
    if (this->pc_rate_window_.size() > kRateWindowSize) {
      this->pc_rate_window_.pop_front();
    }
    if (this->pc_rate_window_.size() >= 2) {
      const auto now = this->pc_rate_window_.back();
      const double since_print = std::chrono::duration<double>(
          now - this->pc_rate_last_print_).count();
      if (since_print >= 2.0) {
        const double span = std::chrono::duration<double>(
            now - this->pc_rate_window_.front()).count();
        const double rate = static_cast<double>(this->pc_rate_window_.size() - 1) / span;
        const std::time_t t = std::time(nullptr);
        std::tm tm{};
        localtime_r(&t, &tm);
        printf("\033[32m[%02d:%02d:%02d] [DLIO] Pointcloud rate: %.2f Hz\033[0m\n",
               tm.tm_hour, tm.tm_min, tm.tm_sec, rate);
        this->pc_rate_last_print_ = now;
      }
    }
  };

  if (!this->debug_enabled_) {
    record_pointcloud_rate();
  }

  if (this->imu_calibrate_ && !this->imu_calibrated.load()) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "\033[38;5;214m[STARTUP] Dropping pointcloud before IMU calibration is complete.\033[0m");
    return;
  }

  // Keep callback lightweight to avoid blocking DDS receive threads.
  this->enqueuePointCloud(pc);
}

void dlio::OdomNode::processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(main_loop_running_mutex);
  this->main_loop_running = true;
  lock.unlock();

  // Use steady_clock, not this->now(), which is the ROS node clock.
  // When use_sim_time=true the ROS clock is driven by /clock messages and
  // does not advance during computation, so now()-then would be ~0.
  const auto then = std::chrono::steady_clock::now();

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = rclcpp::Time(pc->header.stamp).seconds();
  }

  // DLIO Initialization procedures (IMU calib, gravity align)
  if (!this->dlio_initialized) {
    this->initializeDLIO();
  }

  // Convert incoming scan into DLIO format
  this->getScanFromROS(pc);

  if (!this->original_scan || this->original_scan->empty()) {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
    return;
  }

  // Preprocess points
  this->preprocessPoints();

  if (!this->first_valid_scan) {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
    return;
  }

  if (!this->current_scan || this->current_scan->points.size() <= this->gicp_min_num_points_) {
    RCLCPP_FATAL(this->get_logger(), "Low number of points in the cloud!");
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
    return;
  }

  this->applyDynamicFilterBeforeRegistration();
  if (!this->registration_scan || this->registration_scan->points.size() <= this->gicp_min_num_points_) {
    RCLCPP_FATAL(this->get_logger(), "Low number of points in the registration cloud after dynamic filtering!");
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
    return;
  }

  // Compute Metrics
  this->computeMetrics();

  // Set Adaptive Parameters
  if (this->adaptive_params_) {
    this->setAdaptiveParams();
  }

  // Set new frame as input source
  this->setInputSource();

  // Set initial frame as first keyframe
  if (this->keyframes.size() == 0) {
    this->updateDynamicFilterAfterCorrection();
    this->initializeInputTarget();
    Eigen::Vector3f first_vlin_b, first_vang_b, first_accel_bias, first_gyro_bias;
    {
      std::lock_guard<std::mutex> geo_lock(this->geo.mtx);
      first_vlin_b = this->state.v.lin.b;
      first_vang_b = this->state.v.ang.b;
      first_accel_bias = this->state.b.accel;
      first_gyro_bias = this->state.b.gyro;
    }
    this->recordRunStats(this->scan_stamp, this->T,
                         first_vlin_b, first_vang_b,
                         first_accel_bias, first_gyro_bias);
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_future =
      std::async(std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state);
    this->submap_future.wait(); // wait until completion
    return;
  }

  // Get the next pose via IMU + S2M + GEO
  if (!this->getNextPose()) {
    return;
  }

  this->updateDynamicFilterAfterCorrection();

  // Capture a scan-time odom snapshot immediately after the LiDAR update.
  // This snapshot must travel with the scan so that map<->odom for the published cloud
  // is computed from the same timestamped state as T_all/T_cloud.
  Eigen::Vector3f state_p_scan, state_vlin_b_scan, state_vang_b_scan;
  Eigen::Vector3f state_accel_bias_scan, state_gyro_bias_scan;
  Eigen::Quaternionf state_q_scan;
  {
    std::lock_guard<std::mutex> geo_lock(this->geo.mtx);
    state_p_scan = this->state.p;
    state_q_scan = this->state.q.normalized();
    state_vlin_b_scan = this->state.v.lin.b;
    state_vang_b_scan = this->state.v.ang.b;
    state_accel_bias_scan = this->state.b.accel;
    state_gyro_bias_scan = this->state.b.gyro;
  }
  this->recordRunStats(this->scan_stamp, this->T,
                       state_vlin_b_scan, state_vang_b_scan,
                       state_accel_bias_scan, state_gyro_bias_scan);
  // Update latest map->odom at scan time so IMU-rate map propagation can use it immediately.
  {
    const Eigen::Quaternionf q_bo_scan = state_q_scan.conjugate();
    const Eigen::Vector3f p_bo_scan = -(q_bo_scan._transformVector(state_p_scan));
    Eigen::Matrix4f T_bl_odom_scan = Eigen::Matrix4f::Identity();
    T_bl_odom_scan.block<3,3>(0,0) = q_bo_scan.toRotationMatrix();
    T_bl_odom_scan.block<3,1>(0,3) = p_bo_scan;
    const Eigen::Matrix4f T_map_odom_scan = this->T * T_bl_odom_scan;

    std::lock_guard<std::mutex> map_odom_lock(this->mtx_T_map_odom_latest);
    this->T_map_odom_latest = T_map_odom_scan;
    this->has_T_map_odom_latest = true;
  }

  // Update current keyframe poses and map
  this->updateKeyframes();

  // Build keyframe normals and submap if needed (and if we're not already waiting)
  if (this->new_submap_is_ready) {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_future =
      std::async(std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state);
  } else {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
  }

  // Incremental distance (keep cumulative exact, do not recompute over entire trajectory)
  if (!this->trajectory.empty()) {
    const Eigen::Vector3f& prev = this->trajectory.back().first;
    const double l = (this->state.p - prev).norm();
    if (l >= 0.1) this->length_traversed += l;
  }

  // Keep trajectory only for recent visualization/debug
  this->trajectory.emplace_back(this->state.p, this->state.q);
  if (this->trajectory.size() > 1600) {
    const auto trim_count = static_cast<std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>>::difference_type>(
        this->trajectory.size() - 1600);
    this->trajectory.erase(this->trajectory.begin(), this->trajectory.begin() + trim_count);
  }

  // Update time stamps
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // Publish stuff to ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_) {
    published_cloud = this->current_scan;
  } else {
    published_cloud = this->deskewed_scan;
  }

  this->enqueuePublish(std::move(published_cloud),
                       this->T_corr,
                       this->T,
                       this->scan_stamp,
                       state_p_scan,
                       state_q_scan,
                       state_vlin_b_scan,
                       state_vang_b_scan);

  // Update computation time statistics: rolling 2-second window keyed by scan_stamp.
  {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - then).count();
    std::lock_guard<std::mutex> lk(this->mtx_comp_times_);
    this->comp_times.push_back({this->scan_stamp, elapsed});
    const double cutoff = this->scan_stamp - 2.0;
    while (!this->comp_times.empty() && this->comp_times.front().first < cutoff) {
      this->comp_times.pop_front();
    }
  }

  // this->gicp_hasConverged = this->gicp.hasConverged();

  // Debug statements and publish custom DLIO message
  if (this->debug_enabled_) {
    if (!this->debug_future_.valid() ||
        this->debug_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      this->debug_future_ = std::async(std::launch::async, &dlio::OdomNode::debug, this);
    }
  }

  this->geo.first_opt_done = true;
}

sensor_msgs::msg::Imu::SharedPtr dlio::OdomNode::scaleImuUnitsBeforeTransform(
    const sensor_msgs::msg::Imu::SharedPtr& imu_raw) {

  const Eigen::Vector3f raw_accel(
      static_cast<float>(imu_raw->linear_acceleration.x),
      static_cast<float>(imu_raw->linear_acceleration.y),
      static_cast<float>(imu_raw->linear_acceleration.z));
  const Eigen::Vector3f raw_gyro(
      static_cast<float>(imu_raw->angular_velocity.x),
      static_cast<float>(imu_raw->angular_velocity.y),
      static_cast<float>(imu_raw->angular_velocity.z));
  const double stamp_sec = rclcpp::Time(imu_raw->header.stamp).seconds();

  const dlio::ImuUnitScaleDecision& decision =
      this->imu_unit_scaler_.observe(raw_accel.norm(), raw_gyro.norm(), stamp_sec, std::abs(this->gravity_));

  auto imu_scaled = std::make_shared<sensor_msgs::msg::Imu>(*imu_raw);
  imu_scaled->linear_acceleration.x *= decision.accel_scale;
  imu_scaled->linear_acceleration.y *= decision.accel_scale;
  imu_scaled->linear_acceleration.z *= decision.accel_scale;
  imu_scaled->angular_velocity.x *= decision.gyro_scale;
  imu_scaled->angular_velocity.y *= decision.gyro_scale;
  imu_scaled->angular_velocity.z *= decision.gyro_scale;

  this->maybePrintImuUnitScaleWarning(stamp_sec);
  return imu_scaled;
}

void dlio::OdomNode::maybePrintImuUnitScaleWarning(double stamp_sec) {
  const dlio::ImuUnitScaleDecision& decision = this->imu_unit_scaler_.decision();
  if (!decision.nonIdentity()) {
    return;
  }

  const double warn_period = this->imu_unit_scaler_.config().warn_period_sec;
  if (stamp_sec >= this->imu_unit_scale_last_warn_stamp_ &&
      (stamp_sec - this->imu_unit_scale_last_warn_stamp_) < warn_period) {
    return;
  }
  this->imu_unit_scale_last_warn_stamp_ = stamp_sec;

  const char* orange = "\033[38;5;214m";
  const char* reset = "\033[0m";
  std::ostringstream msg;
  msg << orange << "\n"
      << "======================================================================\n"
      << "  IMU UNIT AUTO-SCALE ACTIVE BEFORE DLIO CALIBRATION\n"
      << "  status: " << (decision.locked ? "locked" : "provisional") << "\n"
      << "  classification: " << decision.classification << "\n"
      << "  raw accel median norm: " << std::fixed << std::setprecision(6)
      << decision.raw_accel_median << "\n"
      << "  raw gyro median norm:  " << std::fixed << std::setprecision(6)
      << decision.raw_gyro_median << "\n"
      << "  accel scale: " << std::fixed << std::setprecision(9)
      << decision.accel_scale << "\n"
      << "  gyro scale:  " << std::fixed << std::setprecision(9)
      << decision.gyro_scale << "\n"
      << "======================================================================\n"
      << reset;
  std::cout << msg.str() << std::flush;
}

// ROS 2 Jazzy subscription callbacks accept SharedPtr by value; const ref is not a supported callback signature here.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
void dlio::OdomNode::callbackImu(sensor_msgs::msg::Imu::SharedPtr imu_raw) {

  this->first_imu_received = true;

  sensor_msgs::msg::Imu::SharedPtr imu_scaled = this->scaleImuUnitsBeforeTransform( imu_raw );
  sensor_msgs::msg::Imu::SharedPtr imu = this->transformImu( imu_scaled );
  this->imu_stamp = imu->header.stamp;
  const double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();

  const Eigen::Vector3f ang_vel(static_cast<float>(imu->angular_velocity.x),
                                static_cast<float>(imu->angular_velocity.y),
                                static_cast<float>(imu->angular_velocity.z));

  const Eigen::Vector3f lin_accel(static_cast<float>(imu->linear_acceleration.x),
                                  static_cast<float>(imu->linear_acceleration.y),
                                  static_cast<float>(imu->linear_acceleration.z));

  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu_stamp_secs;
  }

  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated) {

    if ((imu_stamp_secs - this->first_imu_stamp) < this->imu_calib_time_) {

      this->imu_calib_samples_++;

      this->imu_calib_gyro_sum_[0] += ang_vel[0];
      this->imu_calib_gyro_sum_[1] += ang_vel[1];
      this->imu_calib_gyro_sum_[2] += ang_vel[2];

      this->imu_calib_accel_sum_[0] += lin_accel[0];
      this->imu_calib_accel_sum_[1] += lin_accel[1];
      this->imu_calib_accel_sum_[2] += lin_accel[2];

      if (!this->imu_calib_printed_) {
        std::cout << '\n' << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        this->imu_calib_printed_ = true;
      }

    } else {

      std::cout << "done\n\n";

      const float sample_count = static_cast<float>(std::max(this->imu_calib_samples_, 1));
      const Eigen::Vector3f gyro_avg = this->imu_calib_gyro_sum_ / sample_count;
      const Eigen::Vector3f accel_avg = this->imu_calib_accel_sum_ / sample_count;

      Eigen::Vector3f grav_vec(0.0f, 0.0f, static_cast<float>(this->gravity_));

      if (this->gravity_align_) {
        Eigen::Vector3f accel_bias;
        {
          std::lock_guard<std::mutex> lock(this->geo.mtx);
          accel_bias = this->state.b.accel;
        }

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - accel_bias).normalized() * static_cast<float>(std::abs(this->gravity_));
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(
            grav_vec, Eigen::Vector3f(0.0f, 0.0f, static_cast<float>(this->gravity_)));

        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0/M_PI);
        double pitch = euler[1] * (180.0/M_PI);
        double roll = euler[2] * (180.0/M_PI);

        // use alternate representation if the yaw is smaller
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
          yaw   = remainder(yaw + 180.0,   360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll  = remainder(roll + 180.0,  360.0);
        }
        std::cout << " Estimated initial attitude:\n";
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << '\n';
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << '\n';
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << '\n';
        std::cout << '\n';

        {
          std::lock_guard<std::mutex> lock(this->geo.mtx);
          // set gravity aligned orientation
          this->state.q = grav_q;
          this->T.block(0,0,3,3) = this->state.q.toRotationMatrix();
          this->lidarPose.q = this->state.q;
        }
      }

      if (this->calibrate_accel_) {
        Eigen::Vector3f accel_bias = accel_avg - grav_vec;
        {
          std::lock_guard<std::mutex> lock(this->geo.mtx);
          this->state.b.accel = accel_bias;
        }

        // subtract gravity from avg accel to get bias
        std::cout << " Accel biases [xyz]: " << to_string_with_precision(accel_bias[0], 8) << ", "
                                             << to_string_with_precision(accel_bias[1], 8) << ", "
                                             << to_string_with_precision(accel_bias[2], 8) << '\n';
      }

      if (this->calibrate_gyro_) {
        {
          std::lock_guard<std::mutex> lock(this->geo.mtx);
          this->state.b.gyro = gyro_avg;
        }

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(gyro_avg[0], 8) << ", "
                                             << to_string_with_precision(gyro_avg[1], 8) << ", "
                                             << to_string_with_precision(gyro_avg[2], 8) << '\n';
      }

      this->imu_calib_samples_ = 0;
      this->imu_calib_gyro_sum_.setZero();
      this->imu_calib_accel_sum_.setZero();
      this->imu_calib_printed_ = false;

      Eigen::Quaternionf initial_gravity_align_q = Eigen::Quaternionf::Identity();
      Eigen::Vector3f initial_accel_bias = Eigen::Vector3f::Zero();
      Eigen::Vector3f initial_gyro_bias = Eigen::Vector3f::Zero();
      {
        std::lock_guard<std::mutex> lock(this->geo.mtx);
        initial_gravity_align_q = this->state.q;
        initial_accel_bias = this->state.b.accel;
        initial_gyro_bias = this->state.b.gyro;
      }
      this->captureInitialImuBaseline(
          grav_vec,
          initial_gravity_align_q,
          initial_accel_bias,
          initial_gyro_bias);

      this->clearPointCloudQueue("IMU calibration completed; discarding pre-calibration scans");
      this->imu_calibrated = true;
      this->prev_imu_stamp = rclcpp::Time(imu->header.stamp).seconds();
      this->pc_q_cv_.notify_all();
      this->cv_imu_stamp.notify_all();

    }

  } else {

    double dt = imu_stamp_secs - this->prev_imu_stamp;
    if (dt <= 0) { dt = 1.0/300.0; }
    // this->imu_rates.push_back( 1./dt );

    // Apply the calibrated bias to the new IMU measurements
    this->imu_meas.stamp = imu_stamp_secs;
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f accel_bias = Eigen::Vector3f::Zero();
    Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();
    {
      std::lock_guard<std::mutex> lock(this->geo.mtx);
      accel_bias = this->state.b.accel;
      gyro_bias = this->state.b.gyro;
    }

    Eigen::Vector3f lin_accel_corrected = (this->imu_accel_sm_ * lin_accel) - accel_bias;
    Eigen::Vector3f ang_vel_corrected = ang_vel - gyro_bias;

    this->imu_meas.lin_accel = lin_accel_corrected;
    this->imu_meas.ang_vel = ang_vel_corrected;

    // Store calibrated IMU measurements into imu buffer for manual integration later.
    {
      std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);
      this->imu_buffer.push_front(this->imu_meas);
    }

    // Notify the callbackPointCloud thread that IMU data exists for this time
    this->cv_imu_stamp.notify_one();

    if (this->geo.first_opt_done) {
      // Geometric Observer: Propagate State
      this->propagateState();

      // Publish only after propagation
      this->publishPoseSnapshot();

    }

  }

}

void dlio::OdomNode::publishPoseSnapshot() {
  Eigen::Vector3f p = Eigen::Vector3f::Zero();
  Eigen::Vector3f vlin_b = Eigen::Vector3f::Zero();
  Eigen::Vector3f vang_b = Eigen::Vector3f::Zero();
  Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
  rclcpp::Time stamp(0, 0);

  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    p      = this->state.p;
    q      = this->state.q;
    vlin_b = this->state.v.lin.b;
    vang_b = this->state.v.ang.b;
    stamp  = this->imu_stamp;
  }

  q.normalize();

  // Build and publish Odometry (dlio_odom -> base_link)
  // twist is expressed in child frame (base_link), so use body-frame velocities directly.
  if (hasSubscribers(this->odom_pub)) {
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = this->odom_frame;
    odom_msg.child_frame_id  = this->baselink_frame;

    odom_msg.pose.pose.position.x = p.x();
    odom_msg.pose.pose.position.y = p.y();
    odom_msg.pose.pose.position.z = p.z();
    odom_msg.pose.pose.orientation.w = q.w();
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();

    odom_msg.twist.twist.linear.x  = vlin_b.x();
    odom_msg.twist.twist.linear.y  = vlin_b.y();
    odom_msg.twist.twist.linear.z  = vlin_b.z();
    odom_msg.twist.twist.angular.x = vang_b.x();
    odom_msg.twist.twist.angular.y = vang_b.y();
    odom_msg.twist.twist.angular.z = vang_b.z();

    this->odom_pub->publish(odom_msg);
  }

  // Build and publish PoseStamped (dlio_odom -> base_link)
  if (hasSubscribers(this->pose_pub)) {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = this->odom_frame;
    pose_msg.pose.position.x = p.x();
    pose_msg.pose.position.y = p.y();
    pose_msg.pose.position.z = p.z();
    pose_msg.pose.orientation.w = q.w();
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();

    this->pose_pub->publish(pose_msg);
  }

  // Path: base_link pose in dlio_odom at IMU propagation rate.
  this->path_odom_ros.header.stamp = stamp;
  this->path_odom_ros.header.frame_id = this->odom_frame;

  geometry_msgs::msg::PoseStamped pose_odom;
  pose_odom.header.stamp = stamp;
  pose_odom.header.frame_id = this->odom_frame;
  pose_odom.pose.position.x = p.x();
  pose_odom.pose.position.y = p.y();
  pose_odom.pose.position.z = p.z();
  pose_odom.pose.orientation.w = q.w();
  pose_odom.pose.orientation.x = q.x();
  pose_odom.pose.orientation.y = q.y();
  pose_odom.pose.orientation.z = q.z();

  constexpr size_t kMaxOdomPath = 10000;
  if (this->path_odom_poses_.size() >= kMaxOdomPath) {
    this->path_odom_poses_.pop_front();
  }
  this->path_odom_poses_.push_back(std::move(pose_odom));
  if (hasSubscribers(this->path_odom_pub)) {
    this->path_odom_ros.poses.assign(this->path_odom_poses_.begin(), this->path_odom_poses_.end());
    this->path_odom_pub->publish(this->path_odom_ros);
  }

  // IMU-rate propagated base_link trajectory in dlio_map using latest scan-time map->odom.
  Eigen::Matrix4f T_map_odom = Eigen::Matrix4f::Identity();
  bool has_T_map_odom = false;
  {
    std::lock_guard<std::mutex> map_odom_lock(this->mtx_T_map_odom_latest);
    has_T_map_odom = this->has_T_map_odom_latest;
    if (has_T_map_odom) {
      T_map_odom = this->T_map_odom_latest;
    }
  }
  if (has_T_map_odom) {
    Eigen::Matrix4f T_odom_base = Eigen::Matrix4f::Identity();
    T_odom_base.block<3,3>(0,0) = q.toRotationMatrix();
    T_odom_base.block<3,1>(0,3) = p;

    const Eigen::Matrix4f T_map_base = T_map_odom * T_odom_base;
    const Eigen::Vector3f p_mb_prop = T_map_base.block<3,1>(0,3);
    Eigen::Quaternionf q_mb_prop(T_map_base.block<3,3>(0,0));
    q_mb_prop.normalize();

    this->path_map_prop_ros.header.stamp = stamp;
    this->path_map_prop_ros.header.frame_id = "dlio_map";

    geometry_msgs::msg::PoseStamped pose_map_prop;
    pose_map_prop.header.stamp = stamp;
    pose_map_prop.header.frame_id = "dlio_map";
    pose_map_prop.pose.position.x = p_mb_prop.x();
    pose_map_prop.pose.position.y = p_mb_prop.y();
    pose_map_prop.pose.position.z = p_mb_prop.z();
    pose_map_prop.pose.orientation.w = q_mb_prop.w();
    pose_map_prop.pose.orientation.x = q_mb_prop.x();
    pose_map_prop.pose.orientation.y = q_mb_prop.y();
    pose_map_prop.pose.orientation.z = q_mb_prop.z();

    constexpr size_t kMaxMapPropPath = 10000;
    if (this->path_map_prop_poses_.size() >= kMaxMapPropPath) {
      this->path_map_prop_poses_.pop_front();
    }
    this->path_map_prop_poses_.push_back(std::move(pose_map_prop));
    if (hasSubscribers(this->path_map_prop_pub)) {
      this->path_map_prop_ros.poses.assign(this->path_map_prop_poses_.begin(), this->path_map_prop_poses_.end());
      this->path_map_prop_pub->publish(this->path_map_prop_ros);
    }
  }

  // TF: base_link -> dlio_odom (inverted state)
  const Eigen::Quaternionf q_inv = q.conjugate();
  const Eigen::Vector3f p_inv = -(q_inv._transformVector(p));

  geometry_msgs::msg::TransformStamped tf_bl_odom;
  tf_bl_odom.header.stamp = stamp;
  tf_bl_odom.header.frame_id = this->baselink_frame;
  tf_bl_odom.child_frame_id  = this->odom_frame;
  tf_bl_odom.transform.translation.x = p_inv.x();
  tf_bl_odom.transform.translation.y = p_inv.y();
  tf_bl_odom.transform.translation.z = p_inv.z();
  tf_bl_odom.transform.rotation.w = q_inv.w();
  tf_bl_odom.transform.rotation.x = q_inv.x();
  tf_bl_odom.transform.rotation.y = q_inv.y();
  tf_bl_odom.transform.rotation.z = q_inv.z();
  br->sendTransform(tf_bl_odom);

  this->publishVelocityMarkers(stamp, vlin_b, vang_b);
}

void dlio::OdomNode::publishCorrectionMarker(
    const rclcpp::Time& stamp,
    const Eigen::Ref<const Eigen::Matrix4f>& T_corr,
    const Eigen::Ref<const Eigen::Matrix4f>& T_all) {
  if (!this->viz_corr_marker_) {
    return;
  }
  if (!this->pub_corr_marker_) {
    return;
  }

  if (!hasSubscribers(this->pub_corr_marker_)) {
    return;
  }

  // Recover T_prior from T_all = T_corr * T_prior without a full 4x4 inverse.
  const Eigen::Matrix3f R_corr = T_corr.block<3,3>(0,0);
  const Eigen::Vector3f t_corr = T_corr.block<3,1>(0,3);
  const Eigen::Vector3f p_all  = T_all.block<3,1>(0,3);
  const Eigen::Vector3f p_prior = R_corr.transpose() * (p_all - t_corr);
  const Eigen::Vector3f corr_vec = p_all - p_prior;

  visualization_msgs::msg::Marker marker;
  this->createCorrectionMarker("dlio_map", stamp, p_prior, corr_vec, marker);
  this->pub_corr_marker_->publish(marker);
}

void dlio::OdomNode::analyzeDegeneracyFromCurrentScan(
    const Eigen::Ref<const Eigen::Matrix4f>& T_map_base) {
  const auto covs = this->gicp.getSourceCovariances();
  if (!covs || covs->empty()) {
    this->degen_info_.valid = false;
    return;
  }

  const NormalScatterStats normal_stats = buildNormalScatterMatrix(*covs);
  if (normal_stats.valid_normals < 3 ||
      normal_stats.total_weight <= 0.0 ||
      !normal_stats.scatter.allFinite()) {
    this->degen_info_.valid = false;
    return;
  }

  const Eigen::Matrix3d normal_spread = normal_stats.scatter;
  if (!normal_spread.allFinite()) {
    this->degen_info_.valid = false;
    return;
  }

  this->degen_info_.p_map_base = T_map_base.block<3,1>(0,3).cast<double>();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_trans(normal_spread);
  if (eig_trans.info() != Eigen::Success) {
    this->degen_info_.valid = false;
    return;
  }

  Eigen::Vector3d trans_evals_sorted = Eigen::Vector3d::Zero();
  Eigen::Matrix3d trans_evecs_sorted = Eigen::Matrix3d::Identity();

  sortEigenpairsAscending(
      eig_trans.eigenvalues(), eig_trans.eigenvectors(),
      trans_evals_sorted, trans_evecs_sorted);

  for (int k = 0; k < 3; ++k) {
    trans_evecs_sorted.col(k).normalize();
  }

  this->degen_info_.valid = true;
  this->degen_info_.eigvals_trans_dec = trans_evals_sorted;
  this->degen_info_.eigvecs_trans_map = trans_evecs_sorted;
  this->degen_info_.weak_trans = classifyWeakDirections(
      this->degen_info_.eigvals_trans_dec,
      this->degen_trans_eig_abs_thresh_);

  constexpr double eps = 1e-12;
  const double trans_min = std::max(this->degen_info_.eigvals_trans_dec(0), eps);
  const double trans_max = std::max(this->degen_info_.eigvals_trans_dec(2), eps);
  this->degen_info_.trans_condition = trans_max / trans_min;

  if (hasWeakDirection(this->degen_info_.weak_trans)) {
    logTranslationDegeneracy(
        this->get_logger(),
        this->degen_info_.eigvals_trans_dec,
        this->degen_info_.eigvecs_trans_map,
        this->degen_info_.weak_trans);
  } 
  // else {
  //   logTranslationSpectrumAlways(
  //       this->get_logger(),
  //       this->degen_info_.eigvals_trans_dec,
  //       this->degen_info_.eigvecs_trans_map);
  // }
}

void dlio::OdomNode::publishDegeneracyMarkers(const rclcpp::Time& stamp) {
  if (!this->viz_degen_marker_ || !this->pub_degen_marker_) {
    return;
  }
  if (!hasSubscribers(this->pub_degen_marker_)) {
    return;
  }

  visualization_msgs::msg::MarkerArray marker_array;
  marker_array.markers.reserve(3);

  auto pushDelete = [&](const int id, const std::string& ns) {
    visualization_msgs::msg::Marker m;
    m.header.stamp = stamp;
    m.header.frame_id = "dlio_map";
    m.ns = ns;
    m.id = id;
    m.action = visualization_msgs::msg::Marker::DELETE;
    marker_array.markers.push_back(std::move(m));
  };

  if (!this->degen_info_.valid) {
    for (int i = 0; i < 3; ++i) {
      pushDelete(i, "degeneracy_translation");
    }
    if (hasSubscribers(this->pub_degen_marker_)) {
      this->pub_degen_marker_->publish(marker_array);
    }
    return;
  }

  const double eps = 1e-12;

  const Eigen::Vector3d origin = this->degen_info_.p_map_base;
  const double trans_lambda_max = std::max(this->degen_info_.eigvals_trans_dec.maxCoeff(), eps);

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto pushArrow = [&](const int id,
                       const std::string& ns,
                       const Eigen::Vector3d& dir_map,
                       const double length,
                       const Eigen::Vector3f& color) {
    visualization_msgs::msg::Marker m;
    m.header.stamp = stamp;
    m.header.frame_id = "dlio_map";
    m.ns = ns;
    m.id = id;
    m.type = visualization_msgs::msg::Marker::ARROW;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.lifetime = rclcpp::Duration::from_seconds(this->viz_degen_lifetime_);
    m.scale.x = this->viz_degen_shaft_diam_;
    m.scale.y = this->viz_degen_head_diam_;
    m.scale.z = this->viz_degen_head_len_;
    m.color.a = 1.0;
    m.color.r = color.x();
    m.color.g = color.y();
    m.color.b = color.z();

    m.pose.orientation.x = 0.0;
    m.pose.orientation.y = 0.0;
    m.pose.orientation.z = 0.0;
    m.pose.orientation.w = 1.0;

    geometry_msgs::msg::Point p0;
    p0.x = origin.x();
    p0.y = origin.y();
    p0.z = origin.z();

    geometry_msgs::msg::Point p1;
    p1.x = origin.x() + length * dir_map.x();
    p1.y = origin.y() + length * dir_map.y();
    p1.z = origin.z() + length * dir_map.z();

    m.points.push_back(p0);
    m.points.push_back(p1);
    marker_array.markers.push_back(std::move(m));
  };

  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d dir_trans = this->degen_info_.eigvecs_trans_map.col(i).normalized();
    if (!dir_trans.allFinite()) {
      pushDelete(i, "degeneracy_translation");
      continue;
    }

    // Eigenvectors are sign-ambiguous. Keep temporal sign continuity to avoid RViz flicker.
    if (this->degen_prev_dirs_initialized_ && dir_trans.dot(this->degen_prev_trans_dirs_map_[i]) < 0.0) {
      dir_trans = -dir_trans;
    }
    this->degen_prev_trans_dirs_map_[i] = dir_trans;

    if (this->degen_info_.weak_trans[i]) {
      // Visual severity: weaker curvature (smaller lambda / lambda_max) => longer arrow.
      const double ratio = this->degen_info_.eigvals_trans_dec(i) / trans_lambda_max;
      const double severity = std::min(1.0, std::max(0.0, 1.0 - ratio));
      const double length = this->viz_degen_trans_scale_ * (0.35 + 0.65 * severity);
      const bool is_min_mode = (i == 0);
      pushArrow(i, "degeneracy_translation", dir_trans, length,
                Eigen::Vector3f(1.00f, is_min_mode ? 0.10f : 0.55f, 0.10f));
    } else {
      pushDelete(i, "degeneracy_translation");
    }
  }

  this->degen_prev_dirs_initialized_ = true;
  if (hasSubscribers(this->pub_degen_marker_)) {
    this->pub_degen_marker_->publish(marker_array);
  }
}

void dlio::OdomNode::createCorrectionMarker(
    const std::string& frame_id,
    const rclcpp::Time& stamp,
    const Eigen::Vector3f& start_m,
    const Eigen::Vector3f& corr_vec_m,
    visualization_msgs::msg::Marker& marker) {
  const Eigen::Vector3f end_m = start_m + static_cast<float>(this->viz_corr_gain_) * corr_vec_m;

  geometry_msgs::msg::Point p0, p1;
  p0.x = start_m.x();
  p0.y = start_m.y();
  p0.z = start_m.z();
  p1.x = end_m.x();
  p1.y = end_m.y();
  p1.z = end_m.z();

  this->corr_marker_points_.push_back(p0);
  this->corr_marker_points_.push_back(p1);
  const size_t max_points = static_cast<size_t>(2 * std::max(1, this->viz_corr_max_segments_));
  if (this->corr_marker_points_.size() > max_points) {
    const auto trim_count =
        static_cast<std::vector<geometry_msgs::msg::Point>::difference_type>(
            this->corr_marker_points_.size() - max_points);
    this->corr_marker_points_.erase(
      this->corr_marker_points_.begin(),
      this->corr_marker_points_.begin() + trim_count);
  }

  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.ns = "correction_lines";
  marker.id = 2;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.lifetime = rclcpp::Duration::from_seconds(this->viz_corr_lifetime_);
  marker.scale.x = this->viz_corr_line_width_;

  marker.color.a = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 0.35;
  marker.color.b = 0.0;

  marker.points = this->corr_marker_points_;

  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
}

void dlio::OdomNode::publishVelocityMarkers(const rclcpp::Time& stamp,
                                            const Eigen::Vector3f& vlin_b,
                                            const Eigen::Vector3f& vang_b) {
  if (!viz_vel_markers_) return;

  const bool want_lin_marker = hasSubscribers(this->pub_lin_vel_marker_);
  const bool want_ang_marker = hasSubscribers(this->pub_ang_vel_marker_);
  if (!want_lin_marker && !want_ang_marker) {
    return;
  }

  if (want_lin_marker) {
    visualization_msgs::msg::Marker m_lin;
    createLinVelocityMarker(this->baselink_frame, stamp, vlin_b, m_lin);
    this->pub_lin_vel_marker_->publish(m_lin);
  }
  if (want_ang_marker) {
    visualization_msgs::msg::Marker m_ang;
    createAngularVelocityMarker(this->baselink_frame, stamp, vang_b, m_ang);
    this->pub_ang_vel_marker_->publish(m_ang);
  }
}

void dlio::OdomNode::createLinVelocityMarker(const std::string& frame_id, const rclcpp::Time& stamp,
                                             const Eigen::Vector3f& v_b,
                                             visualization_msgs::msg::Marker& marker) {
  // Arrow
  marker.header.frame_id = frame_id;
  marker.header.stamp    = stamp;
  marker.id   = 0;
  marker.type = visualization_msgs::msg::Marker::ARROW;
  marker.action = visualization_msgs::msg::Marker::ADD;

  // Scale and Color
  marker.scale.x = 0.1;  // shaft diameter
  marker.scale.y = 0.2;  // head diameter
  marker.scale.z = 0.2;  // head length
  marker.color.a = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;

  // Define Arrow through start and end point
  geometry_msgs::msg::Point startPoint, endPoint;
  startPoint.x = 0.0;  // origin
  startPoint.y = 0.0;  // origin
  startPoint.z = 0.0;  // 0 meter above origin
  endPoint.x = startPoint.x + static_cast<double>(v_b.x());
  endPoint.y = startPoint.y + static_cast<double>(v_b.y());
  endPoint.z = startPoint.z + static_cast<double>(v_b.z());
  marker.points.clear();
  marker.points.push_back(startPoint);
  marker.points.push_back(endPoint);

  // Quaternion for orientation
  tf2::Quaternion q;
  q.setRPY(0, 0, 0);
  marker.pose.orientation.x = q.x();
  marker.pose.orientation.y = q.y();
  marker.pose.orientation.z = q.z();
  marker.pose.orientation.w = q.w();
}

void dlio::OdomNode::createAngularVelocityMarker(const std::string& frame_id, const rclcpp::Time& stamp,
                                                 const Eigen::Vector3f& w_b,
                                                 visualization_msgs::msg::Marker& marker) {
  // Cylinder to visualize angular velocity as a disc/ring oriented along rotation axis
  marker.header.frame_id = frame_id;
  marker.header.stamp    = stamp;
  marker.ns   = "angular_velocity";
  marker.id   = 1;
  marker.type = visualization_msgs::msg::Marker::CYLINDER;
  marker.action = visualization_msgs::msg::Marker::ADD;

  // Angular velocity magnitude
  const double angularMagnitude = static_cast<double>(w_b.norm());

  if (angularMagnitude > 1e-6) {
    // Scale based on angular velocity magnitude
    const double baseRadius = std::min(std::max(angularMagnitude * 0.2, 0.1), 0.5);
    marker.scale.x = baseRadius * 2.0;  // diameter in x
    marker.scale.y = baseRadius * 2.0;  // diameter in y
    marker.scale.z = 0.02;              // thin disc height

    // Color: blue for angular velocity with alpha based on magnitude
    marker.color.a = static_cast<float>(std::min(angularMagnitude * 0.5 + 0.3, 1.0));
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
  } else {
    // No significant angular velocity - make marker invisible
    marker.scale.x = 0.0;
    marker.scale.y = 0.0;
    marker.scale.z = 0.0;
    marker.color.a = 0.0;
  }

  // Set lifetime
  marker.lifetime = rclcpp::Duration::from_seconds(0.1);

  // Position at current pose position
  marker.pose.position.x = 0.0;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 0.0;

  // Orient the disc perpendicular to the angular velocity vector (rotation axis)
  if (angularMagnitude > 1e-6) {
    Eigen::Vector3d rotationAxis = w_b.cast<double>().normalized();

    // Create a rotation that aligns the cylinder's z-axis with the rotation axis
    // Default cylinder orientation is along z-axis
    Eigen::Vector3d zAxis(0.0, 0.0, 1.0);

    // Calculate rotation to align z-axis with rotation axis
    Eigen::Quaterniond orientation;
    const double dot = rotationAxis.dot(zAxis);
    if (dot > 0.9999) {
      // Already aligned
      orientation = Eigen::Quaterniond::Identity();
    } else if (dot < -0.9999) {
      // Opposite direction - rotate 180 degrees around x-axis
      orientation = Eigen::Quaterniond(0.0, 1.0, 0.0, 0.0);
    } else {
      // General case - use cross product to find rotation axis
      Eigen::Vector3d rotAxis = zAxis.cross(rotationAxis).normalized();
      const double c = std::max(-1.0, std::min(1.0, zAxis.dot(rotationAxis)));
      const double angle = std::acos(c);
      orientation = Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotAxis));
    }

    marker.pose.orientation.x = orientation.x();
    marker.pose.orientation.y = orientation.y();
    marker.pose.orientation.z = orientation.z();
    marker.pose.orientation.w = orientation.w();
  } else {
    // Default orientation
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
  }
}

bool dlio::OdomNode::getNextPose() {

  // Check if the new submap is ready to be used
  this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);

  if (this->new_submap_is_ready && this->submap_hasChanged) {

    // Set the current global submap as the target cloud
    this->gicp.registerInputTarget(this->submap_cloud);

    // Set submap kdtree
    this->gicp.target_kdtree_ = this->submap_kdtree;

    // Set target cloud's normals as submap normals
    this->gicp.setTargetCovariances(this->submap_normals);

    this->submap_hasChanged = false;
  }

  const Eigen::Matrix4f T_corr_guess = Eigen::Matrix4f::Identity();

  if (this->use_degeneracy_) {
    // Detection and visualization only — does not affect registration or state estimation.
    // Use the current scan's normal spread rather than registration linearization.
    this->analyzeDegeneracyFromCurrentScan(this->T_prior);
    this->publishDegeneracyMarkers(this->scan_header_stamp);

    const bool live_degenerate =
        this->degen_info_.valid && hasWeakDirection(this->degen_info_.weak_trans);

    if (live_degenerate) {
      ++this->degen_consecutive_hits_;

      if (this->degen_reset_consecutive_count_ > 0) {
        RCLCPP_WARN(this->get_logger(),
                    "[DEGEN] Consecutive live degeneracy detections: %d/%d",
                    this->degen_consecutive_hits_,
                    this->degen_reset_consecutive_count_);
      }

      if (this->degen_reset_consecutive_count_ > 0 &&
          this->degen_consecutive_hits_ >= this->degen_reset_consecutive_count_) {
        std::ostringstream reason;
        reason << "live scan degeneracy detected for "
               << this->degen_consecutive_hits_ << " consecutive scans";
        if (this->triggerInternalReset(reason.str())) {
          return false;
        }
      }
    } else {
      this->degen_consecutive_hits_ = 0;
    }
  } else {
    this->degen_consecutive_hits_ = 0;
  }

  // Run scan-to-submap registration unconditionally.
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  this->gicp.align(*aligned, T_corr_guess);

  this->T_corr = this->gicp.getFinalTransformation();
  this->T = this->T_corr * this->T_prior;
  this->gicp_hasConverged = this->gicp.hasConverged();

  // this->computeMotionDeviation();

  // Update next global pose
  // Both source and target clouds are in the global frame now, so transformation is global
  this->propagateGICP();

  // Geometric observer update using LiDAR registration result
  this->updateState();
  return true;
}

bool dlio::OdomNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,  // NOLINT(bugprone-easily-swappable-parameters)
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {

  static thread_local boost::circular_buffer<ImuMeas> imu_snapshot;

  {
    std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);

    // pointCloudWorkerLoop() is now responsible for waiting.
    // This function should only snapshot and search.
    if (this->stop_.load(std::memory_order_relaxed) ||
        this->imu_buffer.empty() ||
        this->imu_buffer.front().stamp < end_time) {
      return false;
    }

    imu_snapshot = this->imu_buffer;
  }

  if (imu_snapshot.empty()) {
    return false;
  }

  auto imu_it = imu_snapshot.begin();
  auto last_imu_it = imu_it;
  ++imu_it;

  while (imu_it != imu_snapshot.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it;
    ++imu_it;
  }

  while (imu_it != imu_snapshot.end() && imu_it->stamp >= start_time) {
    ++imu_it;
  }

  if (imu_it == imu_snapshot.end()) {
    return false;
  }
  ++imu_it;

  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps) {
  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // invalid input, return empty vector
    return {};
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) == false) {
    // not enough IMU measurements, return empty vector
    return {};
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = *begin_imu_it;
  const ImuMeas& f2 = *(begin_imu_it+1);

  // Time between first two IMU samples
  const double dt = f2.dt;
  if (dt <= 0.0) {
    return {};
  }
  const float dtf = static_cast<float>(dt);

  // Time between first IMU sample and start_time
  const double idt = start_time - f1.stamp;
  const float idtf = static_cast<float>(idt);
  const float gravity = static_cast<float>(this->gravity_);

  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dtf;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5f * alpha * idtf);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
    q_init.w() - 0.5f * ( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idtf,
    q_init.x() + 0.5f * ( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idtf,
    q_init.y() + 0.5f * ( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idtf,
    q_init.z() + 0.5f * ( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idtf
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = f1.ang_vel + 0.5f * alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5f * ( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dtf,
    q_init.x() + 0.5f * ( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dtf,
    q_init.y() + 0.5f * ( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dtf,
    q_init.z() + 0.5f * ( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dtf
  );
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= gravity;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= gravity;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dtf;

  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1 * idtf + 0.5f * j * idtf * idtf;

  // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init * idtf + 0.5f * a1 * idtf * idtf + (1.0f / 6.0f) * j * idtf * idtf * idtf;

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, begin_imu_it, end_imu_it);
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImuInternal(const Eigen::Quaternionf& q_init, const Eigen::Vector3f& p_init,
                                     const Eigen::Vector3f& v_init, const std::vector<double>& sorted_timestamps,
                                     const boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,  // NOLINT(bugprone-easily-swappable-parameters)
                                     const boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;

  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel);
  const float gravity = static_cast<float>(this->gravity_);
  a[2] -= gravity;

  // Iterate over IMU measurements and timestamps
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it = sorted_timestamps.begin();

  for (; imu_it != end_imu_it; imu_it++) {

    const ImuMeas& f0 = *prev_imu_it;
    const ImuMeas& f = *imu_it;

    // Time between IMU samples
    const double dt = f.dt;
    if (dt <= 0.0) {
      prev_imu_it = imu_it;
      continue;
    }
    const float dtf = static_cast<float>(dt);

    // Angular acceleration
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dtf;

    // Average angular velocity
    Eigen::Vector3f omega = f0.ang_vel + 0.5f * alpha_dt;

    const Eigen::Quaternionf q0 = q;

    // Orientation at current IMU sample
    q = Eigen::Quaternionf (
      q0.w() - 0.5f * ( q0.x()*omega[0] + q0.y()*omega[1] + q0.z()*omega[2] ) * dtf,
      q0.x() + 0.5f * ( q0.w()*omega[0] - q0.z()*omega[1] + q0.y()*omega[2] ) * dtf,
      q0.y() + 0.5f * ( q0.z()*omega[0] + q0.w()*omega[1] - q0.x()*omega[2] ) * dtf,
      q0.z() + 0.5f * ( q0.x()*omega[1] - q0.y()*omega[0] + q0.w()*omega[2] ) * dtf
    );
    q.normalize();

    // Acceleration
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= gravity;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dtf;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // Time between previous IMU sample and given timestamp
      const double idt = *stamp_it - f0.stamp;
      const float idtf = static_cast<float>(idt);

      // Average angular velocity
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5f * alpha * idtf;

      // Orientation at interpolated timestamp (must start from q0, not q)
      Eigen::Quaternionf q_i (
        q0.w() - 0.5f * ( q0.x()*omega_i[0] + q0.y()*omega_i[1] + q0.z()*omega_i[2] ) * idtf,
        q0.x() + 0.5f * ( q0.w()*omega_i[0] - q0.z()*omega_i[1] + q0.y()*omega_i[2] ) * idtf,
        q0.y() + 0.5f * ( q0.z()*omega_i[0] + q0.w()*omega_i[1] - q0.x()*omega_i[2] ) * idtf,
        q0.z() + 0.5f * ( q0.x()*omega_i[1] - q0.y()*omega_i[0] + q0.w()*omega_i[2] ) * idtf
      );
      q_i.normalize();

      // Position
      Eigen::Vector3f p_i = p + v * idtf + 0.5f * a0 * idtf * idtf + (1.0f / 6.0f) * j * idtf * idtf * idtf;

      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      ++stamp_it;
    }

    // Position
    p += v * dtf + 0.5f * a0 * dtf * dtf + (1.0f / 6.0f) * j_dt * dtf * dtf;

    // Velocity
    v += a0 * dtf + 0.5f * j_dt * dtf;

    prev_imu_it = imu_it;

  }
  return imu_se3;
}

void dlio::OdomNode::propagateGICP() {

  this->lidarPose.p << this->T(0,3), this->T(1,3), this->T(2,3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0,0), this->T(0,1), this->T(0,2),
            this->T(1,0), this->T(1,1), this->T(1,2),
            this->T(2,0), this->T(2,1), this->T(2,2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  const float norm = std::sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

void dlio::OdomNode::propagateState() {
  std::lock_guard<std::mutex> lock(this->geo.mtx);

  const double dt = this->imu_meas.dt;
  if (dt <= 0) return;
  const float dtf = static_cast<float>(dt);
  const float half_dt_sq = 0.5f * dtf * dtf;

  // Body specific force (bias already removed in callbackImu)
  const Eigen::Vector3f f_b = this->imu_meas.lin_accel;

  // Rotation and gravity (world frame)
  const Eigen::Matrix3f Rwb = this->state.q.toRotationMatrix();
  const Eigen::Vector3f g_w(0.0f, 0.0f, static_cast<float>(this->gravity_));

  // World-frame acceleration
  const Eigen::Vector3f a_w = Rwb * f_b - g_w;

  // Integrate p, v (world frame)
  this->state.p      += this->state.v.lin.w * dtf + a_w * half_dt_sq;
  this->state.v.lin.w += a_w * dtf;
  this->state.v.lin.b  = Rwb.transpose() * this->state.v.lin.w;

  // Integrate attitude with measured (bias-corrected) omega (body frame)
  const Eigen::Vector3f omega_b = this->state.v.ang.b = this->imu_meas.ang_vel;
  const Eigen::Quaternionf omega_q(0.f, omega_b.x(), omega_b.y(), omega_b.z());
  Eigen::Quaternionf qdot = (this->state.q * omega_q);
  this->state.q.coeffs() += 0.5f * dtf * qdot.coeffs();
  this->state.q.normalize();

  // Update angular velocity in world for later use
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;
}

void dlio::OdomNode::updateState() {

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  const double dt = this->scan_stamp - this->prev_scan_stamp;
  const float dtf = static_cast<float>(dt);

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;
  const Eigen::Quaternionf qhat_conj = qhat.conjugate();

  // Constuct error quaternion
  qe = qhat_conj * qin;

  float sgn = 1.0f;
  if (qe.w() < 0) {
    sgn = -1.0f;
  }

  // Construct quaternion correction
  qcorr.w() = 1.0f - std::abs(qe.w());
  qcorr.vec() = sgn * qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;

  err_body = qhat_conj._transformVector(err);

  const float abias_max = static_cast<float>(this->geo_abias_max_);
  const float gbias_max = static_cast<float>(this->geo_gbias_max_);
  const float kab = static_cast<float>(this->geo_Kab_);
  const float kgb = static_cast<float>(this->geo_Kgb_);
  const float kp = static_cast<float>(this->geo_Kp_);
  const float kv = static_cast<float>(this->geo_Kv_);
  const float kq = static_cast<float>(this->geo_Kq_);

  // Update accel bias
  this->state.b.accel -= dtf * kab * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  this->state.b.gyro[0] -= dtf * kgb * qe.w() * qe.x();
  this->state.b.gyro[1] -= dtf * kgb * qe.w() * qe.y();
  this->state.b.gyro[2] -= dtf * kgb * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  this->state.p += dtf * kp * err;
  this->state.v.lin.w += dtf * kv * err;

  this->state.q.w() += dtf * kq * qcorr.w();
  this->state.q.x() += dtf * kq * qcorr.x();
  this->state.q.y() += dtf * kq * qcorr.y();
  this->state.q.z() += dtf * kq * qcorr.z();
  this->state.q.normalize();

  // Recompute body-frame velocity to match the corrected world velocity and orientation.
  this->state.v.lin.b = this->state.q.toRotationMatrix().transpose() * this->state.v.lin.w;

  // store previous pose, orientation, and velocity
  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

}

sensor_msgs::msg::Imu::SharedPtr dlio::OdomNode::transformImu(const sensor_msgs::msg::Imu::SharedPtr& imu_raw) {

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();

  // Copy header
  imu->header = imu_raw->header;

  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();
  const bool have_prev_transform = this->imu_transform_prev_valid_;
  double dt = have_prev_transform ? (imu_stamp_secs - this->imu_transform_prev_stamp_) : (1.0 / 300.0);
  this->imu_transform_prev_stamp_ = imu_stamp_secs;
  this->imu_transform_prev_valid_ = true;
  
  if (dt <= 0) { dt = 1.0/300.0; }
  const float dtf = static_cast<float>(dt);

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(static_cast<float>(imu_raw->angular_velocity.x),
                          static_cast<float>(imu_raw->angular_velocity.y),
                          static_cast<float>(imu_raw->angular_velocity.z));

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  const Eigen::Vector3f ang_vel_cg_prev = have_prev_transform ? this->imu_transform_ang_vel_prev_ : ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(static_cast<float>(imu_raw->linear_acceleration.x),
                            static_cast<float>(imu_raw->linear_acceleration.y),
                            static_cast<float>(imu_raw->linear_acceleration.z));

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dtf).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  this->imu_transform_ang_vel_prev_ = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;

}

void dlio::OdomNode::computeMetrics() {
  this->computeSpaciousness();
  this->computeDensity();
}

void dlio::OdomNode::computeSpaciousness() {

  if (!this->original_scan || this->original_scan->empty()) {
    return;
  }

  // compute range of points
  std::vector<float> ds;
  ds.reserve(this->original_scan->points.size());

  for (const auto& point : this->original_scan->points) {
    const float d = std::sqrt(point.x * point.x + point.y * point.y);
    ds.push_back(d);
  }

  if (ds.empty()) {
    return;
  }

  // median
  const auto median_index =
      static_cast<std::vector<float>::difference_type>(ds.size() / 2U);
  std::nth_element(ds.begin(), ds.begin() + median_index, ds.end());
  float median_curr = ds[static_cast<std::size_t>(median_index)];
  if (!this->spaciousness_lpf_initialized_) {
    this->spaciousness_lpf_prev_ = median_curr;
    this->spaciousness_lpf_initialized_ = true;
  }
  float median_lpf = 0.95f * this->spaciousness_lpf_prev_ + 0.05f * median_curr;
  this->spaciousness_lpf_prev_ = median_lpf;

  std::lock_guard<std::mutex> lock(g_metrics_mutex);

  // push
  this->metrics.spaciousness.push_back( median_lpf );
  if (this->metrics.spaciousness.size() > 400) {
    this->metrics.spaciousness.pop_front();
  }

  if (this->metrics.density.size() > 400) {
    this->metrics.density.pop_front();
  }

}

void dlio::OdomNode::computeDensity() {

  float density;

  if (!this->geo.first_opt_done) {
    density = 0.;
  } else {
    density = this->gicp.source_density_;
  }

  if (!this->density_lpf_initialized_) {
    this->density_lpf_prev_ = density;
    this->density_lpf_initialized_ = true;
  }
  float density_lpf = 0.95f * this->density_lpf_prev_ + 0.05f * density;
  this->density_lpf_prev_ = density_lpf;

  std::lock_guard<std::mutex> lock(g_metrics_mutex);
  this->metrics.density.push_back( density_lpf );

}

void dlio::OdomNode::computeConvexHull() {

  // at least 4 keyframes for convex hull
  if (this->num_processed_keyframes < 4) {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].position[0];
    pt.y = this->keyframes[i].position[1];
    pt.z = this->keyframes[i].position[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the convex hull of the point cloud
  this->convex_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the convex hull
  pcl::PointCloud<PointType>::Ptr convex_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->convex_hull.reconstruct(*convex_points);

  pcl::PointIndices::Ptr convex_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->convex_hull.getHullPointIndices(*convex_hull_point_idx);

  this->keyframe_convex.clear();
  for (int i=0; i<convex_hull_point_idx->indices.size(); ++i) {
    this->keyframe_convex.push_back(convex_hull_point_idx->indices[i]);
  }

}

void dlio::OdomNode::computeConcaveHull() {

  // at least 5 keyframes for concave hull
  if (this->num_processed_keyframes < 5) {
    return;
  }

  // create a pointcloud with points at keyframes
  auto cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].position[0];
    pt.y = this->keyframes[i].position[1];
    pt.z = this->keyframes[i].position[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the concave hull of the point cloud
  this->concave_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the concave hull
  pcl::PointCloud<PointType>::Ptr concave_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->concave_hull.reconstruct(*concave_points);

  pcl::PointIndices::Ptr concave_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->concave_hull.getHullPointIndices(*concave_hull_point_idx);

  this->keyframe_concave.clear();
  for (int i=0; i<concave_hull_point_idx->indices.size(); ++i) {
    this->keyframe_concave.push_back(concave_hull_point_idx->indices[i]);
  }

}

void dlio::OdomNode::updateKeyframes() {
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  if (this->keyframes.empty()) {
    KeyframeData kf;
    kf.position = this->lidarPose.p;
    kf.orientation = this->lidarPose.q;
    kf.registration_cloud = this->registration_scan;
    kf.mapping_cloud = this->keyframe_mapping_scan;
    kf.covariances = this->gicp.getSourceCovariances();
    kf.timestamp = this->scan_header_stamp;
    kf.transform = this->T_corr;
    this->keyframes.emplace_back(std::move(kf));
    return;
  }

  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;
  int num_nearby = 0;
  bool nearby_nonempty_mapping = false;

  for (const auto& k : this->keyframes) {
    float dx = this->lidarPose.p[0] - k.position[0];
    float dy = this->lidarPose.p[1] - k.position[1];
    float dz = this->lidarPose.p[2] - k.position[2];
    float delta_d = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5f) ++num_nearby;
    if (delta_d <= this->keyframe_thresh_dist_ &&
        k.mapping_cloud &&
        !k.mapping_cloud->empty()) {
      nearby_nonempty_mapping = true;
    }
    if (delta_d < closest_d) { closest_d = delta_d; closest_idx = keyframes_idx; }
    ++keyframes_idx;
  }

  const Eigen::Vector3f&    closest_pose   = this->keyframes[closest_idx].position;
  const Eigen::Quaternionf& closest_pose_r = this->keyframes[closest_idx].orientation;

  const float dd = closest_d;

  Eigen::Quaternionf dq;
  if (this->lidarPose.q.dot(closest_pose_r) < 0.f) {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w()*=-1.f; lq.x()*=-1.f; lq.y()*=-1.f; lq.z()*=-1.f;
    dq = this->lidarPose.q * lq.inverse();
  } else {
    dq = this->lidarPose.q * closest_pose_r.inverse();
  }

  const double theta_rad = 2.0 * std::atan2(std::sqrt(dq.x()*dq.x()+dq.y()*dq.y()+dq.z()*dq.z()), dq.w());
  const double theta_deg = theta_rad * (180.0 / M_PI);

  bool newKeyframe = false;
  if (std::abs(dd) > this->keyframe_thresh_dist_ || std::abs(theta_deg) > this->keyframe_thresh_rot_) newKeyframe = true;
  if (std::abs(dd) <= this->keyframe_thresh_dist_) newKeyframe = false;
  if (std::abs(dd) <= this->keyframe_thresh_dist_ && std::abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1) newKeyframe = true;

  const bool closest_mapping_empty =
      !this->keyframes[closest_idx].mapping_cloud ||
      this->keyframes[closest_idx].mapping_cloud->empty();
  const bool current_mapping_ready =
      this->keyframe_mapping_scan &&
      this->keyframe_mapping_scan->size() > static_cast<std::size_t>(this->gicp_min_num_points_);
  const bool same_pose_keyframe =
      std::abs(dd) <= this->keyframe_thresh_dist_ &&
      std::abs(theta_deg) <= this->keyframe_thresh_rot_;
  const bool dynamic_filter_active = this->m_detector_filter_.enabled();
  if (!newKeyframe &&
      dynamic_filter_active &&
      closest_mapping_empty &&
      current_mapping_ready &&
      !nearby_nonempty_mapping &&
      same_pose_keyframe) {
    newKeyframe = true;
  }

  if (newKeyframe) {
    if (this->keyframes.size() >= kMaxKeyframes) {
      const std::size_t removed = this->keyframes.size() - (kMaxKeyframes - 1);

      const auto removed_diff =
          static_cast<decltype(this->keyframes)::difference_type>(removed);

      this->keyframes.erase(this->keyframes.begin(), this->keyframes.begin() + removed_diff);

      this->onKeyframesTrim(removed);   // <<< keep all index-based state consistent

      if (removed >= 16) { // only when we dropped a chunk
        keyframes.shrink_to_fit();
      }
    }

    KeyframeData kf;
    kf.position = this->lidarPose.p;
    kf.orientation = this->lidarPose.q;
    kf.registration_cloud = this->registration_scan;
    kf.mapping_cloud = this->keyframe_mapping_scan;
    kf.covariances = this->gicp.getSourceCovariances();
    kf.timestamp = this->scan_header_stamp;
    kf.transform = this->T_corr;
    this->keyframes.emplace_back(std::move(kf));
  }

}

void dlio::OdomNode::onKeyframesTrim(std::size_t removed) {
  if (removed == 0) return;

  // num_processed_keyframes tracks how many keyframes have been transformed/published
  if (this->num_processed_keyframes <= removed) this->num_processed_keyframes = 0;
  else                                          this->num_processed_keyframes -= static_cast<int>(removed);

  auto shift_down = [removed](std::vector<int>& idxs) {
    const int r = static_cast<int>(removed);
    int w = 0;
    for (int i = 0; i < idxs.size(); ++i) {
      const int v = idxs[i] - r;
      if (v >= 0) idxs[w++] = v;    // keep only still-valid indices
    }
    idxs.resize(w);
  };

  shift_down(this->submap_kf_idx_prev);
  shift_down(this->submap_kf_idx_curr);
  shift_down(this->keyframe_convex);
  shift_down(this->keyframe_concave);
}

void dlio::OdomNode::setAdaptiveParams() {

  // Spaciousness
  float sp = this->metrics.spaciousness.back();

  if (sp < this->adaptive_sp_min_) { sp = this->adaptive_sp_min_; }
  if (sp > this->adaptive_sp_max_) { sp = this->adaptive_sp_max_; }

  this->keyframe_thresh_dist_ = sp;

  // Density
  float den = this->metrics.density.back();
  const float den_min = this->adaptive_den_factor_min_ * static_cast<float>(this->gicp_max_corr_dist_);
  const float den_max = this->adaptive_den_factor_max_ * static_cast<float>(this->gicp_max_corr_dist_);

  if (den < den_min) { den = den_min; }
  if (den > den_max) { den = den_max; }

  if (sp < this->adaptive_sp_max_) { den = den_min; }
  if (sp > this->adaptive_sp_max_) { den = den_max; }

  this->gicp.setMaxCorrespondenceDistance(den);

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);

}

void dlio::OdomNode::pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames) {

  // make sure dists is not empty and k is valid
  if (dists.empty() || k <= 0) { return; }

  // maintain max heap of at most k elements
  std::priority_queue<float> pq;

  for (auto d : dists) {
    if (pq.size() >= k && pq.top() > d) {
      pq.push(d);
      pq.pop();
    } else if (pq.size() < k) {
      pq.push(d);
    }
  }

  if (pq.empty()) {
    return;
  }

  // get the kth smallest element, which should be at the top of the heap
  float kth_element = pq.top();

  // get all elements smaller or equal to the kth smallest element
  for (int i = 0; i < dists.size(); ++i) {
    if (dists[i] <= kth_element)
      this->submap_kf_idx_curr.push_back(frames[i]);
  }

}

void dlio::OdomNode::buildSubmap(const State& vehicle_state) {

  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();

  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    const Eigen::Vector3f delta = vehicle_state.p - this->keyframes[i].position;
    const float d = delta.norm();
    ds.push_back(d);
    keyframe_nn.push_back(i);
  }
  lock.unlock();

  // get indices for top K nearest neighbor keyframe poses
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);

  // get convex hull indices
  this->computeConvexHull();

  // get distances for each keyframe on convex hull
  std::vector<float> convex_ds;
  std::vector<int> convex_frames;
  for (const auto& c : this->keyframe_convex) {
    if (c >= 0 && c < ds.size()) {
      convex_ds.push_back(ds[c]);
      convex_frames.push_back(c);
    }
  }

  // get indices for top kNN for convex hull
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, convex_frames);

  // get concave hull indices
  this->computeConcaveHull();

  // get distances for each keyframe on concave hull
  std::vector<float> concave_ds;
  std::vector<int> concave_frames;
  for (const auto& c : this->keyframe_concave) {
    if (c >= 0 && c < ds.size()) {
      concave_ds.push_back(ds[c]);
      concave_frames.push_back(c);
    }
  }

  // get indices for top kNN for concave hull
  this->pushSubmapIndices(concave_ds, this->submap_kcc_, concave_frames);

  // sort current and previous submap kf list of indices
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());

  // remove duplicate indices
  auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

  // check if submap has changed from previous iteration
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev){

    this->submap_hasChanged = true;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    // reinitialize submap cloud and normals
    pcl::PointCloud<PointType>::Ptr submap_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<nano_gicp::CovarianceList> submap_normals_ (std::make_shared<nano_gicp::CovarianceList>());

    {
      std::unique_lock<decltype(this->keyframes_mutex)> submap_lock(this->keyframes_mutex);
      for (auto k : this->submap_kf_idx_curr) {
        if (k < 0 || k >= static_cast<int>(this->keyframes.size()) ||
            !this->keyframes[k].registration_cloud ||
            !this->keyframes[k].covariances) {
          continue;
        }

        // create current submap cloud
        *submap_cloud_ += *this->keyframes[k].registration_cloud;

        // grab corresponding submap cloud's normals
        submap_normals_->insert( std::end(*submap_normals_),
            std::begin(*(this->keyframes[k].covariances)), std::end(*(this->keyframes[k].covariances)) );
      }
    }

    this->submap_cloud = submap_cloud_;
    this->submap_normals = submap_normals_;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    this->gicp_temp.setInputTarget(this->submap_cloud);
    this->submap_kdtree = this->gicp_temp.target_kdtree_;

    this->submap_kf_idx_prev = this->submap_kf_idx_curr;
  }
}

void dlio::OdomNode::buildKeyframesAndSubmap(const State& vehicle_state) {

  // transform the new keyframe(s) and associated covariance list(s)
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++) {
    pcl::PointCloud<PointType>::ConstPtr raw_registration_keyframe = this->keyframes[i].registration_cloud;
    pcl::PointCloud<PointType>::ConstPtr raw_mapping_keyframe = this->keyframes[i].mapping_cloud;
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframes[i].covariances;
    Eigen::Matrix4f T = this->keyframes[i].transform;

    if (!raw_registration_keyframe || !raw_mapping_keyframe || !raw_covariances) {
      continue;
    }

    Eigen::Matrix4d Td = T.cast<double>();

    pcl::PointCloud<PointType>::Ptr transformed_registration_keyframe = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*raw_registration_keyframe, *transformed_registration_keyframe, T);

    pcl::PointCloud<PointType>::Ptr transformed_mapping_keyframe = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*raw_mapping_keyframe, *transformed_mapping_keyframe, T);

    std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances (std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));
    std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
                   [&Td](const Eigen::Matrix4d& cov) { return Td * cov * Td.transpose(); });

    ++this->num_processed_keyframes;

    this->keyframes[i].registration_cloud = transformed_registration_keyframe;
    this->keyframes[i].mapping_cloud = transformed_mapping_keyframe;
    this->keyframes[i].covariances = transformed_covariances;

    this->publishKeyframe(this->keyframes[i]);
  }

  lock.unlock();

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();

  this->buildSubmap(vehicle_state);
}

void dlio::OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]{ return !this->main_loop_running || this->shouldStop(); });
}

void dlio::OdomNode::debug() {

  // Only one debug() call may run at a time: concurrent calls race on cpu_percents
  // and the CPU usage sampling state (lastCPU/lastSysCPU/lastUserCPU).
  std::unique_lock<std::mutex> debug_lock(this->mtx_debug_, std::try_to_lock);
  if (!debug_lock.owns_lock()) {
    return;
  }

  // length_traversed is already maintained incrementally in processPointCloud().
  const double length_traversed = this->length_traversed;

  // Snapshot comp_times under lock to avoid racing with processPointCloud().
  std::vector<double> comp_snapshot;
  {
    std::lock_guard<std::mutex> lk(this->mtx_comp_times_);
    comp_snapshot.reserve(this->comp_times.size());
    for (const auto& [ts, ct] : this->comp_times) {
      comp_snapshot.push_back(ct);
    }
  }

  const double avg_comp_time = comp_snapshot.empty() ? 0.0 :
    std::accumulate(comp_snapshot.begin(), comp_snapshot.end(), 0.0) /
        static_cast<double>(comp_snapshot.size());
  const double max_comp_time = comp_snapshot.empty() ? 0.0 :
    *std::max_element(comp_snapshot.begin(), comp_snapshot.end());
  const double last_comp_time = comp_snapshot.empty() ? 0.0 : comp_snapshot.back();

  // RAM Usage
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  const long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  resident_set = static_cast<double>(rss) * static_cast<double>(page_size_kb);

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = static_cast<double>(timeSample.tms_stime - this->lastSysCPU) +
                    static_cast<double>(timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= static_cast<double>(now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  this->cpu_percents.push_back(cpu_percent);
  if (this->cpu_percents.size() > 400) {
    this->cpu_percents.pop_front();
  }
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) /
        static_cast<double>(this->cpu_percents.size());

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout << '\n'
            << "+-------------------------------------------------------------------+\n";
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << '\n';
  std::cout << "+-------------------------------------------------------------------+\n";

  const std::time_t curr_time = static_cast<std::time_t>(this->scan_stamp);
  std::tm tm_info{};
  localtime_r(&curr_time, &tm_info);
  std::array<char, 32> time_buf{};
  std::strftime(time_buf.data(), time_buf.size(), "%a %b %d %H:%M:%S %Y", &tm_info);
  const std::string asc_time(time_buf.data());
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
    << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
    << "|" << std::endl;

  if ( !this->cpu_type.empty() ) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << this->cpu_type + " x " + std::to_string(this->numProcessors)
      << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Position     {W}  [xyz] :: " + to_string_with_precision(this->state.p[0], 4) + " "
                                + to_string_with_precision(this->state.p[1], 4) + " "
                                + to_string_with_precision(this->state.p[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Orientation  {W} [wxyz] :: " + to_string_with_precision(this->state.q.w(), 4) + " "
                                + to_string_with_precision(this->state.q.x(), 4) + " "
                                + to_string_with_precision(this->state.q.y(), 4) + " "
                                + to_string_with_precision(this->state.q.z(), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.lin.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.ang.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Accel Bias        [xyz] :: " + to_string_with_precision(this->state.b.accel[0], 8) + " "
                                + to_string_with_precision(this->state.b.accel[1], 8) + " "
                                + to_string_with_precision(this->state.b.accel[2], 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gyro Bias         [xyz] :: " + to_string_with_precision(this->state.b.gyro[0], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[1], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[2], 8)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance Traveled  :: " + to_string_with_precision(length_traversed, 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance to Origin :: "
      + to_string_with_precision( sqrt(pow(this->state.p[0]-this->origin[0],2) +
                                       pow(this->state.p[1]-this->origin[1],2) +
                                       pow(this->state.p[2]-this->origin[2],2)), 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Registration       :: keyframes: " + std::to_string(this->keyframes.size()) + ", "
                               + "deskewed points: " + std::to_string(this->deskew_size)
    << "|" << std::endl;
  std::cout << "|                                                                   |" << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
    << std::setfill(' ') << std::setw(6) << last_comp_time*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << max_comp_time*1000.
    << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: "
    << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * this->numProcessors << " cores // Avg: "
    << std::setw(6) << (avg_cpu_usage/100.) * this->numProcessors << " / Max: "
    << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.)
                       * this->numProcessors
    << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
    << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
    << std::setw(6) << avg_cpu_usage << " / Max: "
    << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
    << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
    << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}
