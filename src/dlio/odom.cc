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
#include "dlio/utils.h"

#include <iomanip>
#include <queue>
#include <algorithm>

#include "rclcpp/qos.hpp"

namespace {

template <typename PublisherPtrT>
inline bool hasSubscribers(const PublisherPtrT& pub) {
  return pub && pub->get_subscription_count() > 0;
}

inline void logTranslationSpectrumAlways(const rclcpp::Logger& logger,
                                         const Eigen::Vector3d& evals,
                                         const Eigen::Matrix3d& evecs) {
  RCLCPP_INFO(
      logger,
      "Translation Hessian spectrum: eigvals(sorted asc) = [%.0f %.0f %.0f]",
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

}  // namespace

dlio::OdomNode::OdomNode() : Node("dlio_odom_node") {

  this->getParams();

  this->num_threads_ = omp_get_max_threads();

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

  this->odom_pub     = this->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  this->pose_pub     = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 1);
  this->path_pub     = this->create_publisher<nav_msgs::msg::Path>("path_map", 1);
  this->path_odom_pub = this->create_publisher<nav_msgs::msg::Path>("path_odom", 1);
  this->path_map_prop_pub = this->create_publisher<nav_msgs::msg::Path>("path_map_prop", 1);
  this->kf_pose_pub  = this->create_publisher<geometry_msgs::msg::PoseArray>("kf_pose", 1);

  rclcpp::QoS reliable_qos(rclcpp::KeepLast(10));
  reliable_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  this->odom_map_pub = this->create_publisher<nav_msgs::msg::Odometry>("map_pose", reliable_qos);

  auto best_effort_qos = rclcpp::QoS(100)
                             .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                             .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE)
                             .history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);

  this->kf_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("kf_cloud", best_effort_qos);
  this->deskewed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed", best_effort_qos);
  this->deskewed_not_transformed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed_not_transformed", best_effort_qos);
  this->deskewed_map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed_and_transformed_to_map", best_effort_qos);

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
  this->corr_marker_points_.reserve(static_cast<size_t>(2 * std::max(1, this->viz_corr_max_segments_)));

  this->pointcloud_worker_ = std::thread([this]{ pointCloudWorkerLoop(); });
  this->pub_worker_ = std::thread([this]{ workerLoop(); });

  {
    std::lock_guard<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    this->keyframes.reserve(this->kMaxKeyframes);
    this->keyframe_timestamps.reserve(this->kMaxKeyframes);
    this->keyframe_normals.reserve(this->kMaxKeyframes);
    this->keyframe_transformations.reserve(this->kMaxKeyframes);
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

  this->gicp_temp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_temp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_temp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_temp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_temp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_temp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(temp, true);
  this->gicp.setSearchMethodTarget(temp, true);
  this->gicp_temp.setSearchMethodSource(temp, true);
  this->gicp_temp.setSearchMethodTarget(temp, true);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);

  {
    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    this->metrics.spaciousness.push_back(0.);
    this->metrics.density.push_back(this->gicp_max_corr_dist_);
    // Start with a neutral model-deviation scale so adaptive gating
    // has a stable value before the first completed registration.
    this->metrics.motion_deviation.push_back(this->gicp_max_corr_dist_);
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
    "Translation Hessian degeneracy detected. eigvals(sorted asc) = [%.0f %.0f %.0f]",
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

  stop_.store(true, std::memory_order_relaxed);
  pc_q_cv_.notify_all();
  q_cv_.notify_all();
  cv_imu_stamp.notify_all();
  submap_build_cv.notify_all();
  if (pointcloud_worker_.joinable()) pointcloud_worker_.join();
  if (pub_worker_.joinable()) pub_worker_.join();

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
  job.cloud = cloud;
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
  while (!stop_.load(std::memory_order_relaxed)) {
    PubJob job;
    {
      std::unique_lock<std::mutex> lk(q_mtx_);
      q_cv_.wait(lk, [this]{ return stop_.load() || !q_.empty(); });
      if (stop_.load()) break;
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

void dlio::OdomNode::pointCloudWorkerLoop() {
  auto required_imu_time_from_cloud =
      [](const sensor_msgs::msg::PointCloud2::SharedPtr& pc) -> double {
        if (!pc) {
          return 0.0;
        }

        const double header_time = rclcpp::Time(pc->header.stamp).seconds();

        bool has_timestamp = false;
        bool has_t = false;
        bool has_time = false;
        for (const auto& field : pc->fields) {
          if (field.name == "timestamp") {
            has_timestamp = true;
          } else if (field.name == "t") {
            has_t = true;
          } else if (field.name == "time") {
            has_time = true;
          }
        }

        // No per-point timing field: fall back to header stamp.
        if (!has_timestamp && !has_t && !has_time) {
          return header_time;
        }

        pcl::PointCloud<PointType> cloud;
        pcl::fromROSMsg(*pc, cloud);
        if (cloud.empty()) {
          return header_time;
        }

        // RoboSense / Hesai / Livox-style timestamp field
        if (has_timestamp) {
          bool have_first = false;
          bool use_ns = false;

          for (const auto& pt : cloud.points) {
            const double t = pt.timestamp;
            if (!std::isfinite(t)) {
              continue;
            }
            use_ns = (t > 1e14);  // same convention already used in your sensor detection
            have_first = true;
            break;
          }

          if (!have_first) {
            return header_time;
          }

          double scan_end = -std::numeric_limits<double>::infinity();
          for (const auto& pt : cloud.points) {
            double t = pt.timestamp;
            if (!std::isfinite(t)) {
              continue;
            }
            if (use_ns) {
              t *= 1e-9;
            }
            scan_end = std::max(scan_end, t);
          }

          return std::isfinite(scan_end) ? scan_end : header_time;
        }

        // Ouster-style relative nanoseconds
        if (has_t) {
          double max_rel_ns = 0.0;
          for (const auto& pt : cloud.points) {
            const double t = pt.t;
            if (!std::isfinite(t)) {
              continue;
            }
            max_rel_ns = std::max(max_rel_ns, t);
          }
          return header_time + 1e-9 * max_rel_ns;
        }

        // Velodyne-style relative seconds
        double max_rel_s = 0.0;
        for (const auto& pt : cloud.points) {
          const double t = pt.time;
          if (!std::isfinite(t)) {
            continue;
          }
          max_rel_s = std::max(max_rel_s, t);
        }
        return header_time + max_rel_s;
      };

  while (!stop_.load(std::memory_order_relaxed)) {
    PointCloudJob job;
    {
      std::unique_lock<std::mutex> lk(pc_q_mtx_);
      pc_q_cv_.wait(lk, [this]{
        return stop_.load(std::memory_order_relaxed) || !pc_q_.empty();
      });
      if (stop_.load(std::memory_order_relaxed)) {
        break;
      }

      job = std::move(pc_q_.front());
      pc_q_.pop_front();
    }

    // Wait here, before any pointcloud processing begins.
    const double required_imu_time = required_imu_time_from_cloud(job.cloud_msg);

    {
      std::unique_lock<decltype(this->mtx_imu)> imu_lock(this->mtx_imu);
      this->cv_imu_stamp.wait(imu_lock, [this, required_imu_time]{
        return this->stop_.load(std::memory_order_relaxed) ||
               (!this->imu_buffer.empty() &&
                this->imu_buffer.front().stamp >= required_imu_time);
      });
    }

    if (stop_.load(std::memory_order_relaxed)) {
      break;
    }

    this->processPointCloud(job.cloud_msg);
  }
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

  // Extrinsics
  std::vector<double> t_default{0., 0., 0.};
  std::vector<double> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<double> baselink2imu_t, baselink2imu_R;
  dlio::declare_param(this, "extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
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
    Eigen::Vector3f(baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
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
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
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

  // Translation-only degeneracy analysis:
  // Weak directions are flagged by absolute eigenvalue thresholds on H_tt.
  dlio::declare_param(this, "odom/gicp/degeneracy/hessian_in_base_frame",
                      this->degen_hessian_in_base_frame_, false);
  dlio::declare_param(this, "odom/gicp/degeneracy/trans_eig_abs_threshold",
                      this->degen_trans_eig_abs_thresh_, 200.0);

  dlio::declare_param(this, "viz/degeneracy_marker/enabled", this->viz_degen_marker_, true);
  dlio::declare_param(this, "viz/degeneracy_marker/trans_scale", this->viz_degen_trans_scale_, 0.75);
  dlio::declare_param(this, "viz/degeneracy_marker/shaft_diameter", this->viz_degen_shaft_diam_, 0.03);
  dlio::declare_param(this, "viz/degeneracy_marker/head_diameter", this->viz_degen_head_diam_, 0.06);
  dlio::declare_param(this, "viz/degeneracy_marker/head_length", this->viz_degen_head_len_, 0.10);
  dlio::declare_param(this, "viz/degeneracy_marker/lifetime", this->viz_degen_lifetime_, 0.0);

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
void dlio::OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void dlio::OdomNode::publishToROS(
    pcl::PointCloud<PointType>::ConstPtr cloud,
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

  if (hasSubscribers(this->odom_map_pub)) {
    this->odom_map_pub->publish(odom_map);
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

  this->publishCorrectionMarker(scan_time, T_cloud, T_all);

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
    pcl::PointCloud<PointType>::ConstPtr cloud,
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
    this->deskewed_pub->publish(std::move(msg));
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
    this->deskewed_not_transformed_pub->publish(std::move(msg));
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
    this->deskewed_map_pub->publish(std::move(msg));
  }
}

void dlio::OdomNode::publishKeyframe(
    std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf,
    rclcpp::Time timestamp) {

  // Push back
  geometry_msgs::msg::Pose p;
  p.position.x = kf.first.first[0];
  p.position.y = kf.first.first[1];
  p.position.z = kf.first.first[2];
  p.orientation.w = kf.first.second.w();
  p.orientation.x = kf.first.second.x();
  p.orientation.y = kf.first.second.y();
  p.orientation.z = kf.first.second.z();
  this->kf_pose_ros.poses.push_back(p);

  // Trim PoseArray to avoid unbounded RViz payload
  if (this->kf_pose_ros.poses.size() > 30) {
    this->kf_pose_ros.poses.erase(
      this->kf_pose_ros.poses.begin(),
      this->kf_pose_ros.poses.begin() + (this->kf_pose_ros.poses.size() - 30)
    );
  }

  // Keyframes are stored/published after being transformed into the map frame.
  this->kf_pose_ros.header.stamp = timestamp;
  this->kf_pose_ros.header.frame_id = "dlio_map";
  if (hasSubscribers(this->kf_pose_pub)) {
    this->kf_pose_pub->publish(this->kf_pose_ros);
  }

  if (hasSubscribers(this->kf_cloud_pub)) {
    auto publish_kf_cloud = [&]() {
      sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = timestamp;
      keyframe_cloud_ros.header.frame_id = "dlio_map";
      this->kf_cloud_pub->publish(keyframe_cloud_ros);
    };
    if (this->vf_use_) {
      if (kf.second->points.size() == kf.second->width * kf.second->height) {
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
      } else if (field.name == "timestamp" && original_scan_->points[0].timestamp < 1e14) {
        this->sensor = dlio::SensorType::HESAI;
        // ROBOSENSE IS ALSO HERE
        break;
      } else if (field.name == "timestamp" && original_scan_->points[0].timestamp > 1e14) {
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
    this->voxel.setInputCloud(current_scan_);
    this->voxel.filter(*current_scan_);
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

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
      { return sweep_ref_time + pt.value().t * 1e-9f; };

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
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp * 1e-9f; };
  
    } else {
    this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>(*this->original_scan);
    this->deskew_status = false;
    this->deskew_size = 0;
    return;
  }

  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

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
    unique_time_indices.push_back(it->index());
  }

  if (timestamps.empty()) {
    this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>(*this->original_scan);
    this->deskew_status = false;
    this->deskew_size = 0;
    return;
  }

  unique_time_indices.push_back(deskewed_scan_->points.size());

  const double first_point_time_raw = timestamps.front() - offset;
  const double last_point_time_raw  = timestamps.back()  - offset;

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

  // don't process scans until IMU data is present
if (!this->first_valid_scan) {
  bool imu_ready = false;
  {
    std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);
    imu_ready = !this->imu_buffer.empty() &&
                this->imu_buffer.front().stamp >= timestamps.back();
  }

  if (!imu_ready) {
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
  this->deskew_size = frames.size(); // if integration successful, equal to timestamps.size()

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
  this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  this->keyframe_timestamps.push_back(this->scan_header_stamp);
  this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
  this->keyframe_transformations.push_back(this->T_corr);

}

void dlio::OdomNode::setInputSource() {
  // Source = current deskewed/filtered scan in world frame.
  // NanoGICP builds a source k-d tree and source covariances from this cloud.
  this->gicp.setInputSource(this->current_scan);
  this->gicp.calculateSourceCovariances();
}

void dlio::OdomNode::initializeDLIO() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }

  this->dlio_initialized = true;
  std::cout << std::endl << " DLIO initialized!" << std::endl;

}

void dlio::OdomNode::callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc) {
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
    this->initializeInputTarget();
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_future =
      std::async(std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state);
    this->submap_future.wait(); // wait until completion
    return;
  }

  // Get the next pose via IMU + S2M + GEO
  this->getNextPose();

  // Capture a scan-time odom snapshot immediately after the LiDAR update.
  // This snapshot must travel with the scan so that map<->odom for the published cloud
  // is computed from the same timestamped state as T_all/T_cloud.
  Eigen::Vector3f state_p_scan, state_vlin_b_scan, state_vang_b_scan;
  Eigen::Quaternionf state_q_scan;
  {
    std::lock_guard<std::mutex> geo_lock(this->geo.mtx);
    state_p_scan = this->state.p;
    state_q_scan = this->state.q.normalized();
    state_vlin_b_scan = this->state.v.lin.b;
    state_vang_b_scan = this->state.v.ang.b;
  }
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
    this->trajectory.erase(this->trajectory.begin(),
                           this->trajectory.begin() + (this->trajectory.size() - 1600));
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

  this->enqueuePublish(published_cloud,
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
  this->debug_thread = std::thread(&dlio::OdomNode::debug, this);
  this->debug_thread.detach();

  this->geo.first_opt_done = true;
}

void dlio::OdomNode::callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu_raw) {

  this->first_imu_received = true;

  sensor_msgs::msg::Imu::SharedPtr imu = this->transformImu( imu_raw );
  this->imu_stamp = imu->header.stamp;
  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();

  Eigen::Vector3f lin_accel = Eigen::Vector3f::Zero();
  Eigen::Vector3f ang_vel = Eigen::Vector3f::Zero();

  // Get IMU samples
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  lin_accel[0] = imu->linear_acceleration.x;
  lin_accel[1] = imu->linear_acceleration.y;
  lin_accel[2] = imu->linear_acceleration.z;

  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu_stamp_secs;
  }

  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated) {

    static int num_samples = 0;
    static Eigen::Vector3f gyro_avg (0., 0., 0.);
    static Eigen::Vector3f accel_avg (0., 0., 0.);
    static bool print = true;

    if ((imu_stamp_secs - this->first_imu_stamp) < this->imu_calib_time_) {

      num_samples++;

      gyro_avg[0] += ang_vel[0];
      gyro_avg[1] += ang_vel[1];
      gyro_avg[2] += ang_vel[2];

      accel_avg[0] += lin_accel[0];
      accel_avg[1] += lin_accel[1];
      accel_avg[2] += lin_accel[2];

      if(print) {
        std::cout << std::endl << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        print = false;
      }

    } else {

      std::cout << "done" << std::endl << std::endl;

      gyro_avg /= num_samples;
      accel_avg /= num_samples;

      Eigen::Vector3f grav_vec (0., 0., this->gravity_);

      if (this->gravity_align_) {

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - this->state.b.accel).normalized() * abs(this->gravity_);
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(grav_vec, Eigen::Vector3f(0., 0., this->gravity_));

        // set gravity aligned orientation
        this->state.q = grav_q;
        this->T.block(0,0,3,3) = this->state.q.toRotationMatrix();
        this->lidarPose.q = this->state.q;

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
        std::cout << " Estimated initial attitude:" << std::endl;
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << std::endl;
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << std::endl;
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << std::endl;
        std::cout << std::endl;
      }

      if (this->calibrate_accel_) {

        // subtract gravity from avg accel to get bias
        this->state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: " << to_string_with_precision(this->state.b.accel[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[2], 8) << std::endl;
      }

      if (this->calibrate_gyro_) {

        this->state.b.gyro = gyro_avg;

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(this->state.b.gyro[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[2], 8) << std::endl;
      }

      this->imu_calibrated = true;
      this->prev_imu_stamp = rclcpp::Time(imu->header.stamp).seconds();

    }

  } else {

    double dt = imu_stamp_secs - this->prev_imu_stamp;
    if (dt <= 0) { dt = 1.0/400.0; }
    // this->imu_rates.push_back( 1./dt );

    // Apply the calibrated bias to the new IMU measurements
    this->imu_meas.stamp = imu_stamp_secs;
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f lin_accel_corrected = (this->imu_accel_sm_ * lin_accel) - this->state.b.accel;
    Eigen::Vector3f ang_vel_corrected = ang_vel - this->state.b.gyro;

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
  // Snapshot under the same mutex used in propagate/update
  Eigen::Vector3f p, vlin_b, vang_b;
  Eigen::Quaternionf q;
  rclcpp::Time stamp;

  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    p      = this->state.p;
    q      = this->state.q;
    vlin_b = this->state.v.lin.b;
    vang_b = this->state.v.ang.b;
    stamp  = this->imu_stamp;
  }

  q.normalize();

  // state pose is dlio_odom -> base_link. Invert it to publish base_link -> dlio_odom.
  const Eigen::Quaternionf q_bo = q.conjugate();
  const Eigen::Vector3f p_bo = -(q_bo._transformVector(p));

  // Exact inverse twist for nav_msgs/Odometry semantics:
  // twist is child wrt parent, expressed in child frame.
  const Eigen::Vector3f v_ob_o = q._transformVector(vlin_b);
  const Eigen::Vector3f w_ob_o = q._transformVector(vang_b);
  const Eigen::Vector3f v_bo_o = -(v_ob_o + p.cross(w_ob_o));
  const Eigen::Vector3f w_bo_o = -w_ob_o;

  // Build and publish Odometry
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = this->baselink_frame;
  odom.child_frame_id  = this->odom_frame;

  odom.pose.pose.position.x = p_bo.x();
  odom.pose.pose.position.y = p_bo.y();
  odom.pose.pose.position.z = p_bo.z();
  odom.pose.pose.orientation.w = q_bo.w();
  odom.pose.pose.orientation.x = q_bo.x();
  odom.pose.pose.orientation.y = q_bo.y();
  odom.pose.pose.orientation.z = q_bo.z();

  odom.twist.twist.linear.x  = v_bo_o.x();
  odom.twist.twist.linear.y  = v_bo_o.y();
  odom.twist.twist.linear.z  = v_bo_o.z();
  odom.twist.twist.angular.x = w_bo_o.x();
  odom.twist.twist.angular.y = w_bo_o.y();
  odom.twist.twist.angular.z = w_bo_o.z();

  if (hasSubscribers(this->odom_pub)) {
    this->odom_pub->publish(odom);
  }

  // Build and publish PoseStamped
  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = this->baselink_frame;
  pose.pose.position.x = p_bo.x();
  pose.pose.position.y = p_bo.y();
  pose.pose.position.z = p_bo.z();
  pose.pose.orientation.w = q_bo.w();
  pose.pose.orientation.x = q_bo.x();
  pose.pose.orientation.y = q_bo.y();
  pose.pose.orientation.z = q_bo.z();

  if (hasSubscribers(this->pose_pub)) {
    this->pose_pub->publish(pose);
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

  // TF: base_link -> dlio_odom
  geometry_msgs::msg::TransformStamped tf_bl_odom;
  tf_bl_odom.header.stamp = stamp;
  tf_bl_odom.header.frame_id = this->baselink_frame;
  tf_bl_odom.child_frame_id  = this->odom_frame;
  tf_bl_odom.transform.translation.x = p_bo.x();
  tf_bl_odom.transform.translation.y = p_bo.y();
  tf_bl_odom.transform.translation.z = p_bo.z();
  tf_bl_odom.transform.rotation.w = q_bo.w();
  tf_bl_odom.transform.rotation.x = q_bo.x();
  tf_bl_odom.transform.rotation.y = q_bo.y();
  tf_bl_odom.transform.rotation.z = q_bo.z();
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

void dlio::OdomNode::analyzeDegeneracyFromMatrix(
    const Eigen::Ref<const Eigen::Matrix<double, 6, 6>>& H_in,
    const Eigen::Ref<const Eigen::Matrix4f>& T_map_base) {

  if (!H_in.allFinite()) {
    this->degen_info_.valid = false;
    return;
  }

  const Eigen::Matrix<double, 6, 6> H_raw =
      0.5 * (H_in + H_in.transpose());

  this->degen_info_.p_map_base = T_map_base.block<3,1>(0,3).cast<double>();

  if (this->degen_hessian_in_base_frame_) {
    RCLCPP_WARN_ONCE(
        this->get_logger(),
        "odom/gicp/degeneracy/hessian_in_base_frame is ignored here. "
        "Current Hessian is treated as expressed in the registration/world frame.");
  }

  constexpr double eps = 1e-12;
  const Eigen::Matrix3d H_tt =
      0.5 * (H_raw.block<3,3>(3,3) + H_raw.block<3,3>(3,3).transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_trans(H_tt);
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

  const double trans_min = std::max(this->degen_info_.eigvals_trans_dec(0), eps);
  const double trans_max = std::max(this->degen_info_.eigvals_trans_dec(2), eps);

  this->degen_info_.trans_condition = trans_max / trans_min;
}

void dlio::OdomNode::analyzeDegeneracyFromHessian(
    const Eigen::Ref<const Eigen::Matrix4f>& T_map_reg) {

  const Eigen::Matrix<double, 6, 6> H_raw_unsym = this->gicp.getFinalHessian();
  if (!H_raw_unsym.allFinite()) {
    this->degen_info_.valid = false;
    return;
  }

  const Eigen::Matrix<double, 6, 6> H_raw =
      0.5 * (H_raw_unsym + H_raw_unsym.transpose());

  // Marker origin should be base pose in map.
  this->degen_info_.p_map_base = T_map_reg.block<3,1>(0,3).cast<double>();

  if (this->degen_hessian_in_base_frame_) {
    RCLCPP_WARN_ONCE(
        this->get_logger(),
        "odom/gicp/degeneracy/hessian_in_base_frame is ignored here. "
        "Current NanoGICP final Hessian is treated as already expressed in the "
        "registration/world frame.");
  }

  constexpr double eps = 1e-12;
  const Eigen::Matrix3d H_tt =
      0.5 * (H_raw.block<3,3>(3,3) + H_raw.block<3,3>(3,3).transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_trans_raw(H_tt);

  if (eig_trans_raw.info() != Eigen::Success) {
    this->degen_info_.valid = false;
    return;
  }

  Eigen::Vector3d trans_evals_sorted = Eigen::Vector3d::Zero();
  Eigen::Matrix3d trans_evecs_sorted = Eigen::Matrix3d::Identity();

  sortEigenpairsAscending(
      eig_trans_raw.eigenvalues(), eig_trans_raw.eigenvectors(),
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

  const double trans_min = std::max(this->degen_info_.eigvals_trans_dec(0), eps);
  const double trans_max = std::max(this->degen_info_.eigvals_trans_dec(2), eps);

  this->degen_info_.trans_condition = trans_max / trans_min;
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

  auto pushArrow = [&](const int id,
                       const std::string& ns,
                       const Eigen::Vector3d& dir_map,
                       const double length,
                       const float r,
                       const float g,
                       const float b) {
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
    m.color.r = r;
    m.color.g = g;
    m.color.b = b;

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
                1.00f,
                is_min_mode ? 0.10f : 0.55f,
                0.10f);
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
    this->corr_marker_points_.erase(
      this->corr_marker_points_.begin(),
      this->corr_marker_points_.begin() + (this->corr_marker_points_.size() - max_points));
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

  visualization_msgs::msg::Marker m_lin, m_ang;
  
  createLinVelocityMarker(this->baselink_frame, stamp, vlin_b, m_lin);
  createAngularVelocityMarker(this->baselink_frame, stamp, vang_b, m_ang);

  if (hasSubscribers(this->pub_lin_vel_marker_)) {
    this->pub_lin_vel_marker_->publish(m_lin);
  }
  if (hasSubscribers(this->pub_ang_vel_marker_)) {
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
    marker.color.a = std::min(angularMagnitude * 0.5 + 0.3, 1.0);
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

void dlio::OdomNode::getNextPose() {

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

  // Construct and analyze the pre-optimization Hessian at the initial correction guess.
  // The current scan has already been transformed into the map with T_prior, so the
  // optimizer's initial correction guess is identity.
  const Eigen::Matrix4f T_corr_guess = Eigen::Matrix4f::Identity();
  Eigen::Matrix<double, 6, 6> H0;
  Eigen::Matrix<double, 6, 1> b0;
  double y0 = 0.0;

  const bool have_pre_hessian =
      this->gicp.computeInitialHessianAtGuess(T_corr_guess, H0, b0, y0);

  bool degeneracy_detected = false;

  if (have_pre_hessian) {
    // Degeneracy eigendirections are computed from the initial registration Hessian.
    // Use the prior pose because this Hessian is built before optimization, at the
    // identity correction on top of the already prior-transformed source cloud.
    this->analyzeDegeneracyFromMatrix(H0, this->T_prior);

    // Print the translation eigensystem for every scan.
    // if (this->degen_info_.valid) {
    //   logTranslationSpectrumAlways(
    //       this->get_logger(),
    //       this->degen_info_.eigvals_trans_dec,
    //       this->degen_info_.eigvecs_trans_map);
    // }

    degeneracy_detected =
        this->degen_info_.valid && hasWeakDirection(this->degen_info_.weak_trans);

    // if (degeneracy_detected) {
    //   logTranslationDegeneracy(
    //       this->get_logger(),
    //       this->degen_info_.eigvals_trans_dec,
    //       this->degen_info_.eigvecs_trans_map,
    //       this->degen_info_.weak_trans);
    // }

    this->publishDegeneracyMarkers(this->scan_header_stamp);
  } else {
    this->degen_info_.valid = false;
    this->publishDegeneracyMarkers(this->scan_header_stamp);
  }

  if (degeneracy_detected) {
    // Skip scan-to-submap registration when degeneracy is detected before optimization.
    // Keep the IMU prior as the global pose update.
    this->gicp_hasConverged = false;
    this->T_corr = Eigen::Matrix4f::Identity(); // no registration correction
    this->T = this->T_prior;
  } else {
    // Run scan-to-submap registration with the initial correction guess.
    // Call chain:
    //   gicp.align()
    //     -> NanoGICP::computeTransformation()
    //     -> LsqRegistration::computeTransformation()
    //     -> step_optimize() (LM by default)
    pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
    this->gicp.align(*aligned, T_corr_guess);

    // Correction from registration (source -> target, both already in world frame).
    this->T_corr = this->gicp.getFinalTransformation(); // "correction" transformation
    this->T = this->T_corr * this->T_prior;
    this->gicp_hasConverged = this->gicp.hasConverged();
  }

  // this->computeMotionDeviation();

  // Update next global pose
  // Both source and target clouds are in the global frame now, so transformation is global
  this->propagateGICP();

  if (!degeneracy_detected) {
    // Geometric observer update using accepted LiDAR registration result
    this->updateState();
  } else {
    // No LiDAR update was accepted on this scan.
    // Keep the IMU-propagated observer state unchanged, but refresh bookkeeping
    // used by the next IMU-prior construction.
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    this->geo.prev_p = this->state.p;
    this->geo.prev_q = this->state.q;
    this->geo.prev_vel = this->state.v.lin.w;
  }
}

bool dlio::OdomNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
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

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // invalid input, return empty vector
    return empty;
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) == false) {
    // not enough IMU measurements, return empty vector
    return empty;
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = *begin_imu_it;
  const ImuMeas& f2 = *(begin_imu_it+1);

  // Time between first two IMU samples
  double dt = f2.dt;
  if (dt <= 0.0) {
    return empty;
  }

  // Time between first IMU sample and start_time
  double idt = start_time - f1.stamp;

  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5*alpha*idt);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
    q_init.w() - 0.5*( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idt,
    q_init.x() + 0.5*( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idt,
    q_init.y() + 0.5*( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idt,
    q_init.z() + 0.5*( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idt
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = f1.ang_vel + 0.5*alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5*( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dt,
    q_init.x() + 0.5*( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dt,
    q_init.y() + 0.5*( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dt,
    q_init.z() + 0.5*( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dt
  );
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= this->gravity_;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dt;

  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1*idt + 0.5*j*idt*idt;

  // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init*idt + 0.5*a1*idt*idt + (1/6.)*j*idt*idt*idt;

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, begin_imu_it, end_imu_it);
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                                     const std::vector<double>& sorted_timestamps,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;

  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel);
  a[2] -= this->gravity_;

  // Iterate over IMU measurements and timestamps
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it = sorted_timestamps.begin();

  for (; imu_it != end_imu_it; imu_it++) {

    const ImuMeas& f0 = *prev_imu_it;
    const ImuMeas& f = *imu_it;

    // Time between IMU samples
    double dt = f.dt;
    if (dt <= 0.0) {
      prev_imu_it = imu_it;
      continue;
    }

    // Angular acceleration
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dt;

    // Average angular velocity
    Eigen::Vector3f omega = f0.ang_vel + 0.5*alpha_dt;

    const Eigen::Quaternionf q0 = q;

    // Orientation at current IMU sample
    q = Eigen::Quaternionf (
      q0.w() - 0.5*( q0.x()*omega[0] + q0.y()*omega[1] + q0.z()*omega[2] ) * dt,
      q0.x() + 0.5*( q0.w()*omega[0] - q0.z()*omega[1] + q0.y()*omega[2] ) * dt,
      q0.y() + 0.5*( q0.z()*omega[0] + q0.w()*omega[1] - q0.x()*omega[2] ) * dt,
      q0.z() + 0.5*( q0.x()*omega[1] - q0.y()*omega[0] + q0.w()*omega[2] ) * dt
    );
    q.normalize();

    // Acceleration
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= this->gravity_;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // Time between previous IMU sample and given timestamp
      double idt = *stamp_it - f0.stamp;

      // Average angular velocity
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5*alpha*idt;

      // Orientation at interpolated timestamp (must start from q0, not q)
      Eigen::Quaternionf q_i (
        q0.w() - 0.5*( q0.x()*omega_i[0] + q0.y()*omega_i[1] + q0.z()*omega_i[2] ) * idt,
        q0.x() + 0.5*( q0.w()*omega_i[0] - q0.z()*omega_i[1] + q0.y()*omega_i[2] ) * idt,
        q0.y() + 0.5*( q0.z()*omega_i[0] + q0.w()*omega_i[1] - q0.x()*omega_i[2] ) * idt,
        q0.z() + 0.5*( q0.x()*omega_i[1] - q0.y()*omega_i[0] + q0.w()*omega_i[2] ) * idt
      );
      q_i.normalize();

      // Position
      Eigen::Vector3f p_i = p + v*idt + 0.5*a0*idt*idt + (1/6.)*j*idt*idt*idt;

      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      ++stamp_it;
    }

    // Position
    p += v*dt + 0.5*a0*dt*dt + (1/6.)*j_dt*dt*dt;

    // Velocity
    v += a0*dt + 0.5*j_dt*dt;

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
  double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

void dlio::OdomNode::propagateState() {
  std::lock_guard<std::mutex> lock(this->geo.mtx);

  const double dt = this->imu_meas.dt;
  if (dt <= 0) return;

  // Body specific force (bias already removed in callbackImu)
  const Eigen::Vector3f f_b = this->imu_meas.lin_accel;

  // Rotation and gravity (world frame)
  const Eigen::Matrix3f Rwb = this->state.q.toRotationMatrix();
  const Eigen::Vector3f g_w(0.f, 0.f, this->gravity_);   // or a param vector

  // World-frame acceleration
  const Eigen::Vector3f a_w = Rwb * f_b - g_w;

  // Integrate p, v (world frame)
  this->state.p      += this->state.v.lin.w * dt + 0.5f * a_w * static_cast<float> (dt * dt);
  this->state.v.lin.w += a_w * static_cast<float> (dt);
  this->state.v.lin.b  = Rwb.transpose() * this->state.v.lin.w;

  // Integrate attitude with measured (bias-corrected) omega (body frame)
  const Eigen::Vector3f omega_b = this->state.v.ang.b = this->imu_meas.ang_vel;
  const Eigen::Quaternionf omega_q(0.f, omega_b.x(), omega_b.y(), omega_b.z());
  Eigen::Quaternionf qdot = (this->state.q * omega_q);
  this->state.q.coeffs() += 0.5f * static_cast<float> (dt) * qdot.coeffs();
  this->state.q.normalize();

  // Update angular velocity in world for later use
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;
}

void dlio::OdomNode::updateState() {

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  double dt = this->scan_stamp - this->prev_scan_stamp;

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // Constuct error quaternion
  qe = qhat.conjugate()*qin;

  double sgn = 1.;
  if (qe.w() < 0) {
    sgn = -1;
  }

  // Construct quaternion correction
  qcorr.w() = 1 - abs(qe.w());
  qcorr.vec() = sgn*qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;

  err_body = qhat.conjugate()._transformVector(err);

  double abias_max = this->geo_abias_max_;
  double gbias_max = this->geo_gbias_max_;

  // Update accel bias
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;

  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

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
  static double prev_stamp = imu_stamp_secs;
  double dt = imu_stamp_secs - prev_stamp;
  prev_stamp = imu_stamp_secs;
  
  if (dt <= 0) { dt = 1.0/400.0; }

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  static Eigen::Vector3f ang_vel_cg_prev = ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  ang_vel_cg_prev = ang_vel_cg;

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

  for (int i = 0; i < this->original_scan->points.size(); i++) {
    float d = std::sqrt(pow(this->original_scan->points[i].x, 2) +
                        pow(this->original_scan->points[i].y, 2));
    ds.push_back(d);
  }

  if (ds.empty()) {
    return;
  }

  // median
  std::nth_element(ds.begin(), ds.begin() + ds.size()/2, ds.end());
  float median_curr = ds[ds.size()/2];
  static float median_prev = median_curr;
  float median_lpf = 0.95*median_prev + 0.05*median_curr;
  median_prev = median_lpf;

  std::lock_guard<std::mutex> lock(g_metrics_mutex);

  // push
  this->metrics.spaciousness.push_back( median_lpf );

  if (this->metrics.spaciousness.size() > 400) {
    this->metrics.spaciousness.erase(this->metrics.spaciousness.begin(),
                                    this->metrics.spaciousness.end() - 400);
  }

  if (this->metrics.density.size() > 400) {
    this->metrics.density.erase(this->metrics.density.begin(),
                                this->metrics.density.end() - 400);
  }

}

void dlio::OdomNode::computeDensity() {

  float density;

  if (!this->geo.first_opt_done) {
    density = 0.;
  } else {
    density = this->gicp.source_density_;
  }

  static float density_prev = density;
  float density_lpf = 0.95*density_prev + 0.05*density;
  density_prev = density_lpf;

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
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
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
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
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
    this->keyframes.emplace_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
    this->keyframe_timestamps.emplace_back(this->scan_header_stamp);
    this->keyframe_normals.emplace_back(this->gicp.getSourceCovariances());
    this->keyframe_transformations.emplace_back(this->T_corr);
    return;
  }

  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;
  int num_nearby = 0;

  for (const auto& k : this->keyframes) {
    float dx = this->lidarPose.p[0] - k.first.first[0];
    float dy = this->lidarPose.p[1] - k.first.first[1];
    float dz = this->lidarPose.p[2] - k.first.first[2];
    float delta_d = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5f) ++num_nearby;
    if (delta_d < closest_d) { closest_d = delta_d; closest_idx = keyframes_idx; }
    ++keyframes_idx;
  }

  const Eigen::Vector3f&    closest_pose   = this->keyframes[closest_idx].first.first;
  const Eigen::Quaternionf& closest_pose_r = this->keyframes[closest_idx].first.second;

  float dx = this->lidarPose.p[0] - closest_pose[0];
  float dy = this->lidarPose.p[1] - closest_pose[1];
  float dz = this->lidarPose.p[2] - closest_pose[2];
  float dd = std::sqrt(dx*dx + dy*dy + dz*dz);

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

  if (newKeyframe) {
    if (this->keyframes.size() >= kMaxKeyframes) {
      const std::size_t removed = this->keyframes.size() - (kMaxKeyframes - 1);

      this->keyframes.erase(this->keyframes.begin(), this->keyframes.begin() + removed);
      this->keyframe_timestamps.erase(this->keyframe_timestamps.begin(), this->keyframe_timestamps.begin() + removed);
      this->keyframe_normals.erase(this->keyframe_normals.begin(), this->keyframe_normals.begin() + removed);
      this->keyframe_transformations.erase(this->keyframe_transformations.begin(), this->keyframe_transformations.begin() + removed);

      this->onKeyframesTrim(removed);   // <<< keep all index-based state consistent

      if (removed >= 16) { // only when we dropped a chunk
        keyframes.shrink_to_fit();
        keyframe_timestamps.shrink_to_fit();
        keyframe_normals.shrink_to_fit();
        keyframe_transformations.shrink_to_fit();
      }
    }

    this->keyframes.emplace_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
    this->keyframe_timestamps.emplace_back(this->scan_header_stamp);
    this->keyframe_normals.emplace_back(this->gicp.getSourceCovariances());
    this->keyframe_transformations.emplace_back(this->T_corr);
  }

}

void dlio::OdomNode::onKeyframesTrim(std::size_t removed) {
  if (removed == 0) return;

  // num_processed_keyframes tracks how many keyframes have been transformed/published
  if (this->num_processed_keyframes <= removed) this->num_processed_keyframes = 0;
  else                                          this->num_processed_keyframes -= removed;

  auto shift_down = [removed](std::vector<int>& idxs) {
    const int r = removed;
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

  if (sp < 0.5) { sp = 0.5; }
  if (sp > 5.0) { sp = 5.0; }

  this->keyframe_thresh_dist_ = sp;

  // Density
  float den = this->metrics.density.back();

  if (den < 0.5*this->gicp_max_corr_dist_) { den = 0.5*this->gicp_max_corr_dist_; }
  if (den > 2.0*this->gicp_max_corr_dist_) { den = 2.0*this->gicp_max_corr_dist_; }

  if (sp < 5.0) { den = 0.5*this->gicp_max_corr_dist_; };
  if (sp > 5.0) { den = 2.0*this->gicp_max_corr_dist_; };

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

void dlio::OdomNode::buildSubmap(State vehicle_state) {

  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();

  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    float d = sqrt( pow(vehicle_state.p[0] - this->keyframes[i].first.first[0], 2) +
                    pow(vehicle_state.p[1] - this->keyframes[i].first.first[1], 2) +
                    pow(vehicle_state.p[2] - this->keyframes[i].first.first[2], 2) );
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
            k >= static_cast<int>(this->keyframe_normals.size())) {
          continue;
        }

        // create current submap cloud
        *submap_cloud_ += *this->keyframes[k].second;

        // grab corresponding submap cloud's normals
        submap_normals_->insert( std::end(*submap_normals_),
            std::begin(*(this->keyframe_normals[k])), std::end(*(this->keyframe_normals[k])) );
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

void dlio::OdomNode::buildKeyframesAndSubmap(State vehicle_state) {

  // transform the new keyframe(s) and associated covariance list(s)
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++) {
    pcl::PointCloud<PointType>::ConstPtr raw_keyframe = this->keyframes[i].second;
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframe_normals[i];
    Eigen::Matrix4f T = this->keyframe_transformations[i];

    Eigen::Matrix4d Td = T.cast<double>();

    pcl::PointCloud<PointType>::Ptr transformed_keyframe = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*raw_keyframe, *transformed_keyframe, T);

    std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances (std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));
    std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
                   [&Td](Eigen::Matrix4d cov) { return Td * cov * Td.transpose(); });

    ++this->num_processed_keyframes;

    this->keyframes[i].second = transformed_keyframe;
    this->keyframe_normals[i] = transformed_covariances;

    this->publishKeyframe(this->keyframes[i], this->keyframe_timestamps[i]);
  }

  lock.unlock();

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();

  this->buildSubmap(vehicle_state);
}

void dlio::OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]{ return !this->main_loop_running || this->stop_.load(std::memory_order_relaxed); });
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
    std::accumulate(comp_snapshot.begin(), comp_snapshot.end(), 0.0) / comp_snapshot.size();
  const double max_comp_time = comp_snapshot.empty() ? 0.0 :
    *std::max_element(comp_snapshot.begin(), comp_snapshot.end());
  const double last_comp_time = comp_snapshot.empty() ? 0.0 : comp_snapshot.back();

  // RAM Usage
  double vm_usage = 0.0;
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
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = (timeSample.tms_stime - this->lastSysCPU) + (timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= (now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  if (this->cpu_percents.size() > 400) {
    this->cpu_percents.erase(this->cpu_percents.begin(), this->cpu_percents.end() - 400);
  }


  this->cpu_percents.push_back(cpu_percent);
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) / this->cpu_percents.size();

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

  std::time_t curr_time = this->scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time)); asc_time.pop_back();
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
