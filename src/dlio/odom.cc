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

#include <queue>

#include "rclcpp/qos.hpp"

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
  this->path_pub     = this->create_publisher<nav_msgs::msg::Path>("path", 1);
  this->kf_pose_pub  = this->create_publisher<geometry_msgs::msg::PoseArray>("kf_pose", 1);

  rclcpp::QoS reliable_qos(rclcpp::KeepLast(10));
  reliable_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  this->odom_map_pub = this->create_publisher<nav_msgs::msg::Odometry>("map_pose", reliable_qos);

  rclcpp::QoS qos( rclcpp::KeepLast(1) );
  qos.best_effort();
  qos.durability_volatile();
  qos.reliability( RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT );

  auto best_effort_qos = rclcpp::QoS(100)
                             .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                             .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE)
                             .history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);

  this->kf_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("kf_cloud", best_effort_qos);
  this->deskewed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed", best_effort_qos);
  this->deskewed_not_transformed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed_not_transformed", best_effort_qos);

  this->br = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

  // Velocity markers
  this->pub_lin_vel_marker_ = this->create_publisher<visualization_msgs::msg::Marker>("markers/velocity_linear", best_effort_qos);
  this->pub_ang_vel_marker_ = this->create_publisher<visualization_msgs::msg::Marker>("markers/velocity_angular", best_effort_qos);

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

dlio::OdomNode::~OdomNode() {

  stop_.store(true, std::memory_order_relaxed);
  pc_q_cv_.notify_all();
  q_cv_.notify_all();
  cv_imu_stamp.notify_all();
  submap_build_cv.notify_all();
  if (pointcloud_worker_.joinable()) pointcloud_worker_.join();
  if (pub_worker_.joinable()) pub_worker_.join();

}

void dlio::OdomNode::enqueuePublish(pcl::PointCloud<PointType>::ConstPtr cloud,
                                    const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
                                    const Eigen::Ref<const Eigen::Matrix4f>& T_all,
                                    double scanStamp) {
  {
    std::lock_guard<std::mutex> lk(q_mtx_);
    // Drop oldest if queue is backing up (keeps latency low)
    if (q_.size() > 2) q_.pop_front();
    q_.push_back(PubJob{cloud, T_cloud, T_all, this->scan_header_stamp, scanStamp});
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
    publishToROS(job.cloud, job.T_cloud, job.T_all, job.scanStamp);
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
  dlio::declare_param(this, "frames/odom", this->odom_frame, "odom");
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

}

void dlio::OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void dlio::OdomNode::publishPose() {

  // nav_msgs::msg::Odometry
  this->odom_ros.header.stamp = this->imu_stamp;
  this->odom_ros.header.frame_id = this->odom_frame;
  this->odom_ros.child_frame_id = this->baselink_frame;

  this->odom_ros.pose.pose.position.x = this->state.p[0];
  this->odom_ros.pose.pose.position.y = this->state.p[1];
  this->odom_ros.pose.pose.position.z = this->state.p[2];

  this->odom_ros.pose.pose.orientation.w = this->state.q.w();
  this->odom_ros.pose.pose.orientation.x = this->state.q.x();
  this->odom_ros.pose.pose.orientation.y = this->state.q.y();
  this->odom_ros.pose.pose.orientation.z = this->state.q.z();

  // Publish twist in child (base) frame per REP-103 conventions
  this->odom_ros.twist.twist.linear.x = this->state.v.lin.b[0];
  this->odom_ros.twist.twist.linear.y = this->state.v.lin.b[1];
  this->odom_ros.twist.twist.linear.z = this->state.v.lin.b[2];

  this->odom_ros.twist.twist.angular.x = this->state.v.ang.b[0];
  this->odom_ros.twist.twist.angular.y = this->state.v.ang.b[1];
  this->odom_ros.twist.twist.angular.z = this->state.v.ang.b[2];

  this->odom_pub->publish(this->odom_ros);

  // geometry_msgs::msg::PoseStamped
  this->pose_ros.header.stamp = this->imu_stamp;
  this->pose_ros.header.frame_id = this->odom_frame;

  this->pose_ros.pose.position.x = this->state.p[0];
  this->pose_ros.pose.position.y = this->state.p[1];
  this->pose_ros.pose.position.z = this->state.p[2];

  this->pose_ros.pose.orientation.w = this->state.q.w();
  this->pose_ros.pose.orientation.x = this->state.q.x();
  this->pose_ros.pose.orientation.y = this->state.q.y();
  this->pose_ros.pose.orientation.z = this->state.q.z();

  this->pose_pub->publish(this->pose_ros);

}

void dlio::OdomNode::publishToROS(pcl::PointCloud<PointType>::ConstPtr cloud,
                                  const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
                                  const Eigen::Ref<const Eigen::Matrix4f>& T_all,
                                  const double scanStamp)
{

  // ---- Odometry (map frame) ----
  nav_msgs::msg::Odometry odom_map;
  odom_map.header.frame_id = "dlio_map";
  odom_map.child_frame_id  = this->baselink_frame;

  // stamp (avoid double→int roundtrip each time if you already have rclcpp::Time)
  const uint64_t nsec = static_cast<uint64_t>(scanStamp * 1e9);
  odom_map.header.stamp.sec     = static_cast<int32_t>(nsec / 1000000000ULL);
  odom_map.header.stamp.nanosec = static_cast<uint32_t>(nsec % 1000000000ULL);
  const rclcpp::Time scan_time(odom_map.header.stamp);

  Eigen::Vector3f state_p, state_vlin_b, state_vang_b;
  Eigen::Quaternionf state_q;
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    state_p = this->state.p;
    state_q = this->state.q;
    state_vlin_b = this->state.v.lin.b;
    state_vang_b = this->state.v.ang.b;
  }

  // pose from T_all
  odom_map.pose.pose.position.x = T_all(0,3);
  odom_map.pose.pose.position.y = T_all(1,3);
  odom_map.pose.pose.position.z = T_all(2,3);

  const Eigen::Matrix3f R = T_all.block<3,3>(0,0);
  const Eigen::Quaternionf q(R); // assume R orthonormal from your pipeline

  odom_map.pose.pose.orientation.w = q.w();
  odom_map.pose.pose.orientation.x = q.x();
  odom_map.pose.pose.orientation.y = q.y();
  odom_map.pose.pose.orientation.z = q.z();

  // twist from state snapshot
  odom_map.twist.twist.linear.x  = state_vlin_b[0];
  odom_map.twist.twist.linear.y  = state_vlin_b[1];
  odom_map.twist.twist.linear.z  = state_vlin_b[2];
  odom_map.twist.twist.angular.x = state_vang_b[0];
  odom_map.twist.twist.angular.y = state_vang_b[1];
  odom_map.twist.twist.angular.z = state_vang_b[2];

  // ---- Publish odom ----
  this->odom_map_pub->publish(odom_map);

  // ---- Path (bounded) ----
  this->path_ros.header.stamp = scan_time;
  this->path_ros.header.frame_id = this->odom_frame;

  geometry_msgs::msg::PoseStamped p;
  p.header.stamp = scan_time;
  p.header.frame_id = this->odom_frame;
  p.pose.position.x = T_all(0,3);
  p.pose.position.y = T_all(1,3);
  p.pose.position.z = T_all(2,3);
  p.pose.orientation.w = q.w();
  p.pose.orientation.x = q.x();
  p.pose.orientation.y = q.y();
  p.pose.orientation.z = q.z();

  constexpr size_t kMaxPath = 1500; // cap
  if (this->path_ros.poses.size() >= kMaxPath) {
    this->path_ros.poses.erase(this->path_ros.poses.begin(),
                               this->path_ros.poses.begin() + (this->path_ros.poses.size() - kMaxPath + 1));
  }
  this->path_ros.poses.push_back(std::move(p));
  this->path_pub->publish(this->path_ros);

  // ---- TFs (batch) ----
  std::vector<geometry_msgs::msg::TransformStamped> tfs;
  tfs.reserve(2);

  geometry_msgs::msg::TransformStamped tf_map_bl;
  tf_map_bl.header.stamp = odom_map.header.stamp;
  tf_map_bl.header.frame_id = "dlio_map";
  tf_map_bl.child_frame_id  = this->baselink_frame;
  tf_map_bl.transform.translation.x = T_all(0,3);
  tf_map_bl.transform.translation.y = T_all(1,3);
  tf_map_bl.transform.translation.z = T_all(2,3);
  tf_map_bl.transform.rotation.w = q.w();
  tf_map_bl.transform.rotation.x = q.x();
  tf_map_bl.transform.rotation.y = q.y();
  tf_map_bl.transform.rotation.z = q.z();
  tfs.emplace_back(std::move(tf_map_bl));

  geometry_msgs::msg::TransformStamped tf_bl_odom;
  tf_bl_odom.header.stamp = scan_time;
  tf_bl_odom.header.frame_id = this->baselink_frame;
  tf_bl_odom.child_frame_id  = this->odom_frame;

  const Eigen::Quaternionf inv_q = state_q.conjugate();
  const Eigen::Vector3f inv_t = -(inv_q._transformVector(state_p));
  tf_bl_odom.transform.translation.x = inv_t[0];
  tf_bl_odom.transform.translation.y = inv_t[1];
  tf_bl_odom.transform.translation.z = inv_t[2];
  tf_bl_odom.transform.rotation.w = inv_q.w();
  tf_bl_odom.transform.rotation.x = inv_q.x();
  tf_bl_odom.transform.rotation.y = inv_q.y();
  tf_bl_odom.transform.rotation.z = inv_q.z();
  tfs.emplace_back(std::move(tf_bl_odom));

  br->sendTransform(tfs);


  publishCloud(cloud, T_cloud, T_all);
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

void dlio::OdomNode::publishCloud(pcl::PointCloud<PointType>::ConstPtr cloud,
                                  const Eigen::Ref<const Eigen::Matrix4f>& T_cloud,
                                  const Eigen::Ref<const Eigen::Matrix4f>& T_all)
{
  if (this->wait_until_move_ && this->length_traversed < 0.1) return;

  const size_t n = cloud->size();
  if (n == 0) return;

  const Eigen::Matrix4f T_revert = T_all.inverse() * T_cloud;
  const rclcpp::Time cloud_stamp = this->path_ros.header.stamp;

  sensor_msgs::msg::PointCloud2 deskewed_ros;
  sensor_msgs::msg::PointCloud2 deskewed_original_ros;

  prepare_xyz_msg(deskewed_ros,          this->odom_frame,     cloud_stamp, n);
  prepare_xyz_msg(deskewed_original_ros, this->baselink_frame, cloud_stamp, n);

  sensor_msgs::PointCloud2Iterator<float> x1(deskewed_ros, "x");
  sensor_msgs::PointCloud2Iterator<float> y1(deskewed_ros, "y");
  sensor_msgs::PointCloud2Iterator<float> z1(deskewed_ros, "z");

  sensor_msgs::PointCloud2Iterator<float> x2(deskewed_original_ros, "x");
  sensor_msgs::PointCloud2Iterator<float> y2(deskewed_original_ros, "y");
  sensor_msgs::PointCloud2Iterator<float> z2(deskewed_original_ros, "z");

  bool all_finite = true;
  for (size_t i = 0; i < n; ++i, ++x1, ++y1, ++z1, ++x2, ++y2, ++z2) {
    const auto& p = (*cloud)[i];
    const Eigen::Vector4f v(p.x, p.y, p.z, 1.f);

    const Eigen::Vector4f v1 = T_cloud  * v;   // odom frame
    const Eigen::Vector4f v2 = T_revert * v;   // baselink frame

    *x1 = v1.x(); *y1 = v1.y(); *z1 = v1.z();
    *x2 = v2.x(); *y2 = v2.y(); *z2 = v2.z();

    all_finite &= std::isfinite(v1.x()) & std::isfinite(v1.y()) & std::isfinite(v1.z())
               &  std::isfinite(v2.x()) & std::isfinite(v2.y()) & std::isfinite(v2.z());
  }

  deskewed_ros.is_dense          = all_finite;
  deskewed_original_ros.is_dense = all_finite;

  auto m1 = std::make_unique<sensor_msgs::msg::PointCloud2>(std::move(deskewed_ros));
  auto m2 = std::make_unique<sensor_msgs::msg::PointCloud2>(std::move(deskewed_original_ros));
  this->deskewed_pub->publish(std::move(m1));
  this->deskewed_not_transformed_pub->publish(std::move(m2));
}


void dlio::OdomNode::publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp) {

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

  // Publish
  this->kf_pose_ros.header.stamp = timestamp;
  this->kf_pose_ros.header.frame_id = this->odom_frame;
  this->kf_pose_pub->publish(this->kf_pose_ros);

  // publish keyframe scan for map
  if (this->vf_use_) {
    if (kf.second->points.size() == kf.second->width * kf.second->height) {
      sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = timestamp;
      keyframe_cloud_ros.header.frame_id = this->odom_frame;
      this->kf_cloud_pub->publish(keyframe_cloud_ros);
    }
  } else {
    sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
    pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
    keyframe_cloud_ros.header.stamp = timestamp;
    keyframe_cloud_ros.header.frame_id = this->odom_frame;
    this->kf_cloud_pub->publish(keyframe_cloud_ros);
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

  if (original_scan_->empty()) {
    this->deskew_ = false;
    return;
  }

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

  if (this->prev_scan_stamp > 0.0) {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "[deskew interval dbg] prev_first=%.9f curr_first=%.9f curr_last=%.9f "
        "scan_span=%.3f ms query_span=%.3f ms",
        this->prev_scan_stamp,
        timestamps.front(),
        timestamps.back(),
        1e3 * (timestamps.back() - timestamps.front()),
        1e3 * (timestamps.back() - this->prev_scan_stamp));
  }

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
    RCLCPP_FATAL(this->get_logger(),"Bad time sync between LiDAR and IMU!");

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

  double then = this->now().seconds();

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
      std::async( std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state );
    this->submap_future.wait(); // wait until completion
    return;
  }

  // Get the next pose via IMU + S2M + GEO
  this->getNextPose();

  // Update current keyframe poses and map
  this->updateKeyframes();

  // Build keyframe normals and submap if needed (and if we're not already waiting)
  if (this->new_submap_is_ready) {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_future =
      std::async( std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state );
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
  // this->lidar_rates.push_back( 1. / (this->scan_stamp - this->prev_scan_stamp) );
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // Publish stuff to ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_) {
    published_cloud = this->current_scan;
  } else {
    published_cloud = this->deskewed_scan;
  }
  this->enqueuePublish(published_cloud, this->T_corr, this->T, this->scan_stamp);

  // Update some statistics
  this->comp_times.push_back(this->now().seconds() - then);

  if (this->comp_times.size() > 400) {
    this->comp_times.erase(this->comp_times.begin(), this->comp_times.end() - 400);
  }

  this->gicp_hasConverged = this->gicp.hasConverged();

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

  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

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
    stamp  = this->imu_stamp;   // set in callbackImu before propagateState()
  }

  // Build and publish Odometry
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = this->odom_frame;
  odom.child_frame_id  = this->baselink_frame;

  odom.pose.pose.position.x = p.x();
  odom.pose.pose.position.y = p.y();
  odom.pose.pose.position.z = p.z();
  odom.pose.pose.orientation.w = q.w();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();

  // Twist in child (base) per REP-103
  odom.twist.twist.linear.x  = vlin_b.x();
  odom.twist.twist.linear.y  = vlin_b.y();
  odom.twist.twist.linear.z  = vlin_b.z();
  odom.twist.twist.angular.x = vang_b.x();
  odom.twist.twist.angular.y = vang_b.y();
  odom.twist.twist.angular.z = vang_b.z();

  this->odom_pub->publish(odom);

  // Build and publish PoseStamped
  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = this->odom_frame;
  pose.pose.position.x = p.x();
  pose.pose.position.y = p.y();
  pose.pose.position.z = p.z();
  pose.pose.orientation.w = q.w();
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();

  this->pose_pub->publish(pose);

  this->publishVelocityMarkers(stamp, vlin_b, vang_b);
}

void dlio::OdomNode::publishVelocityMarkers(const rclcpp::Time& stamp,
                                            const Eigen::Vector3f& vlin_b,
                                            const Eigen::Vector3f& vang_b) {
  if (!viz_vel_markers_) return;

  visualization_msgs::msg::Marker m_lin, m_ang;
  
  createLinVelocityMarker(this->baselink_frame, stamp, vlin_b, m_lin);
  createAngularVelocityMarker(this->baselink_frame, stamp, vang_b, m_ang);

  if (this->pub_lin_vel_marker_ && this->pub_lin_vel_marker_->get_subscription_count() > 0) {
    this->pub_lin_vel_marker_->publish(m_lin);
  }
  if (this->pub_ang_vel_marker_ && this->pub_ang_vel_marker_->get_subscription_count() > 0) {
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

  // Run scan-to-submap registration with T_prior as initial guess.
  // Call chain:
  //   gicp.align()
  //     -> NanoGICP::computeTransformation()
  //     -> LsqRegistration::computeTransformation()
  //     -> step_optimize() (LM by default)
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  this->gicp.align(*aligned);

  // Correction from registration (source -> target, both already in world frame).
  this->T_corr = this->gicp.getFinalTransformation(); // "correction" transformation
  this->T = this->T_corr * this->T_prior;
  // this->computeMotionDeviation();

  // Update next global pose
  // Both source and target clouds are in the global frame now, so tranformation is global
  this->propagateGICP();

  // Geometric observer update
  this->updateState();

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
    float dx = this->state.p[0] - k.first.first[0];
    float dy = this->state.p[1] - k.first.first[1];
    float dz = this->state.p[2] - k.first.first[2];
    float delta_d = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5f) ++num_nearby;
    if (delta_d < closest_d) { closest_d = delta_d; closest_idx = keyframes_idx; }
    ++keyframes_idx;
  }

  const Eigen::Vector3f&    closest_pose   = this->keyframes[closest_idx].first.first;
  const Eigen::Quaternionf& closest_pose_r = this->keyframes[closest_idx].first.second;

  float dx = this->state.p[0] - closest_pose[0];
  float dy = this->state.p[1] - closest_pose[1];
  float dz = this->state.p[2] - closest_pose[2];
  float dd = std::sqrt(dx*dx + dy*dy + dz*dz);

  Eigen::Quaternionf dq;
  if (this->state.q.dot(closest_pose_r) < 0.f) {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w()*=-1.f; lq.x()*=-1.f; lq.y()*=-1.f; lq.z()*=-1.f;
    dq = this->state.q * lq.inverse();
  } else {
    dq = this->state.q * closest_pose_r.inverse();
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

    for (auto k : this->submap_kf_idx_curr) {
      std::unique_lock<decltype(this->keyframes_mutex)> submap_lock(this->keyframes_mutex);
      if (k < 0 || k >= this->keyframes.size() || k >= this->keyframe_normals.size()) {
        continue;
      }

      // create current submap cloud
      *submap_cloud_ += *this->keyframes[k].second;

      // grab corresponding submap cloud's normals
      submap_normals_->insert( std::end(*submap_normals_),
          std::begin(*(this->keyframe_normals[k])), std::end(*(this->keyframe_normals[k])) );
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

  // Total length traversed
  double length_traversed = 0.;
  Eigen::Vector3f p_curr = Eigen::Vector3f(0., 0., 0.);
  Eigen::Vector3f p_prev = Eigen::Vector3f(0., 0., 0.);
  for (const auto& t : this->trajectory) {
    if (p_prev == Eigen::Vector3f(0., 0., 0.)) {
      p_prev = t.first;
      continue;
    }
    p_curr = t.first;
    double l = sqrt(pow(p_curr[0] - p_prev[0], 2) + pow(p_curr[1] - p_prev[1], 2) + pow(p_curr[2] - p_prev[2], 2));

    if (l >= 0.1) {
      length_traversed += l;
      p_prev = p_curr;
    }
  }
  this->length_traversed = length_traversed;

  // Average computation time
  double avg_comp_time =
    std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) / this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  (void)win_size;

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
    << std::setfill(' ') << std::setw(6) << this->comp_times.back()*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << *std::max_element(this->comp_times.begin(), this->comp_times.end())*1000.
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
