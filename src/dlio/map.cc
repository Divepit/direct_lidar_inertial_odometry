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

#include "dlio/map.h"
#include "dlio/utils.h"

namespace {

template <typename PublisherPtrT>
bool hasSubscribers(const PublisherPtrT& pub) {
  return pub &&
         (pub->get_subscription_count() > 0 ||
          pub->get_intra_process_subscription_count() > 0);
}

}  // namespace

dlio::MapNode::MapNode() : Node("dlio_map_node") {
  this->getParams();

  // Subscribers / publishers
  this->keyframe_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "kf_cloud",
      rclcpp::QoS(100)
          .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
          .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE)
          .history(RMW_QOS_POLICY_HISTORY_KEEP_LAST),
      std::bind(&dlio::MapNode::callbackKeyframe, this, std::placeholders::_1));

  if (save_dynamic_removed_enabled_) {
    auto removed_qos = rclcpp::QoS(100)
                           .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                           .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE)
                           .history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);
    this->dynamic_removed_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "dynamic_removed",
        removed_qos,
        std::bind(&dlio::MapNode::callbackDynamicRemoved, this, std::placeholders::_1));
  }

  this->map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map", 100);

  auto qos_pose = rclcpp::QoS(10)
                      .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                      .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

  this->map_pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "map_pose", qos_pose,
      std::bind(&dlio::MapNode::callbackMapPose, this, std::placeholders::_1));

  // Periodic crop timer (only if enabled with positive period)
  if (crop_enabled_ && crop_period_sec_ > 0.0) {
    auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(this->crop_period_sec_));
    this->crop_timer_ = this->create_wall_timer(
        period, std::bind(&dlio::MapNode::doPeriodicCrop, this));
  }

  // Runtime param updates
  params_cb_ = this->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> &ps) {
        rcl_interfaces::msg::SetParametersResult res;
        res.successful = true;
        bool restart_timer = false;
        bool want_timer = crop_enabled_;
        double new_period = crop_period_sec_;

        for (const auto &p : ps) {
          const auto &n = p.get_name();
          if (n == "map/crop/enabled") {
            crop_enabled_ = p.as_bool();
            want_timer = crop_enabled_;
            restart_timer = true;
          } else if (n == "map/crop/box_size") {
            crop_box_size_ = p.as_double();
          } else if (n == "map/crop/padding") {
            crop_padding_ = p.as_double();
          } else if (n == "map/crop/period_sec") {
            new_period = p.as_double();
            crop_period_sec_ = new_period;
            restart_timer = true;
          }
        }

        if (restart_timer) {
          // Cancel existing timer if any
          if (crop_timer_) {
            crop_timer_->cancel();
            crop_timer_.reset();
          }
          // Recreate only if enabled and period is positive
          if (want_timer && crop_period_sec_ > 0.0) {
            auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(crop_period_sec_));
            crop_timer_ = this->create_wall_timer(
                period, std::bind(&dlio::MapNode::doPeriodicCrop, this));
          }
        }
        return res;
      });

  // Ensure the map cloud is allocated
  if (!this->dlio_map) {
    this->dlio_map = std::make_shared<pcl::PointCloud<PointType>>();
  }

  // SavePCD service
  save_pcd_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  save_pcd_srv = this->create_service<direct_lidar_inertial_odometry::srv::SavePCD>(
      "save_pcd",
      std::bind(&dlio::MapNode::savePCD, this, std::placeholders::_1, std::placeholders::_2),
      rclcpp::ServicesQoS(),
      save_pcd_cb_group);

  // ResetMap service
  reset_map_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  reset_map_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "dlio/reset_map",
      std::bind(&dlio::MapNode::resetMap, this, std::placeholders::_1, std::placeholders::_2),
      rclcpp::ServicesQoS(),
      reset_map_cb_group_);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
}

dlio::MapNode::~MapNode() {
  this->requestStop();
  params_cb_.reset();
  crop_timer_.reset();
}

void dlio::MapNode::requestStop() {
  if (stop_requested_.exchange(true, std::memory_order_relaxed)) {
    return;
  }

  RCLCPP_INFO(this->get_logger(),
              "\033[38;5;214m[SHUTDOWN] Map node stopping. Cancelling timers and exiting...\033[0m");

  if (crop_timer_) {
    crop_timer_->cancel();
    crop_timer_.reset();
  }
}

bool dlio::MapNode::shouldStop() {
  if (stop_requested_.load(std::memory_order_relaxed)) {
    return true;
  }

  const auto context = this->get_node_base_interface()->get_context();
  return !context || !context->is_valid();
}

void dlio::MapNode::getParams() {
  this->declare_parameter<std::string>("frames/odom", "odom");
  this->declare_parameter<double>("map/sparse/leafSize", 0.5);
  this->declare_parameter<bool>("map/crop/enabled", false);
  this->declare_parameter<double>("map/crop/box_size", 30.0);
  this->declare_parameter<double>("map/crop/period_sec", 2.0);
  this->declare_parameter<double>("map/crop/padding", 0.0);
  this->declare_parameter<bool>("map/save_dynamic_removed/enabled", false);

  this->get_parameter("frames/odom", this->odom_frame);
  this->get_parameter("map/sparse/leafSize", this->leaf_size_);

  this->get_parameter("map/crop/enabled", this->crop_enabled_);
  this->get_parameter("map/crop/box_size", this->crop_box_size_);
  this->get_parameter("map/crop/period_sec", this->crop_period_sec_);
  this->get_parameter("map/crop/padding", this->crop_padding_);
  this->get_parameter("map/save_dynamic_removed/enabled", this->save_dynamic_removed_enabled_);
}

void dlio::MapNode::start() {}

void dlio::MapNode::callbackMapPose(const nav_msgs::msg::Odometry::ConstSharedPtr &odom) {
  if (this->shouldStop()) {
    return;
  }

  std::lock_guard<std::mutex> lk(pose_mtx_);
  robot_xyz_ = Eigen::Vector3f(static_cast<float>(odom->pose.pose.position.x),
                               static_cast<float>(odom->pose.pose.position.y),
                               static_cast<float>(odom->pose.pose.position.z));
  have_pose_ = true;
}

void dlio::MapNode::callbackKeyframe(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &keyframe) {
  if (this->shouldStop()) {
    return;
  }

  // Convert to PCL
  pcl::PointCloud<PointType>::Ptr keyframe_pcl(new pcl::PointCloud<PointType>());
  pcl::fromROSMsg(*keyframe, *keyframe_pcl);

  // Voxel filter (kept as member for minimal changes; safe with default mutually-exclusive callback group)
  const float leaf_size = static_cast<float>(this->leaf_size_);
  this->voxelgrid.setLeafSize(leaf_size, leaf_size, leaf_size);
  this->voxelgrid.setInputCloud(keyframe_pcl);
  this->voxelgrid.filter(*keyframe_pcl);

  // Accumulate into map — epoch check is inside the lock so it is coherent with
  // the write in resetMap() which holds the same lock.
  {
    std::lock_guard<std::mutex> lk(map_mtx_);
    if (rclcpp::Time(keyframe->header.stamp) < reset_epoch_stamp_) {
      RCLCPP_DEBUG(this->get_logger(),
                   "[MAP] Discarding stale pre-reset keyframe (stamp %.3f < epoch %.3f).",
                   rclcpp::Time(keyframe->header.stamp).seconds(),
                   reset_epoch_stamp_.seconds());
      return;
    }
    *this->dlio_map += *keyframe_pcl;
  }

  // Publish (original organized check kept)
  const auto organized_point_count =
      static_cast<decltype(this->dlio_map->points.size())>(this->dlio_map->width) *
      static_cast<decltype(this->dlio_map->points.size())>(this->dlio_map->height);
  if (this->dlio_map->points.size() == organized_point_count) {
    if (hasSubscribers(this->map_pub)) {
      // Snapshot pose flags and map ptr
      Eigen::Vector3f pose;
      bool have_pose = false;
      {
        std::lock_guard<std::mutex> pk(pose_mtx_);
        pose = robot_xyz_;
        have_pose = have_pose_;
      }
      pcl::PointCloud<PointType>::Ptr map_ptr;
      {
        std::lock_guard<std::mutex> lk(map_mtx_);
        map_ptr = dlio_map;  // shared_ptr copy
      }

      // If periodic cropping is active, don't crop again here
      const bool do_pub_crop = (crop_enabled_ && have_pose && crop_period_sec_ <= 0.0);

      pcl::PointCloud<PointType>::Ptr out_cloud = map_ptr;
      if (do_pub_crop) {
        const float half = static_cast<float>(0.5f * this->crop_box_size_);
        pcl::CropBox<PointType> box;
        box.setMin(Eigen::Vector4f(pose.x() - half, pose.y() - half, pose.z() - half, 1.0f));
        box.setMax(Eigen::Vector4f(pose.x() + half, pose.y() + half, pose.z() + half, 1.0f));
        auto cropped = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
        box.setInputCloud(map_ptr);
        box.filter(*cropped);
        out_cloud = cropped;
      }

      sensor_msgs::msg::PointCloud2 map_ros;
      pcl::toROSMsg(*out_cloud, map_ros);
      map_ros.header.stamp = this->now();
      map_ros.header.frame_id = this->odom_frame;
      this->map_pub->publish(map_ros);
    }
  }
}

void dlio::MapNode::callbackDynamicRemoved(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &removed) {
  if (this->shouldStop() || !save_dynamic_removed_enabled_) {
    return;
  }
  if (!removed || removed->width == 0U || removed->height == 0U || removed->data.empty()) {
    return;
  }

  pcl::PointCloud<PointType>::Ptr removed_pcl(new pcl::PointCloud<PointType>());
  pcl::fromROSMsg(*removed, *removed_pcl);
  if (removed_pcl->empty()) {
    return;
  }

  std::lock_guard<std::mutex> lk(dynamic_removed_mtx_);
  ++dynamic_removed_topic_count_;
  dynamic_removed_raw_count_ += removed_pcl->size();
  *dynamic_removed_map_ += *removed_pcl;
}

void dlio::MapNode::doPeriodicCrop() {
  if (this->shouldStop()) {
    return;
  }

  if (!crop_enabled_) return;

  Eigen::Vector3f pose;
  {
    std::lock_guard<std::mutex> pk(pose_mtx_);
    if (!have_pose_) return;
    pose = robot_xyz_;
  }

  const float half = static_cast<float>(0.5f * crop_box_size_);
  const float pad  = static_cast<float>(crop_padding_);

  pcl::CropBox<PointType> box;
  box.setMin(Eigen::Vector4f(pose.x() - (half + pad),
                             pose.y() - (half + pad),
                             pose.z() - (half + pad), 1.0f));
  box.setMax(Eigen::Vector4f(pose.x() + (half + pad),
                             pose.y() + (half + pad),
                             pose.z() + (half + pad), 1.0f));

  // Allocate a fresh output and swap under lock to avoid races with publishers
  auto new_map = std::make_shared<pcl::PointCloud<PointType>>();
  {
    std::lock_guard<std::mutex> lk(map_mtx_);
    if (!dlio_map || dlio_map->empty()) return;
    box.setInputCloud(dlio_map);
    box.filter(*new_map);
    dlio_map.swap(new_map);  // old map kept alive in new_map until scope ends
  }
  // new_map (old map) released here when no other refs exist.
}

void dlio::MapNode::resetMap(std::shared_ptr<std_srvs::srv::Trigger::Request> /*unused*/,  // NOLINT(performance-unnecessary-value-param)
                             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {  // NOLINT(performance-unnecessary-value-param)
  if (this->shouldStop()) {
    res->success = false;
    res->message = "Map reset aborted: node is shutting down.";
    return;
  }

  // Acquire mutexes together to prevent reset/save/callback races.
  std::scoped_lock map_pose_lock(map_mtx_, pose_mtx_, dynamic_removed_mtx_);

  // Clear the accumulated map
  dlio_map = std::make_shared<pcl::PointCloud<PointType>>();
  dynamic_removed_map_ = std::make_shared<pcl::PointCloud<PointType>>();
  dynamic_removed_topic_count_ = 0U;
  dynamic_removed_raw_count_ = 0U;

  // Reset pose tracking
  have_pose_ = false;
  robot_xyz_ = Eigen::Vector3f::Zero();

  // Reset the crop timer so it doesn't fire against an empty map with stale state
  if (crop_timer_) {
    crop_timer_->cancel();
    crop_timer_.reset();
  }
  if (!this->shouldStop() && crop_enabled_ && crop_period_sec_ > 0.0) {
    auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(crop_period_sec_));
    crop_timer_ = this->create_wall_timer(
        period, [this] { this->doPeriodicCrop(); });
  }

  // Record epoch so stale in-flight keyframes are discarded in callbackKeyframe.
  reset_epoch_stamp_ = this->now();

  RCLCPP_INFO(this->get_logger(),
              "\033[32m[MAP RESET] Map cleared, pose reset. Epoch stamp set to %.3f s.\033[0m",
              reset_epoch_stamp_.seconds());
  res->success = true;
  res->message = "Map reset successfully.";
}

void dlio::MapNode::savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,  // NOLINT(performance-unnecessary-value-param)
                            std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res) {  // NOLINT(performance-unnecessary-value-param)
  pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr removed_cloud(new pcl::PointCloud<PointType>());
  std::size_t removed_topic_count = 0U;
  std::size_t removed_raw_count = 0U;
  {
    std::lock_guard<std::mutex> map_lock(map_mtx_);
    *map_cloud = *this->dlio_map;  // copy under lock
  }
  {
    std::lock_guard<std::mutex> removed_lock(dynamic_removed_mtx_);
    *removed_cloud = *this->dynamic_removed_map_;
    removed_topic_count = dynamic_removed_topic_count_;
    removed_raw_count = dynamic_removed_raw_count_;
  }

  float leaf_size = req->leaf_size;
  std::string save_path = req->save_path;

  std::cout << std::setprecision(2) << "Saving map to " << save_path + "/dlio_map.pcd"
            << " with leaf size " << to_string_with_precision(leaf_size, 2) << "... ";
  std::cout.flush();

  const std::size_t map_raw_count = map_cloud->size();
  pcl::VoxelGrid<PointType> voxel_grid;
  voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_grid.setInputCloud(map_cloud);
  voxel_grid.filter(*map_cloud);

  int ret = pcl::io::savePCDFileBinary(save_path + "/dlio_map.pcd", *map_cloud);
  const int clean_alias_ret =
      pcl::io::savePCDFileBinary(save_path + "/clean_map.pcd", *map_cloud);
  if (ret == 0 && clean_alias_ret != 0) {
    ret = clean_alias_ret;
  }
  const std::size_t map_voxel_count = map_cloud->size();
  int removed_ret = 0;

  std::size_t removed_voxel_count = removed_cloud->size();
  if (save_dynamic_removed_enabled_) {
    if (!removed_cloud->empty()) {
      pcl::VoxelGrid<PointType> removed_voxel_grid;
      removed_voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
      removed_voxel_grid.setInputCloud(removed_cloud);
      removed_voxel_grid.filter(*removed_cloud);
      removed_voxel_count = removed_cloud->size();
    }
    removed_ret =
        pcl::io::savePCDFileBinary(save_path + "/dlio_dynamic_removed_map.pcd", *removed_cloud);
    if (removed_ret == 0) {
      removed_ret =
          pcl::io::savePCDFileBinary(save_path + "/dynamic_points.pcd", *removed_cloud);
    }
  }

  {
    std::ofstream summary(save_path + "/save_summary.txt");
    if (summary) {
      summary << "leaf_size=" << leaf_size << '\n'
              << "clean_map_alias=clean_map.pcd\n"
              << "map_raw_points=" << map_raw_count << '\n'
              << "map_voxel_points=" << map_voxel_count << '\n'
              << "dynamic_removed_enabled=" << (save_dynamic_removed_enabled_ ? "true" : "false") << '\n'
              << "dynamic_removed_topics=" << removed_topic_count << '\n'
              << "dynamic_removed_alias="
              << (save_dynamic_removed_enabled_ ? "dynamic_points.pcd" : "") << '\n'
              << "dynamic_removed_raw_points=" << removed_raw_count << '\n'
              << "dynamic_removed_voxel_points=" << removed_voxel_count << '\n';
    }
  }

  res->success = (ret == 0) &&
                 (!save_dynamic_removed_enabled_ ||
                  removed_ret == 0);

  if (res->success) {
    std::cout << "done\n";
  } else {
    std::cout << "failed\n";
  }
}
