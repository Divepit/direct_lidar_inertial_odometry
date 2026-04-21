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

#include "dlio/dlio.h"

// ROS
#include "rclcpp/rclcpp.hpp"
#include "direct_lidar_inertial_odometry/srv/save_pcd.hpp"
#include <std_srvs/srv/trigger.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// PCL
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/crop_box.h>
#include <nav_msgs/msg/odometry.hpp>

#include <mutex>
#include <chrono>

class dlio::MapNode: public rclcpp::Node {

public:
  MapNode();
  ~MapNode() override;

  void start();
  void requestStop();

private:
  void getParams();
  bool shouldStop();

  void callbackKeyframe(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keyframe);
  void doPeriodicCrop();

  void savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,  // NOLINT(performance-unnecessary-value-param)
               std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res);  // NOLINT(performance-unnecessary-value-param)
  void resetMap(std::shared_ptr<std_srvs::srv::Trigger::Request> req,  // NOLINT(performance-unnecessary-value-param)
                std::shared_ptr<std_srvs::srv::Trigger::Response> res);  // NOLINT(performance-unnecessary-value-param)
  void callbackMapPose(const nav_msgs::msg::Odometry::ConstSharedPtr& odom);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr keyframe_sub;
  rclcpp::CallbackGroup::SharedPtr keyframe_cb_group, save_pcd_cb_group;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub;

  rclcpp::Service<direct_lidar_inertial_odometry::srv::SavePCD>::SharedPtr save_pcd_srv;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_map_srv_;
  rclcpp::CallbackGroup::SharedPtr reset_map_cb_group_;

  // Map storage & filtering
  pcl::PointCloud<PointType>::Ptr dlio_map{new pcl::PointCloud<PointType>()};
  pcl::PointCloud<PointType>::Ptr crop_buf_{new pcl::PointCloud<PointType>()};
  std::mutex map_mtx_;
  std::mutex pose_mtx_;
  pcl::VoxelGrid<PointType> voxelgrid;

  std::string odom_frame;
  double leaf_size_;

  // Cropping controls
  bool   crop_enabled_{false};
  double crop_box_size_{30.0};
  double crop_period_sec_{2.0};
  double crop_padding_{0.0};   // optional hysteresis/padding
  bool   have_pose_{false};
  Eigen::Vector3f robot_xyz_{0.f, 0.f, 0.f};
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr map_pose_sub_;

  // Reset epoch: keyframes stamped before this time are discarded as stale.
  rclcpp::Time reset_epoch_stamp_{0, 0, RCL_ROS_TIME};

  // (optional) live param updates
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_cb_;
  rclcpp::TimerBase::SharedPtr crop_timer_;
  std::atomic_bool stop_requested_{false};
};
