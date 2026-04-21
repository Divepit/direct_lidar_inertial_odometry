# ros2_gazebo ← ros2_hiking Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `feature/ros2_hiking` into `feature/ros2_gazebo`, resolve all conflicts, preserve the Gazebo sim workflow (external-odom mode) via a new `dlio_sim.launch.py`, and leave the standard real-robot `dlio.launch.py` identical to the hiking branch version.

**Architecture:** The hiking branch is the authoritative base for all DLIO core code. The gazebo branch's one conceptual addition — replacing IMU with external ground-truth odometry — is re-applied at four integration points in `odom.cc`/`odom.h` on top of the merged hiking code. A separate `dlio_sim.launch.py` carries the sim-specific launch configuration.

**Tech Stack:** ROS 2, C++17, nav_msgs/Odometry, colcon build

---

### Task 1: Start the merge

**Files:**
- (git operations only)

- [ ] **Step 1: Start the merge**

```bash
git merge --no-ff feature/ros2_hiking
```

Expected: merge fails with conflicts in `CMakeLists.txt`, `cfg/params.yaml`, `include/dlio/odom.h`, `launch/dlio.launch.py`, `package.xml`, `src/dlio/map.cc`, `src/dlio/odom.cc`.

- [ ] **Step 2: Verify conflict list**

```bash
git diff --name-only --diff-filter=U
```

Expected output (all seven files):
```
CMakeLists.txt
cfg/params.yaml
include/dlio/odom.h
launch/dlio.launch.py
package.xml
src/dlio/map.cc
src/dlio/odom.cc
```

---

### Task 2: Resolve "take hiking entirely" files

These five files have no gazebo-specific content — take the hiking branch version verbatim.
Note: `cfg/dlio.yaml` is NOT in this list — git auto-merges it cleanly (different lines changed on each branch).

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `cfg/params.yaml`
- Modify: `package.xml`
- Modify: `src/dlio/map.cc`
- Modify: `launch/dlio.launch.py`

- [ ] **Step 1: Check out all five files from the hiking branch**

```bash
git checkout feature/ros2_hiking -- \
  CMakeLists.txt \
  cfg/params.yaml \
  package.xml \
  src/dlio/map.cc \
  launch/dlio.launch.py
```

- [ ] **Step 2: Stage them**

```bash
git add CMakeLists.txt cfg/params.yaml package.xml src/dlio/map.cc launch/dlio.launch.py
```

- [ ] **Step 3: Confirm no conflict markers remain in those files**

```bash
grep -l "<<<<<<" CMakeLists.txt cfg/dlio.yaml cfg/params.yaml package.xml src/dlio/map.cc launch/dlio.launch.py 2>/dev/null || echo "clean"
```

Expected: `clean`

---

### Task 3: Resolve include/dlio/odom.h

Take the hiking branch as base, then add the four gazebo external-odom member declarations.

**Files:**
- Modify: `include/dlio/odom.h`

- [ ] **Step 1: Check out hiking branch version**

```bash
git checkout feature/ros2_hiking -- include/dlio/odom.h
```

- [ ] **Step 2: Add `callbackExternalOdom` to the private method declarations**

Find this block in `include/dlio/odom.h`:
```cpp
  void initializeDLIO();

  bool getNextPose();
```

Replace with:
```cpp
  void initializeDLIO();

  void callbackExternalOdom(nav_msgs::msg::Odometry::SharedPtr odom);  // NOLINT(performance-unnecessary-value-param)

  bool getNextPose();
```

- [ ] **Step 3: Add `external_odom_sub` and `external_odom_cb_group` to the subscriber members**

Find this line:
```cpp
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group, imu_cb_group, reset_srv_cb_group_;
```

Replace with:
```cpp
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group, imu_cb_group, reset_srv_cb_group_, external_odom_cb_group;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr external_odom_sub;
```

- [ ] **Step 4: Add external odom pose fields after `imuPose`**

Find this block:
```cpp
  Pose lidarPose;
  Pose imuPose;

  // Metrics
```

Replace with:
```cpp
  Pose lidarPose;
  Pose imuPose;

  // External odometry (replaces IMU for initialization and T_prior in sim mode)
  Pose externalOdomPose;
  Pose prevExternalOdomPose;
  bool first_external_odom_received;
  std::mutex mtx_external_odom;

  // Metrics
```

- [ ] **Step 5: Stage the file**

```bash
git add include/dlio/odom.h
```

- [ ] **Step 6: Confirm no conflict markers**

```bash
grep -c "<<<<<<" include/dlio/odom.h && echo "CONFLICT" || echo "clean"
```

Expected: `clean`

---

### Task 4: Resolve src/dlio/odom.cc — base

Take the hiking branch as the base for the whole file.

**Files:**
- Modify: `src/dlio/odom.cc`

- [ ] **Step 1: Check out hiking branch version**

```bash
git checkout feature/ros2_hiking -- src/dlio/odom.cc
```

---

### Task 5: odom.cc — constructor: init flag and subscription

**Files:**
- Modify: `src/dlio/odom.cc`

- [ ] **Step 1: Initialize `first_external_odom_received` in the constructor**

Find this block in the constructor (around line 133):
```cpp
  this->first_imu_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
```

Replace with:
```cpp
  this->first_imu_received = false;
  this->first_external_odom_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
```

- [ ] **Step 2: Add external odom subscription after the IMU subscription block**

Find this block (the end of the IMU subscription setup):
```cpp
  this->imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", imu_qos,
      std::bind(&dlio::OdomNode::callbackImu, this, std::placeholders::_1), imu_sub_opt);

  this->reset_srv_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
```

Replace with:
```cpp
  this->imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", imu_qos,
      std::bind(&dlio::OdomNode::callbackImu, this, std::placeholders::_1), imu_sub_opt);

  this->external_odom_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto external_odom_sub_opt = rclcpp::SubscriptionOptions();
  external_odom_sub_opt.callback_group = this->external_odom_cb_group;
  this->external_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
      "external_odom", 100,
      std::bind(&dlio::OdomNode::callbackExternalOdom, this, std::placeholders::_1),
      external_odom_sub_opt);

  this->reset_srv_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
```

---

### Task 6: odom.cc — `initializeDLIO`: accept external odom as init signal

**Files:**
- Modify: `src/dlio/odom.cc`

- [ ] **Step 1: Replace the IMU-only init guard with a dual-path check**

Find this exact block:
```cpp
void dlio::OdomNode::initializeDLIO() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }

  this->dlio_initialized = true;
  std::cout << '\n' << " DLIO initialized!" << '\n';

}
```

Replace with:
```cpp
void dlio::OdomNode::initializeDLIO() {

  bool imu_ready = this->first_imu_received && this->imu_calibrated;
  bool ext_odom_ready = this->first_external_odom_received;

  if (!imu_ready && !ext_odom_ready) {
    return;
  }

  this->dlio_initialized = true;
  std::cout << '\n' << " DLIO initialized! (via "
            << (ext_odom_ready ? "external odometry" : "IMU") << ")\n";

}
```

---

### Task 7: odom.cc — `preprocessPoints`: extend scan-ready gate and T_prior fallback

**Files:**
- Modify: `src/dlio/odom.cc`

- [ ] **Step 1: Extend the first-valid-scan gate to accept external odom**

Find this block inside `preprocessPoints()`:
```cpp
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
```

Replace with:
```cpp
    // don't process scans until IMU data or external odometry is present
    if (!this->first_valid_scan) {
      bool imu_ready = false;
{
  std::lock_guard<decltype(this->mtx_imu)> lock(this->mtx_imu);
  imu_ready = !this->imu_buffer.empty() && this->imu_buffer.front().stamp >= this->scan_stamp;
}
      bool ext_odom_ready = this->first_external_odom_received;

      if (!imu_ready && !ext_odom_ready) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan
```

- [ ] **Step 2: Add external odom T_prior branch for subsequent scans**

Find this block (the IMU prior section for second scan onwards):
```cpp
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
```

Replace with:
```cpp
    } else {

      if (!this->imu_buffer.empty()) {
        // IMU prior: integrate IMU between scans
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
        frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                  this->geo.prev_vel.cast<float>(), {this->scan_stamp});

        if (frames.size() > 0) {
          this->T_prior = frames.back();
        } else {
          this->T_prior = this->T;
        }
      } else if (this->first_external_odom_received) {
        // External odom prior: delta pose from ground truth gives GICP a good initial guess
        std::unique_lock<std::mutex> lock(this->mtx_external_odom);
        Eigen::Vector3f dp = this->externalOdomPose.p - this->prevExternalOdomPose.p;
        Eigen::Quaternionf dq = this->prevExternalOdomPose.q.inverse() * this->externalOdomPose.q;
        lock.unlock();

        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3,3>(0,0) = dq.toRotationMatrix();
        delta.block<3,1>(0,3) = dp;

        this->T_prior = delta * this->T;
      } else {
        this->T_prior = this->T;
      }

    }
```

---

### Task 8: odom.cc — `performReset`: reset external odom state

**Files:**
- Modify: `src/dlio/odom.cc`

- [ ] **Step 1: Add external odom state reset in `performReset()`**

Find this line in `performReset()` (the last IMU-state reset line before the log message):
```cpp
  this->first_imu_received           = false;
  RCLCPP_INFO(this->get_logger(), "[RESET] IMU buffer cleared, IMU stamps zeroed.");
```

Replace with:
```cpp
  this->first_imu_received           = false;
  {
    std::unique_lock<std::mutex> lock(this->mtx_external_odom);
    this->first_external_odom_received = false;
    this->externalOdomPose.p    = Eigen::Vector3f::Zero();
    this->externalOdomPose.q    = Eigen::Quaternionf::Identity();
    this->prevExternalOdomPose  = this->externalOdomPose;
  }
  RCLCPP_INFO(this->get_logger(), "[RESET] IMU buffer cleared, IMU stamps zeroed, external odom state reset.");
```

---

### Task 9: odom.cc — add `callbackExternalOdom` function

**Files:**
- Modify: `src/dlio/odom.cc`

- [ ] **Step 1: Add the callback function at the end of the file**

Append to the end of `src/dlio/odom.cc` (after the last closing brace):

```cpp

void dlio::OdomNode::callbackExternalOdom(nav_msgs::msg::Odometry::SharedPtr odom)  // NOLINT(performance-unnecessary-value-param)
{
  std::unique_lock<std::mutex> lock(this->mtx_external_odom);

  this->prevExternalOdomPose = this->externalOdomPose;

  this->externalOdomPose.p = Eigen::Vector3f(
      static_cast<float>(odom->pose.pose.position.x),
      static_cast<float>(odom->pose.pose.position.y),
      static_cast<float>(odom->pose.pose.position.z));
  this->externalOdomPose.q = Eigen::Quaternionf(
      static_cast<float>(odom->pose.pose.orientation.w),
      static_cast<float>(odom->pose.pose.orientation.x),
      static_cast<float>(odom->pose.pose.orientation.y),
      static_cast<float>(odom->pose.pose.orientation.z));

  if (!this->first_external_odom_received) {
    // Seed DLIO's world pose from the absolute Gazebo ground-truth pose on first message.
    this->T = Eigen::Matrix4f::Identity();
    this->T.block<3,3>(0,0) = this->externalOdomPose.q.toRotationMatrix();
    this->T.block<3,1>(0,3) = this->externalOdomPose.p;

    this->prevExternalOdomPose = this->externalOdomPose;
    this->first_external_odom_received = true;
  }
}
```

- [ ] **Step 2: Stage odom.cc**

```bash
git add src/dlio/odom.cc
```

- [ ] **Step 3: Confirm no conflict markers in odom.cc or odom.h**

```bash
grep -c "<<<<<<" src/dlio/odom.cc src/dlio/odom_node.cc include/dlio/odom.h 2>/dev/null | grep -v ":0" || echo "clean"
```

Expected: `clean`

---

### Task 10: Create dlio_sim.launch.py

**Files:**
- Create: `launch/dlio_sim.launch.py`

- [ ] **Step 1: Create the sim launch file**

Create `launch/dlio_sim.launch.py` with this content:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    current_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    # Args
    rviz = LaunchConfiguration('rviz', default='false')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic', default='/lidar/point_cloud')
    imu_topic = LaunchConfiguration('imu_topic', default='/imu_sensor_broadcaster/imu')
    external_odom_topic = LaunchConfiguration('external_odom_topic', default='/odometry/filtered')

    declare_rviz_arg = DeclareLaunchArgument('rviz', default_value=rviz, description='Launch RViz')
    declare_pointcloud_topic_arg = DeclareLaunchArgument('pointcloud_topic', default_value=pointcloud_topic, description='Pointcloud topic name')
    declare_imu_topic_arg = DeclareLaunchArgument('imu_topic', default_value=imu_topic, description='IMU topic name')
    declare_external_odom_topic_arg = DeclareLaunchArgument(
        'external_odom_topic',
        default_value=external_odom_topic,
        description='Gazebo ground-truth odometry topic (nav_msgs/Odometry) used as GICP prior and init signal')

    # Params
    dlio_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'dlio.yaml'])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'params.yaml'])

    # Nodes
    dlio_odom_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_odom_node',
        output='screen',
        parameters=[dlio_yaml_path, dlio_params_yaml_path, {'use_sim_time': True}],
        remappings=[
            ('pointcloud', pointcloud_topic),
            ('imu', imu_topic),
            ('external_odom', external_odom_topic),
            # Prevent DLIO from publishing TF that conflicts with the Gazebo bridge
            ('/tf', '/dlio/tf_unused'),
            ('map_pose', 'dlio/odom_node/map_pose'),
            ('map_pose_inverted', 'dlio/odom_node/map_pose_inverted'),
            ('odom', 'dlio/odom_node/lidar_odom'),
            ('pose', 'dlio/odom_node/pose'),
            ('path_map', 'dlio/odom_node/path_map'),
            ('path_odom', 'dlio/odom_node/path_odom'),
            ('path_map_prop', 'dlio/odom_node/path_map_prop'),
            ('kf_pose', 'dlio/odom_node/keyframes'),
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('deskewed', 'dlio/odom_node/pointcloud/deskewed'),
            ('deskewed_not_transformed', 'dlio/odom_node/pointcloud/deskewed_not_transformed'),
            ('deskewed_and_transformed_to_map', 'dlio/odom_node/pointcloud/deskewed_and_transformed_to_map'),
            ('markers/velocity_linear', 'dlio/odom_node/markers/velocity_linear'),
            ('markers/velocity_angular', 'dlio/odom_node/markers/velocity_angular'),
            ('markers/correction', 'dlio/odom_node/markers/correction'),
            ('markers/degeneracy_directions', 'dlio/odom_node/markers/degeneracy_directions'),
        ],
        respawn=True,
    )

    dlio_map_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_map_node',
        output='screen',
        parameters=[dlio_yaml_path, dlio_params_yaml_path, {'use_sim_time': True}],
        remappings=[
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('map_pose', 'dlio/odom_node/map_pose'),
        ],
        respawn=True,
    )

    rviz_config_path = PathJoinSubstitution([current_pkg, 'launch', 'dlio.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='dlio_rviz',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        declare_rviz_arg,
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        declare_external_odom_topic_arg,
        dlio_odom_node,
        dlio_map_node,
        rviz_node,
    ])
```

- [ ] **Step 2: Stage the new file**

```bash
git add launch/dlio_sim.launch.py
```

---

### Task 11: Verify and complete the merge commit

**Files:**
- (git operations only)

- [ ] **Step 1: Confirm nothing is left unresolved**

```bash
git diff --name-only --diff-filter=U
```

Expected: empty (no output)

- [ ] **Step 2: Check overall staging status**

```bash
git status
```

Expected: all modified files staged under "Changes to be committed", no "Unmerged paths".

- [ ] **Step 3: Create the merge commit**

```bash
git commit --no-edit
```

Expected: merge commit created with auto-generated message referencing `feature/ros2_hiking`.

---

### Task 12: Build verification

- [ ] **Step 1: Build the package**

```bash
cd /home/pascalr/robot-workspaces/hike_ws && colcon build --packages-select direct_lidar_inertial_odometry 2>&1 | tail -20
```

Expected: `[direct_lidar_inertial_odometry] Build finished` with no errors.

- [ ] **Step 2: Confirm both launch files are installed**

```bash
find install/direct_lidar_inertial_odometry -name "*.launch.py" | sort
```

Expected: both `dlio.launch.py` and `dlio_sim.launch.py` appear.

- [ ] **Step 3: Smoke-check launch file syntax**

```bash
python3 -c "import launch; exec(open('launch/dlio.launch.py').read())" && echo "dlio.launch.py OK"
python3 -c "import launch; exec(open('launch/dlio_sim.launch.py').read())" && echo "dlio_sim.launch.py OK"
```

Expected: both print `OK`.
