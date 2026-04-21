# Merge feature/ros2_hiking into feature/ros2_gazebo

**Date:** 2026-04-21
**Branch:** feature/ros2_gazebo
**Merging:** feature/ros2_hiking

## Goal

Bring the `feature/ros2_gazebo` branch up to date with all new functionality from `feature/ros2_hiking`, while preserving the Gazebo simulation workflow (external odometry replacing IMU). The result must:

- Keep the standard real-robot DLIO workflow fully functional via `dlio.launch.py`
- Provide a separate Gazebo sim entry point via a new `dlio_sim.launch.py`

## Background

Common ancestor: `0892b6c`

`feature/ros2_hiking` (16 commits since ancestor) added:
- Visualization markers (velocity linear/angular, correction, degeneracy)
- Reset/restart service (`std_srvs/Trigger`)
- Degeneracy detection with eigenvalue thresholding
- Map pose topics (`map_pose`, `map_pose_inverted`)
- Separate path topics (`path_map`, `path_odom`, `path_map_prop`)
- Map cropping (periodic sliding-window crop)
- Shutdown improvements
- New dependencies: `visualization_msgs`, `std_srvs`
- Significant refactor of `odom.cc` and `odom.h`

`feature/ros2_gazebo` (7 commits since ancestor) added:
- External odometry subscription replacing IMU as GICP prior and initialization signal
- `callbackExternalOdom` handler
- Sim-specific `dlio.launch.py` (external_odom_topic arg, use_sim_time=true, TF isolation)

## Approach: Direct Merge (Approach A)

Run `git merge feature/ros2_hiking` and resolve each conflict manually. The hiking branch is the authoritative base for all DLIO core code. Gazebo-specific additions are re-applied on top.

## Conflict Resolution Per File

| File | Strategy |
|---|---|
| `CMakeLists.txt` | Take hiking branch entirely |
| `package.xml` | Take hiking branch (`visualization_msgs`, `std_srvs`) |
| `cfg/params.yaml` | Take hiking branch (full params including map crop, viz markers, degeneracy, restart) |
| `cfg/dlio.yaml` | Take hiking branch (adaptive spaciousness params, precise extrinsics values) |
| `src/dlio/map.cc` | Take hiking branch |
| `include/dlio/odom.h` | Hiking branch base + gazebo external odom members/declaration |
| `src/dlio/odom.cc` | Hiking branch base + gazebo external odom implementation |
| `launch/dlio.launch.py` | Take hiking branch version |

## odom.h Additions (on top of hiking base)

```cpp
// In private section — subscribers
rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr external_odom_sub;
rclcpp::CallbackGroup::SharedPtr external_odom_cb_group;  // added to existing cb_group line

// In private section — declarations
void callbackExternalOdom(nav_msgs::msg::Odometry::SharedPtr odom);

// In private section — state fields (near lidarPose/imuPose)
Pose externalOdomPose;
Pose prevExternalOdomPose;
bool first_external_odom_received;
std::mutex mtx_external_odom;
```

## odom.cc Integration Points (on top of hiking base)

1. **Constructor** — subscribe to `external_odom` topic; init `first_external_odom_received = false`
2. **`getNextPose()`** — in the T_prior block, add `else if (first_external_odom_received)` branch: compute delta pose from `externalOdomPose - prevExternalOdomPose` and apply to current `T`
3. **`processPointCloud()` gate** — allow scan processing when either IMU calibration OR `first_external_odom_received` is true
4. **`initializeDLIO()`** — accept either IMU calibration completion OR `first_external_odom_received` as initialization signal; log which path triggered initialization

## Launch File Structure

### `dlio.launch.py` (unchanged from hiking branch)
Standard real-robot launch. Supports `use_sim_time` arg (default false), new topic remappings, respawn enabled.

### `dlio_sim.launch.py` (new file)
Gazebo simulation variant. Based on the hiking launch file structure with:
- `use_sim_time` hardcoded to `True`
- `external_odom_topic` arg (default: Gazebo ground truth topic)
- `external_odom` remapping on the odom node
- `/tf` → `/dlio/tf_unused` remapping (prevents conflict with Gazebo bridge TF)
- Output odom topic: `dlio/odom_node/lidar_odom` (avoids shadowing Gazebo's `/odom`)
- All new topic remappings from hiking branch included (`path_map`, `path_odom`, `map_pose`, markers, etc.)

## Non-Goals

- No changes to the sim-side (Gazebo world, robot description, bridge config)
- No new DLIO features beyond what hiking already provides
- No changes to params values beyond conflict resolution
