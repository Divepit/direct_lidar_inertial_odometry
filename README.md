# Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction

#### [[ IEEE ICRA ](https://ieeexplore.ieee.org/document/10160508)] [[ arXiv ](https://arxiv.org/abs/2203.03749)] [[ Video ](https://www.youtube.com/watch?v=4-oXjG8ow10)] [[ Presentation ](https://www.youtube.com/watch?v=Hmiw66KZ1tU)]

DLIO is a new lightweight LiDAR-inertial odometry algorithm with a novel coarse-to-fine approach in constructing continuous-time trajectories for precise motion correction. It features several algorithmic improvements over its predecessor, [DLO](https://github.com/vectr-ucla/direct_lidar_odometry), and was presented at the IEEE International Conference on Robotics and Automation (ICRA) in London, UK in 2023.

<br>
<p align='center'>
    <img src="./doc/img/dlio.png" alt="drawing" width="720"/>
</p>

## Current Repository Features

This branch is a ROS 2 DLIO pipeline with several additions on top of the original odometry core:

- ROS 2 launch files for generic DLIO and an A2 front-lidar setup.
- A one-command `replay.launch.py` that plays a hardcoded MCAP (sim clock + RViz) and auto-saves the map and run-state plots when the bag ends.
- Online LiDAR-only dynamic object filtering with a clean-room M-detector-style backend.
- Dynamic-removed point accumulation in the map node for saving `dynamic_points.pcd`.
- Automatic run-state CSV export and plot generation for pose, twist, and estimated IMU biases.
- Early IMU unit detection and auto-scaling before DLIO calibration. This handles IMUs that publish acceleration in `g` and angular velocity in `deg/s` instead of ROS-standard `m/s^2` and `rad/s`.
- RViz launch environment fixes for common container/X11 DBus and Qt issues.

## Getting Started (Step By Step)

New to ROS 2 or this repo? Start here. The lab runs everything inside a ROS 2
Jazzy environment; in our case that is the perception Docker container. In the
commands below, `<workspace>` is the colcon workspace that contains this package
(in the lab container that is `/home/tutuna/colcon_ws/src`).

### 1. Open the environment

This package is developed and run inside the lab's perception container. Open a
shell in it:

```bash
docker exec -it ros2jazzy-perception bash
```

(If you run ROS 2 Jazzy natively instead, skip this step — everything below is
the same.)

### 2. Source ROS 2 and the workspace

"Sourcing" a `setup.bash` just adds packages to your shell's search path so that
`ros2 ...` can find them. You always source **two** files, in this order:

```bash
# (a) the ROS 2 system install — gives you ros2, rviz2, ros2 bag, etc.
source /opt/ros/jazzy/setup.bash

# (b) this workspace's build output — gives you DLIO and its launch files
source <workspace>/install/setup.bash
```

You must re-source in **every new terminal**. Tip: put both lines in your
`~/.bashrc` so they run automatically.

### 3. Build (first time only, or after editing C++)

```bash
cd <workspace>
source /opt/ros/jazzy/setup.bash
colcon build --packages-select direct_lidar_inertial_odometry --symlink-install
source install/setup.bash
```

`--symlink-install` links the config and launch files into the install space, so
**edits to YAML / launch files take effect with no rebuild** — you only rebuild
after changing C++ code under `src/`.

> Build it as a plain Release (the default). Do **not** add sanitizer flags such
> as `-fsanitize=undefined`: they make the node run several times slower and can
> abort it mid-run.

### 4. Run it — easiest path: the replay launch

One command replays the dataset, shows it in RViz, and produces the maps + plots
automatically:

```bash
source /opt/ros/jazzy/setup.bash
source <workspace>/install/setup.bash
ros2 launch direct_lidar_inertial_odometry replay.launch.py
```

It plays a hardcoded MCAP with the sim clock, brings up the DLIO nodes + RViz,
and when the bag ends it saves the map, renders the dynamic-removal check image,
and writes the run-state plots. Details: [Replay An MCAP End-To-End](#replay-an-mcap-end-to-end).

### 5. Find your results

Everything lands next to the dataset:

```text
<bag_dir>/replay_output/
├── maps/                      # clean_map.pcd (static map), dynamic_points.pcd, ...
├── run_stats/                 # pose / twist / IMU-bias plots (.png/.pdf) + run_stats.csv
└── human_removal_verify.png   # removed dynamic points (red) over the static map
```

Open a `.pcd` in CloudCompare or `pcl_viewer`, and the `.png` files in any image
viewer.

The rest of this README is the detailed reference: manual two-terminal runs,
saving maps on demand, tuning the dynamic filter, etc.

## Sensor Inputs

DLIO expects:

- LiDAR: `sensor_msgs/msg/PointCloud2`
- IMU: `sensor_msgs/msg/Imu`

The generic launch remaps these internal topic names:

- `pointcloud`
- `imu`

The A2 front-lidar launch uses:

- `/front_lidar/points`
- `/front_lidar/imu`

For best accuracy, set the LiDAR and IMU extrinsics in [cfg/dlio.yaml](./cfg/dlio.yaml). The LiDAR and IMU must be time-synchronized. If the bag or driver publishes IMU acceleration in `g`, the new unit checker can correct it before calibration, but it cannot fix bad time synchronization.

## Build

Use a ROS 2 Jazzy environment. The validation container used for this repository is:

```bash
rslethz/ros2jazzy.perception:latest
```

Build from the colcon workspace that contains this package:

```bash
cd <workspace>
source /opt/ros/jazzy/setup.bash
colcon build --packages-select direct_lidar_inertial_odometry --symlink-install
source install/setup.bash
```

The package's `CMakeLists.txt` already builds in Release (`-O3`), so no extra
`--cmake-args` are needed. Do not inject `-fsanitize=...` flags — they make the
node much slower and can abort it on benign overflows. `--symlink-install` lets
you edit YAML/launch files without rebuilding (rebuild only after C++ changes).

Run tests:

```bash
cd <workspace>
source /opt/ros/jazzy/setup.bash
colcon test --packages-select direct_lidar_inertial_odometry --event-handlers console_direct+
```

## Run Generic DLIO

Terminal 1:

```bash
cd <workspace>
source install/setup.bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  rviz:=true \
  use_sim_time:=true \
  pointcloud_topic:=/lidar_points \
  imu_topic:=/lidar_imu
```

Terminal 2 (remember to source ROS 2 here too — it is a new shell):

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play <bag_or_mcap_path> \
  -r 1.0 \
  --clock \
  --read-ahead-queue-size 2000 \
  --remap /tf:=/tf_bag
```

`--clock` publishes `/clock` so the nodes can run on sim time; `--remap /tf:=/tf_bag`
keeps the bag's TF from fighting the live one.

Set `rviz:=false` if you do not want RViz:

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py rviz:=false use_sim_time:=true
```

## Run The A2 Front-Lidar Setup

This launch is for bags that publish the front sensor pair on `/front_lidar/points` and `/front_lidar/imu`.

Terminal 1:

```bash
cd <workspace>
source install/setup.bash
ros2 launch direct_lidar_inertial_odometry a2_front.launch.py
```

Terminal 2 (new shell — source ROS 2 first):

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play <a2_bag_or_mcap_path> \
  -r 1.0 \
  --clock \
  --read-ahead-queue-size 2000 \
  --remap /tf:=/tf_bag
```

Tip: prefer [`replay.launch.py`](#replay-an-mcap-end-to-end) below — it does both
terminals (nodes + bag) and the saving for you in one command.

The A2 launch enables:

- `dynamic_filter/max_range:=10.0`
- `map/save_dynamic_removed/enabled:=true`
- run-state export and plot generation

By default, run-state files and plots go to:

```bash
/tmp/dlio_run_stats
```

Override that directory:

```bash
ros2 launch direct_lidar_inertial_odometry a2_front.launch.py output_dir:=/tmp/my_dlio_run
```

## Replay An MCAP End-To-End

`replay.launch.py` is a one-command, turnkey alternative to the two-terminal A2
workflow above. It plays a hardcoded MCAP with the sim clock, brings up the A2
front DLIO odom + map nodes and RViz2, and — when the bag finishes — automatically
saves the map and generates the run-state plots, then shuts down.

```bash
cd <workspace>
source install/setup.bash
ros2 launch direct_lidar_inertial_odometry replay.launch.py
```

It does the following automatically:

- Plays the hardcoded MCAP at `-r 1.0` with `--clock`, `--remap /tf:=/tf_bag`, and
  a large read-ahead queue (started a few seconds after the nodes so no data is
  missed).
- Launches the A2 front odom + map nodes and RViz2 (`a2_front.rviz`) with
  `use_sim_time:=true`.
- On bag end: drains the backlog, saves the map (`clean_map.pcd`, `dlio_map.pcd`,
  `dynamic_points.pcd` + `save_summary.txt`), renders the dynamic-removal
  verification image (`human_removal_verify.png`: removed dynamic points in red
  over the static map, plus a removed-point height profile), then shuts down so
  the odom node writes the run-state plots.

Outputs land next to the MCAP in `<bag_dir>/replay_output/{maps,run_stats}` plus
`replay_output/human_removal_verify.png`.

The verification image can also be regenerated standalone from any run's outputs:

```bash
python3 scripts/render_dynamic_verify.py <bag_dir>/replay_output
```

Set the dataset by editing the constants at the top of
[launch/replay.launch.py](./launch/replay.launch.py) (`BAG`, plus `SAVE_LEAF` and
the `*_SEC` drain/save timing).

The bag runs as a launch process, so pause/resume with the player service rather
than the SPACE key:

```bash
ros2 service call /rosbag2_player/pause          rosbag2_interfaces/srv/Pause        "{}"
ros2 service call /rosbag2_player/resume         rosbag2_interfaces/srv/Resume       "{}"
ros2 service call /rosbag2_player/toggle_paused  rosbag2_interfaces/srv/TogglePaused "{}"
```

## Save Maps

The map node exposes:

```bash
/dlio_map_node/save_pcd
```

Create an output directory first:

```bash
mkdir -p /tmp/dlio_maps
```

Save the current map:

```bash
ros2 service call /dlio_map_node/save_pcd direct_lidar_inertial_odometry/srv/SavePCD \
  "{leaf_size: 0.15, save_path: '/tmp/dlio_maps'}"
```

Call this while the DLIO launch is still running and after enough keyframes have been published.

The service always writes:

- `/tmp/dlio_maps/dlio_map.pcd`
- `/tmp/dlio_maps/clean_map.pcd`
- `/tmp/dlio_maps/save_summary.txt`

If `map/save_dynamic_removed/enabled:=true`, it also writes:

- `/tmp/dlio_maps/dlio_dynamic_removed_map.pcd`
- `/tmp/dlio_maps/dynamic_points.pcd`

`clean_map.pcd` is the persistent map built from cleaned keyframes. `dynamic_points.pcd` contains points actually suppressed from keyframes/maps by the online dynamic filter. Registration-only candidates and intermediate debug likelihood points are not supposed to be accumulated into `dynamic_points.pcd`.

For full-mission maps, disable map cropping before the run:

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  use_sim_time:=true \
  pointcloud_topic:=/lidar_points \
  imu_topic:=/lidar_imu
```

and set in [cfg/params.yaml](./cfg/params.yaml):

```yaml
map/crop/enabled: false
```

The A2 front launch already overrides map cropping off.

## Save Run-State Plots

Run-state export is controlled by these parameters in [cfg/params.yaml](./cfg/params.yaml):

```yaml
run_stats/enabled: true
run_stats/output_dir: "/tmp/dlio_run_stats"
run_stats/overwrite: true
run_stats/plot_on_shutdown: true
run_stats/plot_dpi: 600
run_stats/plot_script: ""
```

When `run_stats/enabled` is true, the odom node writes:

- `run_stats.csv`

When `run_stats/plot_on_shutdown` is true, Ctrl+C or normal node shutdown also generates:

- `pose_position.pdf`
- `pose_position.png`
- `pose_orientation_rpy.pdf`
- `pose_orientation_rpy.png`
- `twist_linear_body.pdf`
- `twist_linear_body.png`
- `twist_angular_body.pdf`
- `twist_angular_body.png`
- `bias_accel.pdf`
- `bias_accel.png`
- `bias_gyro.pdf`
- `bias_gyro.png`
- `run_stats_summary.txt`

The x-axis is time in seconds. The y-axes use physical units: meters, radians, meters per second, radians per second, `m/s^2`, and `rad/s`.

The default output directory is:

```bash
/tmp/dlio_run_stats
```

For a normal bag workflow, let the bag finish, then press Ctrl+C in the DLIO launch terminal. The odom node closes the CSV and runs the plotter during shutdown.

If the process is killed with `SIGKILL`, plots may not be generated, but rows already flushed to `run_stats.csv` remain.

## Dynamic Object Removal

Dynamic filtering is configured under `dynamic_filter/*` in [cfg/params.yaml](./cfg/params.yaml).

Common controls:

```yaml
dynamic_filter/enabled: true
dynamic_filter/min_range: 1.0
dynamic_filter/max_range: 10.0
dynamic_filter/local_radius: 30.0
dynamic_filter/force_removed_cloud_output: false
```

Important range semantics:

- `dynamic_filter/min_range` and `dynamic_filter/max_range` gate dynamic detection/removal only.
- Finite points outside this range still pass into the clean map.
- Out-of-range points should not appear in `dynamic_points.pcd`.
- The active backend is M-detector-style ego-motion-compensated temporal depth comparison, cluster filtering, and object tracking.

For deterministic dynamic-point PCD generation, use:

```yaml
map/save_dynamic_removed/enabled: true
dynamic_filter/force_removed_cloud_output: true
```

The A2 front launch sets these for map artifact generation.

## IMU Unit Auto-Scaling

The unit checker runs at the start of `OdomNode::callbackImu()` and scales the raw IMU message before:

- IMU extrinsic transform
- calibration accumulation
- gravity alignment
- bias estimation
- IMU buffering
- deskew integration
- propagation

Parameters:

```yaml
odom/imu/unit_check/enabled: true
odom/imu/unit_check/auto_scale: true
odom/imu/unit_check/min_samples: 50
odom/imu/unit_check/max_wait_sec: 0.5
odom/imu/unit_check/warn_period_sec: 2.0
odom/imu/unit_check/assume_deg_per_sec_when_accel_g: true
odom/imu/unit_check/accel_scale_override: 0.0
odom/imu/unit_check/gyro_scale_override: 0.0
```

Detection behavior:

- Raw accel norm near `9.81`: treated as ROS-standard `m/s^2`; scale remains identity.
- Raw accel norm near `1.0`: treated as `g`; accel is multiplied by gravity.
- If accel is detected as `g` and `assume_deg_per_sec_when_accel_g` is true, gyro is multiplied by `pi / 180`.
- Positive override values force the scale and still happen before calibration.

When non-identity scaling is active, the odom node prints a large orange warning periodically with the raw medians, scales, and classification.

## RViz

Both launch files include RViz environment fixes for common container/X11 issues:

```bash
DBUS_FATAL_WARNINGS=0
NO_AT_BRIDGE=1
QT_ACCESSIBILITY=0
QT_QPA_PLATFORM=xcb
QT_X11_NO_MITSHM=1
```

If RViz still fails in your environment, run DLIO without RViz:

```bash
ros2 launch direct_lidar_inertial_odometry a2_front.launch.py rviz:=false
```

and start RViz separately after fixing host/container display access.

## Citation
If you found this work useful, please cite our manuscript:

```bibtex
@article{chen2022dlio,
  title={Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction},
  author={Chen, Kenny and Nemiroff, Ryan and Lopez, Brett T},
  journal={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023},
  pages={3983-3989},
  doi={10.1109/ICRA48891.2023.10160508}
}
```

## Acknowledgements

We thank the authors of the [FastGICP](https://github.com/SMRT-AIST/fast_gicp) and [NanoFLANN](https://github.com/jlblancoc/nanoflann) open-source packages:

- Kenji Koide, Masashi Yokozuka, Shuji Oishi, and Atsuhiko Banno, “Voxelized GICP for Fast and Accurate 3D Point Cloud Registration,” in _IEEE International Conference on Robotics and Automation (ICRA)_, IEEE, 2021, pp. 11 054–11 059.
- Jose Luis Blanco and Pranjal Kumar Rai, “NanoFLANN: a C++ Header-Only Fork of FLANN, A Library for Nearest Neighbor (NN) with KD-Trees,” https://github.com/jlblancoc/nanoflann, 2014.

We would also like to thank Helene Levy and David Thorne for their help with data collection.

## License
This work is licensed under the terms of the MIT license.

<br>
<p align='center'>
    <img src="./doc/img/ucla.png" alt="drawing" width="720"/>
</p>
<p align='center'>
    <img src="./doc/img/trees.png" alt="drawing" width="720"/>
</p>
