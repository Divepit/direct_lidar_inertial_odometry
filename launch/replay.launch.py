"""Replay launch: play a hardcoded mcap, run DLIO + RViz, then auto-save map + plots.

Run with:
    ros2 launch direct_lidar_inertial_odometry replay.launch.py

What it does:
  * Plays the hardcoded MCAP below at 1x with the sim clock (use_sim_time=True).
  * Launches the A2 front DLIO odom + map nodes and RViz2 (a2_front.rviz).
  * When the bag finishes it drains, saves the map (clean/dynamic PCDs), renders
    the dynamic-removal verification image (human_removal_verify.png), then shuts
    down so the odom node writes the run_stats plots (plot_on_shutdown).

Pause / resume the replay at any time (the bag player runs as a launch process,
so use its service rather than the SPACE key):
    ros2 service call /rosbag2_player/pause           rosbag2_interfaces/srv/Pause   "{}"
    ros2 service call /rosbag2_player/resume          rosbag2_interfaces/srv/Resume  "{}"
    ros2 service call /rosbag2_player/toggle_paused   rosbag2_interfaces/srv/TogglePaused "{}"

Outputs land next to the mcap in   <bag_dir>/replay_output/{maps,run_stats} plus
human_removal_verify.png (removed dynamic points over the static map).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (ExecuteProcess, RegisterEventHandler, TimerAction,
                            IncludeLaunchDescription, LogInfo, EmitEvent)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource

# ----------------------------- HARDCODED INPUTS -----------------------------
BAG = '/home/tutuna/colcon_ws/src/summerschool_2026/outdoor_turcan_dual_lidar/a2_outdoor_turcan_dual_lidar_0.mcap'
SAVE_CLIENT = '/home/tutuna/colcon_ws/src/summerschool_2026/outdoor_turcan_dual_lidar/save_client.py'
RENDER_SCRIPT = '/home/tutuna/colcon_ws/src/direct_lidar_inertial_odometry/scripts/render_dynamic_verify.py'
SAVE_LEAF = '0.10'          # voxel leaf [m] for the saved map
BAG_START_DELAY = 5.0       # s: let nodes subscribe before the bag starts (no missed data)
DRAIN_SEC = 12.0            # s: let the odom node finish its backlog after the bag ends
SAVE_WRITE_SEC = 12.0       # s: time for the saver to write the PCDs after the trigger
VERIFY_SEC = 30.0           # s: time for the dynamic-removal verification render
# ----------------------------------------------------------------------------


def generate_launch_description():
    out_root = os.path.join(os.path.dirname(BAG), 'replay_output')
    stats_dir = os.path.join(out_root, 'run_stats')
    maps_dir = os.path.join(out_root, 'maps')
    trigger = os.path.join(out_root, '.do_save')
    done = os.path.join(out_root, '.save_done')
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    for f in (trigger, done):
        try:
            os.remove(f)
        except OSError:
            pass

    a2_launch = os.path.join(
        get_package_share_directory('direct_lidar_inertial_odometry'),
        'launch', 'a2_front.launch.py')

    # DLIO odom + map nodes + RViz (a2_front.rviz), use_sim_time=True,
    # run_stats -> stats_dir with plot_on_shutdown enabled.
    dlio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(a2_launch),
        launch_arguments={'rviz': 'true', 'output_dir': stats_dir}.items())

    # In-launch map saver: a launch sibling of the map node, so it reliably
    # discovers /save_pcd. It waits for the trigger file, then saves the map.
    saver = ExecuteProcess(
        cmd=['python3', SAVE_CLIENT, maps_dir, SAVE_LEAF],
        name='dlio_map_saver', output='screen',
        additional_env={'DLIO_SAVE_TRIGGER': trigger, 'DLIO_SAVE_DONE': done,
                        'DLIO_SAVE_SRV': '/save_pcd', 'DLIO_SAVE_WAIT': '300'})

    # Bag player: sim clock, tf remap, large read-ahead. Started after a short
    # delay so the nodes are subscribed first.
    bag_player = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', BAG,
             '-r', '1.0', '--clock', '--read-ahead-queue-size', '2000',
             '--remap', '/tf:=/tf_bag'],
        name='bag_player', output='screen')

    # On bag end: drain -> trigger save -> render verification image -> shutdown
    # (odom writes the run_stats plots on SIGINT).
    on_bag_end = RegisterEventHandler(OnProcessExit(
        target_action=bag_player,
        on_exit=[
            LogInfo(msg='[replay] Bag finished -> draining, saving map, verify image, plots...'),
            TimerAction(period=DRAIN_SEC,
                        actions=[ExecuteProcess(cmd=['bash', '-c', f'touch "{trigger}"'],
                                                output='screen')]),
            TimerAction(period=DRAIN_SEC + SAVE_WRITE_SEC,
                        actions=[LogInfo(msg='[replay] Rendering dynamic-removal verification image...'),
                                 ExecuteProcess(cmd=['python3', RENDER_SCRIPT, out_root],
                                                name='dlio_dynamic_verify', output='screen')]),
            TimerAction(period=DRAIN_SEC + SAVE_WRITE_SEC + VERIFY_SEC,
                        actions=[LogInfo(msg='[replay] Shutting down to generate run_stats plots...'),
                                 EmitEvent(event=Shutdown(reason='replay complete'))]),
        ]))

    return LaunchDescription([
        LogInfo(msg=f'[replay] bag   = {BAG}'),
        LogInfo(msg=f'[replay] maps  -> {maps_dir}'),
        LogInfo(msg=f'[replay] plots -> {stats_dir}'),
        LogInfo(msg='[replay] pause/resume: ros2 service call /rosbag2_player/toggle_paused rosbag2_interfaces/srv/TogglePaused "{}"'),
        dlio,
        saver,
        TimerAction(period=BAG_START_DELAY, actions=[bag_player]),
        on_bag_end,
    ])
