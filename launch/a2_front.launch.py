import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def rviz_environment():
    runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
    if not runtime_dir:
        runtime_dir = f'/tmp/runtime-{os.getuid()}'
        os.makedirs(runtime_dir, mode=0o700, exist_ok=True)
        os.chmod(runtime_dir, 0o700)

    return {
        'XDG_RUNTIME_DIR': runtime_dir,
        'DBUS_FATAL_WARNINGS': '0',
        'NO_AT_BRIDGE': '1',
        'QT_ACCESSIBILITY': '0',
        'QT_QPA_PLATFORM': 'xcb',
        'QT_X11_NO_MITSHM': '1',
    }


def generate_launch_description():
    current_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    output_dir_default = '/tmp/dlio_run_stats'
    output_dir = LaunchConfiguration('output_dir')
    rviz = LaunchConfiguration('rviz')

    dlio_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'dlio.yaml'])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'params.yaml'])
    rviz_config_path = PathJoinSubstitution([current_pkg, 'launch', 'a2_front.rviz'])

    odom_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_odom_node',
        output='screen',
        parameters=[
            dlio_yaml_path,
            dlio_params_yaml_path,
            {
                'use_sim_time': True,
                'dynamic_filter/enabled': True,
                'dynamic_filter/max_range': 10.0,
                'dynamic_filter/warmup_scans': 10,
                'dynamic_filter/static_window_scans': 8,
                'dynamic_filter/force_removed_cloud_output': True,
                'dynamic_filter/m_detector/min_history_votes': 4,
                'dynamic_filter/m_detector/case_depth_margin': 0.25,
                'dynamic_filter/m_detector/map_consistency_depth': 0.40,
                'dynamic_filter/m_detector/min_cluster_points': 120,
                'dynamic_filter/m_detector/min_track_cluster_points': 240,
                'dynamic_filter/m_detector/max_cluster_extent': 2.2,
                'dynamic_filter/m_detector/max_assoc_distance': 0.6,
                'dynamic_filter/m_detector/track_confirm_hits': 3,
                'dynamic_filter/m_detector/track_ttl_scans': 8,
                'dynamic_filter/m_detector/static_veto_ratio': 0.10,
                'map/crop/enabled': False,
                'run_stats/enabled': True,
                'run_stats/output_dir': output_dir,
                'run_stats/overwrite': True,
                'run_stats/plot_on_shutdown': True,
                'run_stats/plot_dpi': 600,
            },
        ],
        remappings=[
            ('pointcloud', '/front_lidar/points'),
            ('imu', '/front_lidar/imu'),
            ('map_pose', 'dlio/odom_node/map_pose'),
            ('map_pose_inverted', 'dlio/odom_node/map_pose_inverted'),
            ('odom', 'dlio/odom_node/odom'),
            ('pose', 'dlio/odom_node/pose'),
            ('path_map', 'dlio/odom_node/path_map'),
            ('path_odom', 'dlio/odom_node/path_odom'),
            ('path_map_prop', 'dlio/odom_node/path_map_prop'),
            ('kf_pose', 'dlio/odom_node/keyframes'),
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('deskewed', 'dlio/odom_node/pointcloud/deskewed'),
            ('deskewed_not_transformed', 'dlio/odom_node/pointcloud/deskewed_not_transformed'),
            ('deskewed_and_transformed_to_map', 'dlio/odom_node/pointcloud/deskewed_and_transformed_to_map'),
            ('dynamic_removed', 'dlio/odom_node/pointcloud/dynamic_removed'),
            ('markers/velocity_linear', 'dlio/odom_node/markers/velocity_linear'),
            ('markers/velocity_angular', 'dlio/odom_node/markers/velocity_angular'),
            ('markers/correction', 'dlio/odom_node/markers/correction'),
            ('markers/degeneracy_directions', 'dlio/odom_node/markers/degeneracy_directions'),
        ],
        respawn=False,
    )

    map_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_map_node',
        output='screen',
        parameters=[
            dlio_yaml_path,
            dlio_params_yaml_path,
            {
                'use_sim_time': True,
                'map/crop/enabled': False,
                'map/save_dynamic_removed/enabled': True,
            },
        ],
        remappings=[
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('map', 'dlio/map_node/map'),
            ('map_pose', 'dlio/odom_node/map_pose'),
            ('dynamic_removed', 'dlio/odom_node/pointcloud/dynamic_removed'),
        ],
        respawn=False,
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='dlio_a2_front_rviz',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(rviz),
        parameters=[{'use_sim_time': True}],
        additional_env=rviz_environment(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Start RViz2 with the A2 front-lidar DLIO display config.'),
        DeclareLaunchArgument(
            'output_dir',
            default_value=output_dir_default,
            description='Directory overwritten by run_stats and used by /save_pcd. Defaults under /tmp.'),
        odom_node,
        map_node,
        rviz_node,
    ])
