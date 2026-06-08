from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    current_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    pointcloud_topic = LaunchConfiguration('pointcloud_topic')
    imu_topic = LaunchConfiguration('imu_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')
    dynamic_filter_enabled = LaunchConfiguration('dynamic_filter_enabled')
    dynamic_filter_max_range = LaunchConfiguration('dynamic_filter_max_range')
    num_threads = LaunchConfiguration('num_threads')
    pointcloud_queue_size = LaunchConfiguration('pointcloud_queue_size')
    map_crop_enabled = LaunchConfiguration('map_crop_enabled')

    dlio_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'dlio.yaml'])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, 'cfg', 'params.yaml'])

    common_output_remappings = [
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
        ('deskewed_and_transformed_to_map',
         'dlio/odom_node/pointcloud/deskewed_and_transformed_to_map'),
        ('dynamic_removed', 'dlio/odom_node/pointcloud/dynamic_removed'),
        ('markers/velocity_linear', 'dlio/odom_node/markers/velocity_linear'),
        ('markers/velocity_angular', 'dlio/odom_node/markers/velocity_angular'),
        ('markers/correction', 'dlio/odom_node/markers/correction'),
        ('markers/degeneracy_directions', 'dlio/odom_node/markers/degeneracy_directions'),
    ]

    odom_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_odom_node',
        output='screen',
        parameters=[
            dlio_yaml_path,
            dlio_params_yaml_path,
            {
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'odom/num_threads': ParameterValue(num_threads, value_type=int),
                'pointcloud/queueSize': ParameterValue(pointcloud_queue_size, value_type=int),
                'dynamic_filter/enabled': ParameterValue(dynamic_filter_enabled, value_type=bool),
                'dynamic_filter/max_range': ParameterValue(dynamic_filter_max_range, value_type=float),
                'dynamic_filter/force_removed_cloud_output': False,
                'map/save_dynamic_removed/enabled': False,
                'run_stats/enabled': False,
                'run_stats/plot_on_shutdown': False,
            },
        ],
        remappings=[
            ('pointcloud', pointcloud_topic),
            ('imu', imu_topic),
            *common_output_remappings,
        ],
        respawn=True,
        respawn_delay=2.0,
    )

    map_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_map_node',
        output='screen',
        parameters=[
            dlio_yaml_path,
            dlio_params_yaml_path,
            {
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'map/crop/enabled': ParameterValue(map_crop_enabled, value_type=bool),
                'map/save_dynamic_removed/enabled': False,
            },
        ],
        remappings=[
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('map', 'dlio/map_node/map'),
            ('map_pose', 'dlio/odom_node/map_pose'),
            ('dynamic_removed', 'dlio/odom_node/pointcloud/dynamic_removed'),
        ],
        respawn=True,
        respawn_delay=2.0,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'pointcloud_topic',
            default_value='/front_lidar/points',
            description='A2 front LiDAR PointCloud2 topic.'),
        DeclareLaunchArgument(
            'imu_topic',
            default_value='/front_lidar/imu',
            description='A2 front LiDAR IMU topic.'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use /clock. Keep false for live robot, true for bag replay.'),
        DeclareLaunchArgument(
            'dynamic_filter_enabled',
            default_value='true',
            description='Enable the production M-detector dynamic object filter.'),
        DeclareLaunchArgument(
            'dynamic_filter_max_range',
            default_value='10.0',
            description='Dynamic detection/removal range in meters; clean-map range is unaffected.'),
        DeclareLaunchArgument(
            'num_threads',
            default_value='4',
            description='OpenMP worker count used by GICP and scan deskewing.'),
        DeclareLaunchArgument(
            'pointcloud_queue_size',
            default_value='50',
            description='Bounded post-calibration LiDAR queue for transient processing spikes.'),
        DeclareLaunchArgument(
            'map_crop_enabled',
            default_value='true',
            description='Crop the persistent map around the robot for live memory control.'),
        odom_node,
        map_node,
    ])
