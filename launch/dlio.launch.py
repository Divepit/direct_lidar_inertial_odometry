from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    current_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    # Args
    pointcloud_topic = LaunchConfiguration('pointcloud_topic', default='/lidar/point_cloud')
    imu_topic = LaunchConfiguration('imu_topic', default='/imu_sensor_broadcaster/imu')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    declare_pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/lidar/point_cloud',
        description='Pointcloud topic name'
    )
    declare_imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/imu_sensor_broadcaster/imu',
        description='IMU topic name'
    )
    declare_use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value=use_sim_time, description='Use /clock (sim time)')

    # Load DLIO parameters
    dlio_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "dlio.yaml"])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "params.yaml"])

    # DLIO Odometry Node
    dlio_odom_node = Node(
        package="direct_lidar_inertial_odometry",
        executable="dlio_odom_node",
        output="screen",
        parameters=[dlio_yaml_path, dlio_params_yaml_path, {'use_sim_time': use_sim_time}],
        remappings=[
            ("pointcloud", pointcloud_topic),
            ("imu", imu_topic),
            ('map_pose', 'dlio/odom_node/map_pose'),
            ("odom", "dlio/odom_node/odom"),
            ("pose", "dlio/odom_node/pose"),
            ("path", "dlio/odom_node/path"),
            ("kf_pose", "dlio/odom_node/keyframes"),
            ("kf_cloud", "dlio/odom_node/pointcloud/keyframe"),
            ("deskewed", "dlio/odom_node/pointcloud/deskewed"),
            ('deskewed_not_transformed', 'dlio/odom_node/pointcloud/deskewed_not_transformed'),
            ('markers/velocity_linear', 'dlio/odom_node/markers/velocity_linear'),
            ('markers/velocity_angular', 'dlio/odom_node/markers/velocity_angular'),
        ],
        respawn=True,
    )

    # DLIO Mapping Node
    dlio_map_node = Node(
        package="direct_lidar_inertial_odometry",
        executable="dlio_map_node",
        output="screen",
        parameters=[dlio_yaml_path, dlio_params_yaml_path, {'use_sim_time': use_sim_time}],
        remappings=[
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('map_pose', 'dlio/odom_node/map_pose'),
        ],
        respawn=True,
    )

    return LaunchDescription([
        # 1) Nodes
        dlio_odom_node,
        dlio_map_node,
        # 2) Arguments
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        declare_use_sim_time_arg,
    ])
