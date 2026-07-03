from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    current_pkg = FindPackageShare("direct_lidar_inertial_odometry")

    # Args
    robot_namespace = LaunchConfiguration('robot_namespace')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic', default='/livox/lidar')
    imu_topic = LaunchConfiguration('imu_topic', default='/livox/imu')
    odom_frame = LaunchConfiguration('odom_frame', default='robot/odom')
    baselink_frame = LaunchConfiguration('baselink_frame', default='robot/base_link')
    lidar_frame = LaunchConfiguration('lidar_frame', default='robot/lidar')
    imu_frame = LaunchConfiguration('imu_frame', default='robot/imu')

    declare_robot_namespace_arg = DeclareLaunchArgument(
        'robot_namespace',
        default_value='robot',
        description='Namespace for the DLIO nodes and their topics'
    )
    declare_pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/livox/lidar',
        description='Pointcloud topic name'
    )
    declare_imu_topic_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/livox/imu',
        description='IMU topic name'
    )
    declare_odom_frame_arg = DeclareLaunchArgument(
        'odom_frame',
        default_value='robot/odom',
        description='Frame id used for odometry outputs'
    )
    declare_baselink_frame_arg = DeclareLaunchArgument(
        'baselink_frame',
        default_value='robot/base_link',
        description='Base link frame id used by DLIO'
    )
    declare_lidar_frame_arg = DeclareLaunchArgument(
        'lidar_frame',
        default_value='robot/lidar',
        description='Internal LiDAR frame id published by DLIO'
    )
    declare_imu_frame_arg = DeclareLaunchArgument(
        'imu_frame',
        default_value='robot/imu',
        description='Internal IMU frame id published by DLIO'
    )

    # Load DLIO parameters
    dlio_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "dlio.yaml"])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "params.yaml"])

    # DLIO Odometry Node
    dlio_odom_node = Node(
        package="direct_lidar_inertial_odometry",
        executable="dlio_odom_node",
        namespace=robot_namespace,
        output="screen",
        parameters=[
            dlio_yaml_path,
            dlio_params_yaml_path,
            {
                "frames/odom": odom_frame,
                "frames/baselink": baselink_frame,
                "frames/lidar": lidar_frame,
                "frames/imu": imu_frame,
                "frames/publish_sensor_tf": True,
            },
        ],
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
        ],
        respawn=True,
    )

    # DLIO Mapping Node
    dlio_map_node = Node(
        package="direct_lidar_inertial_odometry",
        executable="dlio_map_node",
        namespace=robot_namespace,
        output="screen",
        parameters=[
            dlio_yaml_path,
            dlio_params_yaml_path,
            {
                "odom/odom_frame": odom_frame,
            },
        ],
        remappings=[
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('map_pose', 'dlio/odom_node/map_pose'),
        ],
        respawn=True,
    )

    return LaunchDescription([
        # 1) Arguments
        declare_robot_namespace_arg,
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        declare_odom_frame_arg,
        declare_baselink_frame_arg,
        declare_lidar_frame_arg,
        declare_imu_frame_arg,
        # 2) Nodes
        dlio_odom_node,
        dlio_map_node,
    ])
