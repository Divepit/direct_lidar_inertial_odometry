from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch.actions import RegisterEventHandler, ExecuteProcess
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    current_pkg = FindPackageShare("direct_lidar_inertial_odometry")

    # Args
    pointcloud_topic = LaunchConfiguration('pointcloud_topic', default='/lidar/point_cloud')
    imu_topic = LaunchConfiguration('imu_topic', default='/imu_sensor_broadcaster/imu')
    external_odom_topic = LaunchConfiguration('external_odom_topic', default='/dlio/odom_node/odom')

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
    # ===========================================================================
    # SET YOUR SIMULATION ODOMETRY TOPIC HERE
    # Change default_value to your Gazebo ground truth odometry topic name.
    # Must be of type nav_msgs/msg/Odometry.
    # You can also override at launch time:
    #   ros2 launch ... external_odom_topic:=/your/gazebo/odom/topic
    # ===========================================================================
    declare_external_odom_topic_arg = DeclareLaunchArgument(
        'external_odom_topic',
        default_value='/dlio/odom_node/odom',
        description='External odometry topic (replaces IMU for GICP prior and initialization)'
    )

    # Load DLIO parameters
    dlio_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "dlio.yaml"])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, "cfg", "params.yaml"])

    # DLIO Odometry Node
    dlio_odom_node = Node(
        package="direct_lidar_inertial_odometry",
        executable="dlio_odom_node",
        output="screen",
        parameters=[dlio_yaml_path, dlio_params_yaml_path, {
            "frames/odom": "odom",
            "frames/baselink": "base_link",
            "frames/lidar": "lidar_link",
            "frames/imu": "imu_link",
            "use_sim_time": True,
        }],
        remappings=[
            ("pointcloud", pointcloud_topic),
            ("imu", imu_topic),
            ("external_odom", external_odom_topic),
            ("/tf", "/dlio/tf_unused"),  # prevent DLIO from conflicting with Gazebo bridge TF
            ('map_pose', 'dlio/odom_node/map_pose'),
            ("odom", "dlio/odom_node/lidar_odom"),
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
        output="screen",
        parameters=[dlio_yaml_path, dlio_params_yaml_path, {
            "frames/odom": "odom",
            "use_sim_time": True,
        }],
        remappings=[
            ('keyframes', 'dlio/odom_node/pointcloud/keyframe'),
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
        declare_external_odom_topic_arg,
    ])