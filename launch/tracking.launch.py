import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('object_tracking_lidar')
    config_file = os.path.join(pkg_dir, 'config', 'tracking_cfg.yaml')

    tracking_node = Node(
        package='object_tracking_lidar',
        executable='tracking_node',
        name='object_tracking_lidar',
        parameters=[config_file],
        output='screen',
        # remappings=[
        #     ('/lidar/points_raw', '/lidar/points_raw'),
        # ]
    )

    return LaunchDescription([
        tracking_node,
    ])
