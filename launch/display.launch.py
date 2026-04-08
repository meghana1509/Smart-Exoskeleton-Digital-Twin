import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_path = get_package_share_path('exoskeleton_ai')
    urdf_file = os.path.join(package_path, 'urdf', 'exoskeleton.urdf')

    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    return LaunchDescription([
        # 1. Publishes the 3D Model
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}]
        ),
        # 2. The "Glue" (Publishes joint positions)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui'
        ),
        # 3. Opens RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )
    ])
