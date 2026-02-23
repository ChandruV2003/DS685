import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bringup_dir = get_package_share_directory("tb_worlds")

    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")

    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    yaw = LaunchConfiguration("yaw")

    declare_namespace_cmd = DeclareLaunchArgument("namespace", default_value="")
    declare_use_sim_time_cmd = DeclareLaunchArgument("use_sim_time", default_value="true")
    declare_headless_cmd = DeclareLaunchArgument("headless", default_value="True")

    declare_x_pose_cmd = DeclareLaunchArgument("x_pose", default_value="0.0")
    declare_y_pose_cmd = DeclareLaunchArgument("y_pose", default_value="0.0")
    declare_yaw_cmd = DeclareLaunchArgument("yaw", default_value="0.0")

    sim_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup_dir, "launch", "tb_world.launch.py")),
        launch_arguments={
            "namespace": namespace,
            "use_sim_time": use_sim_time,
            "headless": headless,
            "world": os.path.join(bringup_dir, "worlds", "sim_house_a2.sdf.xacro"),
            "x_pose": x_pose,
            "y_pose": y_pose,
            "yaw": yaw,
        }.items(),
    )

    ld = LaunchDescription()
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_headless_cmd)
    ld.add_action(declare_x_pose_cmd)
    ld.add_action(declare_y_pose_cmd)
    ld.add_action(declare_yaw_cmd)
    ld.add_action(sim_cmd)
    return ld

