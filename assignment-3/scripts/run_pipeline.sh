#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/"${ROS_DISTRO}"/setup.bash
source /turtlebot_ws/install/setup.bash
source /overlay_ws/install/setup.bash

echo "Launching simulation (headless)..."
ros2 launch tb_worlds tb_a2_world.launch.py headless:=True \
  x_pose:="${X_POSE}" y_pose:="${Y_POSE}" yaw:="${YAW}" >/tmp/a3_sim.log 2>&1 &
SIM_PID=$!

cleanup() {
  kill "$SIM_PID" >/dev/null 2>&1 || true
  kill "$NAV_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Launching Nav2 bringup (localization against known map)..."
ros2 launch nav2_bringup bringup_launch.py \
  slam:=False \
  map:="${MAP_YAML}" \
  params_file:="${NAV2_PARAMS}" \
  use_sim_time:=True \
  autostart:=True \
  use_composition:=False \
  use_respawn:=False >/tmp/a3_nav2.log 2>&1 &
NAV_PID=$!

sleep 12
echo "Publishing /initialpose..."
python3 /workspace/scripts/publish_initialpose.py --x "${X_POSE}" --y "${Y_POSE}" --yaw "${YAW}" --frame map

echo "Waiting for TF map -> ${BASE_FRAME} ..."
python3 /workspace/scripts/wait_for_tf.py --target map --source "${BASE_FRAME}" --timeout 45

echo "Starting detector publisher..."
exec python3 /workspace/scripts/ds685_a3_detector.py
