#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found in PATH" >&2
  exit 1
fi

if ! command -v docker compose >/dev/null 2>&1; then
  echo "docker compose not found in PATH" >&2
  exit 1
fi

mkdir -p assets/world

capture_view() {
  local bench="$1"
  local view="$2"
  local x="$3"
  local y="$4"
  local yaw="$5"
  local out="assets/world/${bench}_${view}.png"

  echo
  echo "==> Capturing ${out} @ (x=${x}, y=${y}, yaw=${yaw})"

  docker compose run --rm --no-deps \
    -v "$PWD:/workspace" \
    -w /workspace \
    pipeline bash -lc "
      set -eo pipefail
      source /opt/ros/\"\$ROS_DISTRO\"/setup.bash
      source /turtlebot_ws/install/setup.bash
      source /overlay_ws/install/setup.bash

      ros2 launch tb_worlds tb_a2_world.launch.py headless:=True x_pose:=${x} y_pose:=${y} yaw:=${yaw} >/tmp/a2_world.log 2>&1 &
      SIM_PID=\$!
      trap 'kill \$SIM_PID >/dev/null 2>&1 || true' EXIT

      # Allow Gazebo + bridges to fully start.
      sleep 10

      python3 scripts/capture_camera_once.py --topic /camera/image_raw --out \"${out}\" --timeout 40
    "
}

# Pose list tuned to face each bench and show its objects in the robot camera.
capture_view bench_01 view_01 -1.32 -1.05 -2.20
capture_view bench_01 view_02 -1.32 -0.94 -2.50

capture_view bench_02 view_01 0.08 3.60 1.57
capture_view bench_02 view_02 0.38 3.80 2.06

capture_view bench_03 view_01 6.80 4.01 3.14159
capture_view bench_03 view_02 6.80 3.40 3.14159

capture_view bench_04 view_01 4.13 -0.60 -1.57
capture_view bench_04 view_02 4.13 -0.20 -1.90

echo
echo "Done. Wrote bench screenshots under assets/world/"
