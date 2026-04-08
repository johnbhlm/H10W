#!/bin/bash
# launch_cameras.sh

echo "Starting head camera..."
ros2 launch realsense2_camera rs_launch.py \
    serial_no:="'333422304763'" \
    __ns:=/camera1 \
    camera_name:=camera1 \
    enable_color:=true \
    enable_depth:=true \
    enable_infra1:=false \
    enable_infra2:=false \
    align_depth.enable:=true \
    clip_distance:=10.0 \
    depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_profile:=640x480x30 &

while ! ros2 topic list | grep -q "/camera1/depth/image_rect_raw"; do
    sleep 0.5
done
echo "Head camera ready, starting wrist camera..."



