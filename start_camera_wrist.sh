#!/bin/bash
# launch_cameras.sh

echo "Starting left wrist camera..."
ros2 launch realsense2_camera rs_launch.py \
    serial_no:="'218622271722'" \
    __ns:=/camera2 \
    camera_name:=camera2 \
    enable_color:=true \
    enable_depth:=true \
    enable_infra1:=false \
    enable_infra2:=false \
    align_depth.enable:=true \
    clip_distance:=10.0 \
    depth_module.depth_profile:=640x480x30 \
    depth_module.color_profile:=640x480x30 &

echo "left wrist camera ready, starting wrist camera..."

echo "Starting right wrist camera..."
ros2 launch realsense2_camera rs_launch.py \
    serial_no:="'218722271112'" \
    __ns:=/camera3 \
    camera_name:=camera3 \
    enable_color:=true \
    enable_depth:=true \
    enable_infra1:=false \
    enable_infra2:=false \
    align_depth.enable:=true \
    clip_distance:=10.0 \
    depth_module.depth_profile:=640x480x30 \
    depth_module.color_profile:=640x480x30 &
    
wait

