import dataclasses
import time
import sys
sys.path.append("/home/diana/intern-vla_test/starVLA/")

# sys.path.append("/home/maintenance/vla_ws/InternVLA-M1/")
# from examples.RoboTwin.model2libero_interface import M1Inference
# from examples.real_robot.controller_dual import M1Inference
# from examples.RoboTwin.controller_dual import M1Inference
from examples.H10W.model2h10w_interface import ModelClient
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import torch
import pandas as pd 

# ROS接口
# from examples.H10W_robot.robot_interface import H10WInferfaceConfig, H10WInterface
# 模型推理
# from examples.H10W_robot.controller_dual import M1Inference
#相机数据
from examples.H10W.realsense import Camera
# 机器人控制
from examples.H10W.robot_controller import RobotController
from pynput import keyboard

def read_frame(cap):
    """读取一帧，如果失败就循环播放"""
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame

class Normalizer:
    def __init__(self, mode: str, statistics: dict):
        assert mode in ["mean_std", "min_max", "q99"]
        self.mode = mode
        # convert lists to tensors
        self.mean = torch.tensor(statistics["mean"], dtype=torch.float32)
        self.std = torch.tensor(statistics["std"], dtype=torch.float32)
        self.min = torch.tensor(statistics["min"], dtype=torch.float32)
        self.max = torch.tensor(statistics["max"], dtype=torch.float32)
        self.q01 = torch.tensor(statistics.get("q01", []), dtype=torch.float32)
        self.q99 = torch.tensor(statistics.get("q99", []), dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        if self.mode == "mean_std":
            return (x - self.mean) / (self.std + 1e-8)

        elif self.mode == "min_max":
            return (x - self.min) / (self.max - self.min + 1e-8) * 2 - 1

        elif self.mode == "q99":
            clipped = torch.clamp(x, self.q01, self.q99)
            return clipped / (self.q99 + 1e-8)

        else:
            raise ValueError(f"Unknown mode {self.mode}")
        
def load_stats(path):
    with open(path, "r") as f:
        stats = json.load(f)
    return stats["h10w"]["state"] 

video_top   = cv2.VideoCapture("/home/diana/intern-vla_test/starVLA/playground/data/lerobot_dataset_186_r/videos/chunk-000/observation.images.camera_high/episode_000001_h264.mp4")  
video_left  = cv2.VideoCapture("/home/diana/intern-vla_test/starVLA/playground/data/lerobot_dataset_186_r/videos/chunk-000/observation.images.camera_left_wrist/episode_000001_h264.mp4")
video_right = cv2.VideoCapture("/home/diana/intern-vla_test/starVLA/playground/data/lerobot_dataset_186_r/videos/chunk-000/observation.images.camera_right_wrist/episode_000001_h264.mp4")
df = pd.read_parquet("/home/diana/intern-vla_test/starVLA/playground/data/lerobot_dataset_186_r/data/chunk-000/episode_000001.parquet")
states = np.stack(df["observation.state"].apply(np.array).to_numpy())

# robot_controller = RobotController()  
 
@dataclasses.dataclass
class Args:
    host: str = "192.168.1.77"
    port: int = 10093
    resize_size = [224,224]
    pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_fixed_with_oft_vlm_pre50k"

stats = load_stats("/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_fixed_with_oft_vlm_pre50k/dataset_statistics.json")
normalizer = Normalizer("min_max", stats) 

num = 0
def main():
    args = Args()
    model = ModelClient(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    robot_controller = RobotController() 

    try:
        while True:
            frame_left  = read_frame(video_left)
            frame_right = read_frame(video_right)
            frame_top   = read_frame(video_top)
            
            # plt.imshow(frame_top[..., ::-1])  # BGR → RGB
            # plt.show()
        
            img_top = frame_top
            img_left = frame_left
            img_right = frame_right
              
            stats = load_stats("/home/maintenance/vla_ws/InternVLA-M1/results/Checkpoints/act_freezeqwen_h10w_joint_real_ditb_state_left_all_data/dataset_statistics.json")
            normalizer = Normalizer("min_max", stats)
            # robot_status = robot_controller.get_status()
            # print("robot_status:", robot_status) 
            # left_arm_joints = robot_status['leftjoint'] 
            # right_arm_joints = robot_status['rightjoint']
            # left_gripper_state = robot_status['left_gripper']
            # right_gripper_state = robot_status['right_gripper']
            
            # # state = np.array(left_arm_joints + left_gripper + right_arm_joints + right_gripper)
            state = np.array(left_arm_joints + left_gripper_state)
            # state = np.array(left_arm_joints) 
            # state_ = states[num-1]
            # state = np.concatenate([state_[:7], state_[-1:]], axis=0)
            # print("=" * 30)
            # print(state)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            state = normalizer.forward(state_tensor)
            state = state.numpy().tolist()
            print(state)

            example = {
                    "image": images,
                    "lang": task_description,
                    "state": state,
                }
            
            
            response = model.step(example)
            actions = response["raw_actions"]
            print(actions)
            
    except KeyboardInterrupt:
        print("Shutting down...")
 
    
if __name__ == "__main__":
    main()