import dataclasses
import sys

# from envs.utils import action
sys.path.append("/home/maintenance/workspace/starVLA/")
from examples.H10W.model2h10w_interface import ModelClient
# from examples.real_robot.controller_dual import M1Inference
# from examples.RoboTwin.controller_dual import M1Inference

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import torch
import pandas as pd 
import time
import rclpy
import threading

# ROS接口
from examples.H10W.robot_interface_temp import H10WInferfaceConfig, H10WInterface
# 模型推理
# from examples.H10W_robot.controller_dual import M1Inference
#相机数据
from examples.H10W.realsense import Camera
# 机器人控制
from examples.H10W.robot_controller import RobotController
from vla.action import ActionVLA
from pynput import keyboard

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

stats = load_stats("results/Checkpoints/QwenDual_l800_r800/dataset_statistics.json")
normalizer = Normalizer("min_max", stats)     
robot_controller = RobotController()   

@dataclasses.dataclass
class Args:
    host: str = "192.168.0.88"
    port: int = 10093
    resize_size = [224,224]
    pretrained_path: str = "./results/Checkpoints/QwenDual_l800_r800/checkpoints/steps_24000_pytorch_model.pt"

# 全局标志
task_switch_flag = False
stop_program = False

task_list = {
    "1": "pick plush green dinosaur from the desk with the left hand",
    "2": "place the plush green dinosaur toy on the desk with the left hand",
    "3": "pick up the plush yellow duck toy from the desk with the right hand",
    "4": "place the plush yellow duck toy on the desk with the right hand",
    # 左手抓恐龙右边的鸭子
    "5": "pick the plush yellow duck toy to the right of plush green dinosaur with the left hand",
    # 左手把黄色鸭子放到恐龙左边
    "6": "place the plush yellow duck toy to the left of plush green dinosaur with the left hand",
    # 左手抓恐龙左边的鸭子
    "7": "pick the plush yellow duck toy to the left of plush green dinosaur with the left hand",
    # 左手把鸭子放到恐龙右边
    "8": "place the plush yellow duck toy to the right of plush green dinosaur with the left hand",
    "9": "pick up the plush yellow duck toy from the desk with the left hand",
    "0": "place the plush yellow duck toy on the desk with the left hand",
    "a": "pick up the orange lion toy from the desk",
    "b": "place the orange lion toy on the desk",
    "c": "pick up the brown bear toy from the desk with the left hand",
    "d": "place the brown bear toy on the desk with the left hand",
    "e": "pick up the toy on the left of yellow duck from the desk ",
    "f": "place the toy on the right of yellow duck",
    "g": "pick up the yellow duck toy with the left hand from the desk and pick up the duck with the right hand",
    
}
current_task_id = "3"  # 默认任务
def on_press(key):
    global task_switch_flag, stop_program, current_task_id
    try:
        if key.char == ' ':
            task_switch_flag = True
        elif key.char == 'q':
            stop_program = True

        # NEW: 数字键切换任务
        elif key.char in task_list.keys():
            current_task_id = key.char
            print(f"\n>>> Task switched to [{current_task_id}] : {task_list[current_task_id]}\n")

    except:
        pass

global action_buffer,last_gripper_action
action_buffer = []
last_gripper_action = 0
SMOOTH_WINDOW = 10

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener

def main():
    args = Args()
    start_keyboard_listener()
    rclpy.init()

    # =========================================================
    # 1) 初始化 ROS 相机接口
    # =========================================================
    interface = H10WInterface(H10WInferfaceConfig())

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(interface)

    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()

    print("[INFO] ROS Executor started. Waiting for camera topics...")

    time.sleep(1.0)
    model = ModelClient(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )
    first_run = False
    num = 0
    # 任务执行标志
    
    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()   # 非阻塞启动
    global task_switch_flag, stop_program, current_task_id
    global action_buffer,last_gripper_action
    try:
        while True:
            # ---- 如果按下 q，退出 ----
            if stop_program:
                print("User requested shutdown. Exiting...")
                break
            
            obs_dict = interface.get_observations()
            if obs_dict is None:
                time.sleep(0.01)
                continue
        
            img_top = obs_dict["head_rgb"]["data"]
            img_left = obs_dict["left_rgb"]["data"]
            img_right = obs_dict["right_rgb"]["data"]
            
            # plt.imshow(img_top)  # BGR → RGB
            # plt.show()

            if img_top is None or img_left is None:
                print("[WARN] Missing camera image, skipping...")
                time.sleep(0.01)
                continue
            
            # img_top,depth_455 = camera_d455.get_data()
            # img_left,depth_405_left = camera_d405_left.get_data()
            # # img_right,depth_405_right = camera_d405_right.get_data()
            
            model_input = {
                "obs_camera_left": {"color_image": img_left},  # BGR → RGB
                "obs_camera_right": {"color_image": img_right[:,:, ::-1]},  # BGR → RGB
                "obs_camera_top": {"color_image": img_top},  # BGR → RGB
            }
            
            image_camera_left = model_input["obs_camera_left"]["color_image"]
            image_camera_right = model_input["obs_camera_right"]["color_image"]
            image_camera_top = model_input["obs_camera_top"]["color_image"]
            
            # Ensure images are numpy arrays
            if not isinstance(image_camera_left, np.ndarray):
                image_camera_left = np.array(image_camera_left, dtype=np.uint8)
            if not isinstance(image_camera_right, np.ndarray):
                image_camera_right = np.array(image_camera_right, dtype=np.uint8)
            if not isinstance(image_camera_top, np.ndarray):
                image_camera_top = np.array(image_camera_top, dtype=np.uint8)
            images = [image_camera_top, image_camera_left,
                      image_camera_right
                      ]
            if task_switch_flag:
                print("Task switch triggered by user. Resetting...")
                task_switch_flag = False  # 清掉标志

                # (2) 手动切换任务描述
                new_task = input("Enter new task description: ")
                print(f"Switching to new task: {new_task}")
            
            task_description = task_list[current_task_id]

            robot_status = robot_controller.get_status()
            print("robot_status:", robot_status) 
            left_arm_joints = robot_status['leftjoint'] 
            right_arm_joints = robot_status['rightjoint']
            left_gripper_state = robot_status['left_gripper']
            right_gripper_state = robot_status['right_gripper']
            
            state = np.array(left_arm_joints + left_gripper_state + right_arm_joints + right_gripper_state)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            state = normalizer.forward(state_tensor)
            state = state.numpy().tolist()
            print(state)
            
            example_dict = {
                    "image": [image_camera_top, image_camera_left,
                              image_camera_right
                              ],
                    "lang": task_description,
                    "state": state,
                }
            print("*" * 50)
            print(task_description)
            response = model.step(example=example_dict)
            
            actions = response["raw_actions"]

            for i in range(actions.shape[0]):
                previous_time = time.time()
                
                last_action = actions[i]    
                left_arm_joints = last_action[0:7].tolist()
                # left_arm_joints = None
                # print("=" * 30)
                # print("left_arm_joints:", left_arm_joints)
                right_arm_joints = last_action[8:15].tolist()
                # print("right_arm_joints:", right_arm_joints)
                # right_arm_joints = None
                torso_list = None
                control_time = 0.012
                left_gripper=last_action[7]
                action_buffer.append(left_gripper)
                if len(action_buffer) > SMOOTH_WINDOW:
                    action_buffer.pop(0)
                # #连续 K 次一致
                if len(action_buffer) == SMOOTH_WINDOW:
                    if all(x == 1 for x in action_buffer):
                        stable_gripper = 1
                    elif all(x == 0 for x in action_buffer):
                        stable_gripper = 0
                    else:
                        # 不稳定 → 不改变状态
                        stable_gripper = None
                else:
                    stable_gripper = None

                # 使用上一次的 gripper 动作来维持稳定
                if stable_gripper is not None:
                    last_gripper_action = stable_gripper
                else:
                    last_gripper_action = last_gripper_action  # 保持上一帧
                
                right_gripper=last_action[15]
                # right_gripper=None
                # print("right_gripper:", right_gripper)
                robot_controller.control_joints(
                    left_arm=left_arm_joints,
                    right_arm=right_arm_joints,
                    torso=torso_list,
                    left_gripper=last_gripper_action,
                    right_gripper=right_gripper,
                    control_time=control_time)
                time.sleep(max(0, control_time - (time.time() - previous_time)))
                
    except KeyboardInterrupt:
        print("Shutting down...")
    
if __name__ == "__main__":
    main()