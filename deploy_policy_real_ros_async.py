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

from unitree_sdk2py.utils.thread import RecurrentThread

# ROS接口
from examples.H10W.robot_interface import H10WInferfaceConfig, H10WInterface
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

# stats = load_stats("results/Checkpoints/dual_florence_h10w_joint_real_vr_l15r5_pre17/dataset_statistics.json")
# normalizer = Normalizer("min_max", stats)     
robot_controller = RobotController()   

@dataclasses.dataclass
class Args:
    # host: str = "192.168.0.88"
    host: str = "10.8.8.78"
    port: int = 10093
    resize_size = [224,224]
    # pretrained_path: str = "./results/Checkpoints/internvla_h10w_joint_real_vr_l500_r500/checkpoints/steps_60000_pytorch_model.pt"
    # pretrained_path: str = "./results/Checkpoints/internvla_l500_r500_w_800_finetune/checkpoints/steps_12000_pytorch_model.pt"
    # pretrained_path: str = "./results/Checkpoints/internvla_baseline_l1k_r1k_finetune/checkpoints/steps_36000_pytorch_model.pt"
    pretrained_path: str = "./results/Checkpoints/internvla_l1000_r1000_36k_finetune_no_instruction/checkpoints/steps_6000_pytorch_model.pt"



class H10W:
    def __init__(self):
        self.task_list = {
            "1": "pick the plush green dinosaur from the desk with the left hand",
            "2": "place the plush green dinosaur toy with the left hand",
            "3": "pick the plush yellow duck toy from the desk with the right hand",
            "4": "place the plush yellow duck toy on the desk with the right hand",
            # 左手抓恐龙右边的鸭子
            "5": "pick the plush yellow duck toy to the right of plush green dinosaur with the left hand",
            # 左手把黄色鸭子放到恐龙左边
            "6": "place the plush yellow duck toy to the left of plush green dinosaur with the left hand",
            # 左手抓恐龙左边的鸭子
            "7": "pick the plush yellow duck toy to the left of plush green dinosaur with the left hand",
            # 左手把鸭子放到恐龙右边
            "8": "place the plush yellow duck toy to the right of plush green dinosaur with the left hand",
            "9": "pick the plush yellow duck toy from the black desk",
            "0": "place the plush yellow duck toy on the black desk",
            "s": "pick the orange lion toy from the desk with the left hand",
            "w": "place the orange lion toy on the desk with the left hand",
            "r": "pick the plush yellow duck toy on the right with the right hand",
            "t": "place the plush yellow duck toy with the right hand with the right hand",
            "9": "pick the plush green dinosaur toy from the black desk with the right hand",
            "0": "place the plush green dinosaur toy on the black desk with the right hand",
            "a": "pick the red pepper toy from the desk with the left hand",
            "b": "place the red pepper toy on the desk with the left hand",
            "c": "pick the brown bear toy from the desk with the left hand",
            "d": "place the brown bear toy on the desk with the left hand",
            "e": "pick the carrot  on the black desk with the right hand",
            "f": "place the carrot toy on the black desk with the right hand",
            "g": "pick the yellow duck toy from the desk and place on the desk with the left hand", 
        }
        self.current_task_id = "1"  # 默认任务
        
        # 全局标志
        self.task_switch_flag = False
        self.stop_program = False
        self.infer_flag = False
        self.actions = None

        rclpy.init()
        self.interface = H10WInterface(H10WInferfaceConfig())
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self.interface)

        exec_thread = threading.Thread(target=executor.spin, daemon=True)
        exec_thread.start()

        print("[INFO] ROS Executor started. Waiting for camera topics...")

        time.sleep(1.0)
        self.args = Args()
        self.model = ModelClient(
                policy_ckpt_path=self.args.pretrained_path, # to get unnormalization stats
                host=self.args.host,
                port=self.args.port,
                image_size=self.args.resize_size,
            )
        
        self.inference_thread = RecurrentThread(interval=0.085, target=self.inference, name="inference")
        # global action_buffer,last_gripper_action
        # global interface, model
        

    def setup(self):
        self.action_buffer = []
        self.last_gripper_action = 0
        self.SMOOTH_WINDOW = 3

        self.start_keyboard_listener()
        
        # =========================================================
        # 1) 初始化 ROS 相机接口
        # =========================================================
        first_run = False
        num = 0
        # 任务执行标志
        self.inference_thread.Start()
        time.sleep(1.0)

    def on_press(self, key):
        # global task_switch_flag, stop_program, current_task_id
        try:
            if key.char == ' ':
                self.task_switch_flag = True
            elif key.char == 'q':
                self.stop_program = True

            # NEW: 数字键切换任务
            elif key.char in self.task_list.keys():
                self.current_task_id = key.char
                print(f"\n>>> Task switched to [{self.current_task_id}] : {self.task_list[self.current_task_id]}\n")
        except:
            pass

    def start_keyboard_listener(self):
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.daemon = True
        self.listener.start()
        return self.listener
    
    def run(self):
        # global interface, model
        
        # listener = keyboard.Listener(on_press=on_press)
        # listener.start()   # 非阻塞启动
        # global task_switch_flag, stop_program, current_task_id
        # global action_buffer,last_gripper_action
        # global actions, infer_flag
        try:
            while True:
                # ---- 如果按下 q，退出 ----
                if self.stop_program:
                    print("User requested shutdown. Exiting...")
                    break
                
                if self.actions is not None:
                    for i in range(self.actions.shape[0]):
                        if i > 5:
                            previous_time = time.time()
                            
                            last_action = self.actions[i]    
                            left_arm_joints = last_action[0:7].tolist()
                            # left_arm_joints = None
                            # print("=" * 30)
                            # print("left_arm_joints:", left_arm_joints)
                            right_arm_joints = last_action[8:15].tolist()
                            # print("right_arm_joints:", right_arm_joints)
                            # right_arm_joints = None
                            torso_list = None
                            control_time = 0.01
                            left_gripper=last_action[7]
                            self.action_buffer.append(left_gripper)
                            if len(self.action_buffer) > self.SMOOTH_WINDOW:
                                self.action_buffer.pop(0)
                            # #连续 K 次一致
                            if len(self.action_buffer) == self.SMOOTH_WINDOW:
                                if all(x == 1 for x in self.action_buffer):
                                    stable_gripper = 1
                                elif all(x == 0 for x in self.action_buffer):
                                    stable_gripper = 0
                                else:
                                    # 不稳定 → 不改变状态
                                    stable_gripper = None
                            else:
                                stable_gripper = None

                            # 使用上一次的 gripper 动作来维持稳定
                            if stable_gripper is not None:
                                self.last_gripper_action = stable_gripper
                            else:
                                self.last_gripper_action = self.last_gripper_action  # 保持上一帧
                            
                            right_gripper=last_action[15]
                            # right_gripper=None
                            # print("right_gripper:", right_gripper)
                            robot_controller.control_joints(
                                left_arm=left_arm_joints,
                                right_arm=right_arm_joints,
                                torso=torso_list,
                                left_gripper=self.last_gripper_action,
                                right_gripper=right_gripper,
                                control_time=control_time)
                            time.sleep(max(0, control_time - (time.time() - previous_time)))
                            print("="*20)
                            print("infer_flag = ",self.infer_flag)
                            # if self.infer_flag:
                            #     break
                            # self.infer_flag = False
                # self.infer_flag = False
        except KeyboardInterrupt:
            print("Shutting down...")

    def inference(self):
        # global actions, infer_flag
        # global interface, model
        obs_dict = self.interface.get_observations()
        if obs_dict is None:
            time.sleep(0.01)

        img_top = obs_dict["head_rgb"]["data"]
        img_left = obs_dict["left_rgb"]["data"]
        img_right = obs_dict["right_rgb"]["data"]
        
        # plt.imshow(img_top)  # BGR → RGB
        # plt.show()

        if img_top is None or img_left is None:
            print("[WARN] Missing camera image, skipping...")
            time.sleep(0.01)
        
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
        
        task_description = self.task_list[self.current_task_id]

        # robot_status = robot_controller.get_status()
        # print("robot_status:", robot_status) 
        # left_arm_joints = robot_status['leftjoint'] 
        # right_arm_joints = robot_status['rightjoint']
        # left_gripper_state = robot_status['left_gripper']
        # right_gripper_state = robot_status['right_gripper']
        
        # state = np.array(left_arm_joints + left_gripper_state + right_arm_joints + right_gripper_state)
        # state_tensor = torch.tensor(state, dtype=torch.float32)
        # state = normalizer.forward(state_tensor)
        # state = state.numpy().tolist()
        # print(state)
        
        example_dict = {
                "image": [image_camera_top, image_camera_left,
                            image_camera_right
                            ],
                "lang": task_description,
                # "state": state,
            }
        print("*" * 50)
        print(task_description)
        response = self.model.step(example=example_dict)
        
        self.actions = response["raw_actions"]
        self.infer_flag = True
  
if __name__ == "__main__":
    a = H10W()
    a.setup()
    a.run()