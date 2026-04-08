import dataclasses
import sys

# from envs.utils import action
sys.path.append("/home/diana/intern-vla/starVLA/")
sys.stdout.flush()
from examples.H10W.model2h10w_interface import ModelClient

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
import tyro

# ROS接口
from examples.H10W.robot_interface_temp import H10WInferfaceConfig, H10WInterface
#相机数据
from examples.H10W.realsense import Camera
# 机器人控制
from examples.H10W.robot_controller import RobotController
from vla.action import ActionVLA
from pynput import keyboard

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        
robot_controller = RobotController()   

@dataclasses.dataclass
class Args:
    host: str = "192.168.0.10"
    port: int = 10093
    resize_size = [224,224]
    # pretrained_path: str = "./results/Checkpoints/internvla_full_h10w_joint_real_vr_left_1500_pre_2.4/checkpoints/steps_60000_pytorch_model.pt"
    pretrained_path: str =""
# 全局标志
task_switch_flag = False
stop_program = False

global action_buffer,last_gripper_action
action_buffer = []
last_gripper_action = 0
SMOOTH_WINDOW = 10

def main(args: Args):
    # args = Args()
    rclpy.init()

    # =========================================================
    # 1) 初始化 ROS 相机接口
    # =========================================================
    interface = H10WInterface(H10WInferfaceConfig())
    last_action_id = interface.last_action_id
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
    
    global task_switch_flag, stop_program, current_task_id
    global action_buffer,last_gripper_action
    # last_instruction = ""
    task_complate = False
    FINISH_HOLD_TIME = 0.5  # 连续 1.0 秒满足条件
    finish_start_time = None
    target_pose = [0.600774884223938, 0.6047673225402832, 1.2116774320602417, -2.9465925693511963, -0.04620915278792381, 3.0626158714294434]
    action_finished = False
    try:
        while True:
            # ---- 如果按下 q，退出 ----
            if stop_program:
                print("User requested shutdown. Exiting...")
                break
            
            task_description = interface.latest_instruction
            if task_description is None:
                time.sleep(0.01)
                continue
            if "pick" not in task_description and "place" not in task_description:
                result_msg = ActionVLA.Result()
                result_msg.success = True
                result_msg.final_state = 3
                result_msg.error_code = 0
                result_msg.result_msg = "Task finished"
                
                # 把结果写入接口并通知 execute_callback
                with interface._lock:
                    interface._result_msg = result_msg
                    # 解除等待
                    interface._done_event.set()
                continue
            
             # 只有在有新任务且上一个动作完成时才开始推理
            if not getattr(interface, "new_task_flag", False):
                time.sleep(0.01)
                continue
            
            if "sofa" in interface.target_location or "TV-cabinet" in interface.target_location:
                torso = 0.34
                ret = robot_controller.h10w_system.enableController(True)
                robot_controller.control_torso(torso=torso)
                ret = robot_controller.h10w_system.enableController(False)
                target_pose[2]= 1.01
            else:
                target_pose[2] = 1.21
            
            if action_finished:
                task_complate = False
                finish_start_time = None
                step_counter = 0
                action_buffer.clear()
                # last_gripper_action = 0
                action_finished = False
            ret = robot_controller.h10w_system.enableController(True)
            ret = robot_controller.h10w_motion.enableRealtimeCmd(True)
            step_counter = 0
            while not task_complate:
                # task_description = interface.latest_instruction 
                obs_dict = interface.get_observations()
                if obs_dict is None or task_description is None:
                    time.sleep(0.01)
                    continue
                
                img_top = obs_dict["head_rgb"]["data"]
                img_left = obs_dict["left_rgb"]["data"]
                right_img = obs_dict["right_rgb"]["data"]
                # img_top[:249,:,:] = 255
                # plt.imshow(img_top)  # BGR → RGB
                # plt.show()

                if img_top is None or img_left is None:
                    print("[WARN] Missing camera image, skipping...")
                    time.sleep(0.01)
                    continue
                
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
                 
                current_item = interface.current_item  
                
                robot_status = robot_controller.get_status()
                if robot_status is None:
                    time.sleep(0.001)
                    continue
                ee_leftpose = robot_status["leftPose"]
                if ee_leftpose is None or len(ee_leftpose) < 3:
                    logger.warning("invalid ee_leftpose")
                    continue
                logger.info(f"ee_leftpose: {ee_leftpose}")
                logger.info("*"* 50)
                
                dist = np.linalg.norm(np.array(ee_leftpose[:3]) - np.array(target_pose[:3]))
                logger.info(f"dist: {dist}")
                # if step_counter > 800 and dist < 0.1:
                if dist < 0.1 and step_counter > 500:
                    if finish_start_time is None:
                        finish_start_time = time.time()
                    elif time.time() - finish_start_time >= FINISH_HOLD_TIME:
                        print("[INFO] Task finished by end-effector pose")
                        interface.vla_status["status"] = 3   # 3 = 完成
                        task_complate = True
                        action_finished = True
                        ret = robot_controller.h10w_motion.enableRealtimeCmd(False)
                        robot_controller.control_torso(torso=0.54)
                        ret = robot_controller.h10w_system.enableController(False)
                        break
                
                example_dict = {
                    "image": [image_camera_top, image_camera_left,image_camera_right],
                    "lang": task_description,
                    # "state": state,
                }
                
                response = model.step(example=example_dict)
            
                actions = response["raw_actions"]
                # 设置状态：开始执行
                interface.vla_status["status"] = 2
                for i in range(actions.shape[0]):
                    previous_time = time.time()
                    step_counter += 1
                    last_action = actions[i]    
                    left_arm_joints = last_action[0:7].tolist()
                    # left_arm_joints = None
                    # print("=" * 30)
                    # print("left_arm_joints:", left_arm_joints)
                    # right_arm_joints = last_action[8:15].tolist()
                    # print("right_arm_joints:", right_arm_joints)
                    right_arm_joints = None
                    torso_list = None
                    control_time = 0.01
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
                    
                    
                    # print("=" * 30)
                    # print("left_gripper:", left_gripper)
                    # left_gripper=None
                    # print("left_gripper:", left_gripper)
                    # right_gripper=last_action[15]
                    right_gripper=None
                    # print("right_gripper:", right_gripper)
                    
                    robot_controller.control_joints(
                        left_arm=left_arm_joints,
                        right_arm=right_arm_joints,
                        torso=torso_list,
                        left_gripper=last_gripper_action,
                        right_gripper=right_gripper,
                        control_time=control_time)
                    time.sleep(max(0, control_time - (time.time() - previous_time)))
        
                    # 左手 
                    if left_gripper > 0.5: 
                        interface.vla_status["left_state"] = 1 
                        interface.vla_status["left_item"] = current_item 
                    else: 
                        interface.vla_status["left_state"] = 0 
                        interface.vla_status["left_item"] = "" 
                    # 右手 if right_gripper > 0.5: interface.vla_status["right_state"] = 1 
                    # if right_gripper > 0.5: 
                    #     interface.vla_status["right_state"] = 1 
                    #     interface.vla_status["right_item"] = current_item 
                    # else: 
                    #     interface.vla_status["right_state"] = 0 
                    #     interface.vla_status["right_item"] = ""
                        
                    # robot_status = robot_controller.get_status()
                    # if robot_status is None:
                    #     time.sleep(0.001)
                    #     continue
                    # ee_leftpose = robot_status["leftPose"]
                    # if ee_leftpose is None or len(ee_leftpose) < 3:
                    #     logger.warning("invalid ee_leftpose")
                    #     continue
                    # logger.info(f"ee_leftpose: {ee_leftpose}")
                    # logger.info("*"* 50)
                    
                    # dist = np.linalg.norm(np.array(ee_leftpose[:3]) - np.array(target_pose[:3]))
                    # logger.info(f"dist: {dist}")
                    # # if step_counter > 800 and dist < 0.1:
                    # if dist < 0.1 and step_counter > 1000:
                    #     if finish_start_time is None:
                    #         finish_start_time = time.time()
                    #     elif time.time() - finish_start_time >= FINISH_HOLD_TIME:
                    #         print("[INFO] Task finished by end-effector pose")
                    #         interface.vla_status["status"] = 3   # 3 = 完成
                    #         task_complate = True
                    #         action_finished = True
                    #         ret = robot_controller.h10w_motion.enableRealtimeCmd(False)
                    #         break
            ret = robot_controller.h10w_motion.enableRealtimeCmd(False)
            ret = robot_controller.h10w_system.enableController(False)      
            # 动作完成，返回 ActionServer 结果
            if interface.vla_status["status"] == 3 and interface._current_goal_handle is not None:
                
                # if "sofa" in task_description or "TV-cabinet" in task_description:
                #     torso = 0.56
                #     robot_controller.control_torso(torso=torso)
                
                result_msg = ActionVLA.Result()
                result_msg.success = True
                result_msg.final_state = 3
                result_msg.error_code = 0
                result_msg.result_msg = "Task finished"
                # 
                # ret = robot_controller.h10w_system.enableController(False)
                
                # 把结果写入接口并通知 execute_callback
                with interface._lock:
                    interface._result_msg = result_msg
                    # 解除等待
                    interface._done_event.set()
                    # 不要同时调用 succeed()，execute_callback 返回 result，service 会发送
                    # interface._current_goal_handle = None
                task_complate = False
                finish_start_time = None
                step_counter = 0
                interface.new_task_flag = False
                
    except KeyboardInterrupt:
        print("Shutting down...")
    
if __name__ == "__main__":
    tyro.cli(main)