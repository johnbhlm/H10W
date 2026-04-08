import dataclasses
import sys

# from envs.utils import action
sys.path.append("/home/maintenance/vla_ws/InternVLA-M1/")
from examples.RoboTwin.model2libero_interface import M1Inference
# from examples.real_robot.controller_dual import M1Inference
# from examples.RoboTwin.controller_dual import M1Inference

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import torch
import pandas as pd 

#相机数据
from examples.H10W_robot.realsense import Camera
# 机器人控制
from examples.H10W_robot.robot_controller import RobotController
from pynput import keyboard

import time
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

video_top   = cv2.VideoCapture("/home/maintenance/vla_ws/InternVLA-M1/playground/Datasets/lerobot_dataset_177/videos/chunk-000/observation.images.camera_high/episode_000001_h264.mp4")  
video_left  = cv2.VideoCapture("/home/maintenance/vla_ws/InternVLA-M1/playground/Datasets/lerobot_dataset_177/videos/chunk-000/observation.images.camera_left_wrist/episode_000001_h264.mp4")
# video_right = cv2.VideoCapture("/home/maintenance/vla_ws/InternVLA-M1/playground/Datasets/lerobot_dataset_177/videos/chunk-000/observation.images.camera_right_wrist/episode_000000.mp4")
df = pd.read_parquet("/home/maintenance/vla_ws/InternVLA-M1/playground/Datasets/lerobot_dataset_177/data/chunk-000/episode_000001.parquet")
states = np.stack(df["observation.state"].apply(np.array).to_numpy())
# video_top   = cv2.VideoCapture("/home/maintenance/vla_ws/InternVLA-M1/playground/Datasets/H10W_real_data_1128/lerobot_dataset_167/videos/chunk-000/observation.images.camera_high/episode_000228_h264.mp4")  
# video_left  = cv2.VideoCapture("/home/maintenance/vla_ws/InternVLA-M1/playground/Datasets/H10W_real_data_1128/lerobot_dataset_167/videos/chunk-000/observation.images.camera_left_wrist/episode_000228_h264.mp4")
# video_right = cv2.VideoCapture("/home/maintenance/vla_ws/InternVLA-M1/examples/H10W_robot/h10w_video_left/episode_000000_right.mp4")

serial_number1_d455 = "333422304763"  # d455
serial_number1_d405_right = "218722271112" # d405-right
serial_number1_d405_left = "218622271722" # d405-left

camera_d455 = Camera(width=640, height=480, fps=30, serial_number=serial_number1_d455, name='455', calibrate_done=False)
camera_d405_right = Camera(width=640, height=480, fps=30, serial_number=serial_number1_d405_right, name='405_right', calibrate_done=False)
camera_d405_left = Camera(width=640, height=480, fps=30, serial_number=serial_number1_d405_left, name='405_left', calibrate_done=False)

robot_controller = RobotController()   

def read_frame(cap):
    """读取一帧，如果失败就循环播放"""
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame

def resize_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
    return image

def adjust_brightness(img, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 2] *= factor
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
def adjust_contrast(img, factor=1.3):
    img = img.astype(np.float32)
    img = img * factor
    return np.clip(img, 0, 255).astype(np.uint8)
def color_shift(img, r=10, g=-5, b=5):
    img = img.astype(np.int16)
    img[..., 0] += r
    img[..., 1] += g
    img[..., 2] += b
    return np.clip(img, 0, 255).astype(np.uint8)

def augment_image(img):
    """
    对输入图像 img 进行光照增强，包括随机亮度、对比度、噪声、模糊、色温偏移。
    输入 img 为 RGB 或 BGR 都可以，但建议 BGR 输入（OpenCV 默认格式）。
    返回值保持与输入相同格式。
    """
    
    aug = img.copy().astype(np.float32)

    # --- 1. 随机亮度 ---
    if random.random() < 0.6:
        factor = random.uniform(0.6, 1.4)
        aug = aug * factor

    # --- 2. 随机对比度 ---
    if random.random() < 0.6:
        factor = random.uniform(0.6, 1.4)
        mean = aug.mean()
        aug = (aug - mean) * factor + mean

    # --- 3. 高斯噪声 ---
    if random.random() < 0.4:
        noise = np.random.normal(0, 8, aug.shape)
        aug = aug + noise

    # --- 4. 高斯模糊 ---
    if random.random() < 0.3:
        k = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (k, k), 0)

    # --- 5. 色温变化 ---
    if random.random() < 0.5:
        shift = random.randint(-20, 20)
        b, g, r = cv2.split(aug)
        if shift > 0:
            r = np.clip(r + shift, 0, 255)
        else:
            b = np.clip(b - shift, 0, 255)
        aug = cv2.merge([b, g, r])

    # --- 6. 裁剪回图像范围 ---
    aug = np.clip(aug, 0, 255).astype(np.uint8)

    return aug

@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224,224]
    pretrained_path: str = "/home/maintenance/vla_ws/InternVLA-M1/results/Checkpoints/act_freezeqwen_h10w_joint_real_ditb_state_left_all_data/checkpoints/steps_30000_pytorch_model.pt"


# 全局标志
task_switch_flag = False
stop_program = False

task_list = {
    "1": "pick the plush green dinosaur toy from the black desk with the left hand",
    "2": "place the dinosaur with the left hand",
    "3": "pick up the yellow duck",
    "4": "place the yellow duck with the left hand",
    "5": "pick the red pepper from the black desk with the left hand",
    "6": "place the pepper with the left hand",
    "7": "pick the eggplant with the left hand",
    "8": "place the eggplant with the left hand",
    "9": "pick the orange lion toy from the black desk with the left hand",
    "0": "place the orange lion toy on the black desk with the left hand",
}

current_task_id = "1"  # 默认任务

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
    
global action_buffer,last_gripper_state,close_counter,open_counter,grip_lock
action_buffer = []
last_gripper_state = 0
SMOOTH_WINDOW = 10 
CLOSE_K = 10    # 连续多少帧为 1 就允许闭合（抓取） — 较灵敏
OPEN_K  = 50   # 连续多少帧为 0 才允许打开（放手） — 非常严格，避免误松手
grip_lock = False
close_counter = 0
open_counter = 0 
def main():
    args = Args()
    model = M1Inference(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )
    first_run = False
    num = 0
    listener = keyboard.Listener(on_press=on_press)
    listener.start()   # 非阻塞启动
    global task_switch_flag, stop_program, current_task_id,grip_lock
    global action_buffer, last_gripper_state,close_counter,open_counter
    try:
        while True:
            # ---- 如果按下 q，退出 ----
            if stop_program:
                print("User requested shutdown. Exiting...")
                break
            if first_run == False:
                # frame_left  = read_frame(video_left)
                # frame_right = read_frame(video_right)
                # frame_top   = read_frame(video_top)
                num += 1
                # frame_top = frame_top[30:-30,30:-30,:]
                first_run = True
            
            # frame_left  = augment_image(frame_left)
            # frame_right = augment_image(frame_right)
            # frame_top   = augment_image(frame_top)
            
           
            # plt.imshow(frame_top[..., ::-1])  # BGR → RGB
            # plt.show()
            # 
            # # # 转换成模型需要的格式
            # img_left  = cv2.cvtColor(frame_left,  cv2.COLOR_BGR2RGB)
            # img_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            # img_top   = cv2.cvtColor(frame_top,   cv2.COLOR_BGR2RGB)
            # img_top = frame_top
            # img_left = frame_left
            # img_right = frame_right
            
            img_top,depth_455 = camera_d455.get_data()
            img_left,depth_405_left = camera_d405_left.get_data()
            img_right,depth_405_right = camera_d405_right.get_data()
            
            stats = load_stats("/home/maintenance/vla_ws/InternVLA-M1/results/Checkpoints/act_freezeqwen_h10w_joint_real_ditb_state_left_all_data/dataset_statistics.json")
            normalizer = Normalizer("min_max", stats)
            robot_status = robot_controller.get_status()
            print("robot_status:", robot_status) 
            left_arm_joints = robot_status['leftjoint'] 
            right_arm_joints = robot_status['rightjoint']
            left_gripper_state = robot_status['left_gripper']
            right_gripper_state = robot_status['right_gripper']
            
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
            
            model_input = {
                "obs_camera_left": {"color_image": img_left[:,:, ::-1]},  # BGR → RGB
                # "obs_camera_right": {"color_image": img_right[:,:, ::-1]},  # BGR → RGB
                "obs_camera_top": {"color_image": img_top[:,:, ::-1]},  # BGR → RGB
            }
            
            image_camera_left = model_input["obs_camera_left"]["color_image"]
            # image_camera_right = model_input["obs_camera_right"]["color_image"]
            image_camera_top = model_input["obs_camera_top"]["color_image"]
            
            # Ensure images are numpy arrays
            if not isinstance(image_camera_left, np.ndarray):
                image_camera_left = np.array(image_camera_left, dtype=np.uint8)
            # if not isinstance(image_camera_right, np.ndarray):
            #     image_camera_right = np.array(image_camera_right, dtype=np.uint8)
            if not isinstance(image_camera_top, np.ndarray):
                image_camera_top = np.array(image_camera_top, dtype=np.uint8)
            images = [image_camera_top, image_camera_left, 
                    #   image_camera_right
                      ]
            
            if task_switch_flag:
                print("Task switch triggered by user. Resetting...")
                task_switch_flag = False  # 清掉标志
                global action_buffer
                action_buffer = []

                # (2) 手动切换任务描述
                new_task = input("Enter new task description: ")
                print(f"Switching to new task: {new_task}")
            
            task_description = task_list[current_task_id]
            print(f"Current Task [{current_task_id}]: {task_description}")
            # task_description = "pick the peppre from the black desk with the left hand"
            # task_description = "pick the dinosaur from the black desk with the left hand"
            # task_description = "pick up the yellow duck"
            ##############################################
            # task_description = "Pick the block with righthand and pick the hammer with lefthand"
            # task_description = TASK_ENV.get_instruction()
            obs_input = {
                            # "images": [observation["observation"]["head_camera"]["rgb"], observation["observation"]["left_camera"]["rgb"], observation["observation"]["right_camera"]["rgb"]], 
                            "images": [image_camera_top, 
                                       image_camera_left, 
                                    #    image_camera_right
                                       ], 
                            "task_description": task_description, # observation["instruction"][0],
                            "state": state,
            }
            # print("observation images shape:", [img.shape for img in obs_input["images"]])
            response = model.step(**obs_input)
            # actions = model.forward(model_input, instruction)
            actions = response["raw_actions"]
            print("Predicted raw action from InternVLA-M1:", actions)

            ##############################################
            for i in range(actions.shape[0]):
                previous_time = time.time()
               
                # # ----- 1) 读取模型输出并映射为 0/1 vote -----
                # raw_val = actions[i][7]
                # vote = 1 if float(raw_val) > 0.5 else 0  # 若模型直接输出 0/1 也适用

                # # ----- 2) 更新计数器（互斥递增） -----
                # if vote == 1:
                #     close_counter += 1
                #     open_counter = 0
                # else:  # vote == 0
                #     open_counter += 1
                #     close_counter = 0
                
                # # 检查是否为抓取任务
                # is_pick_task = ("pick" in task_description.lower())

                # # ========= 4) 进入抓取锁定（grip-lock） =========
                # if is_pick_task and last_gripper_state == 0:
                #     if close_counter >= CLOSE_K:
                #         last_gripper_state = 1     # 关闭夹爪
                #         grip_lock = True           # 开启锁定模式
                #         grip_lock_count = 0
                #         print("[Grip] CLOSED → grip-lock ENABLED")

                #         # 重置计数器
                #         close_counter = 0
                #         open_counter = 0

                # # ========= 5) 锁定模式中禁止松手 =========
                # if grip_lock:
                #     last_gripper_state = 1  # 强制锁定为“关闭”
                #     grip_lock_count += 1

                #     # if grip_lock_count >= 100:
                #     #     # 达到锁定期限后自动解除
                #     #     grip_lock = False
                #     #     print("[Grip] grip-lock RELEASED")

                #     left_gripper = 1   # 输出“关闭”指令
                # else:
                #     # ========= 6) 常规 gripper 状态机 =========
                #     if last_gripper_state == 1:
                #         # close -> open 更严格
                #         if open_counter >= OPEN_K:
                #             last_gripper_state = 0
                #             close_counter = 0
                #             open_counter = 0
                #             print("[Grip] OPENED")
                #     else:
                #         # open -> close 较宽松
                #         if close_counter >= CLOSE_K:
                #             last_gripper_state = 1
                #             close_counter = 0
                #             open_counter = 0
                #             print("[Grip] CLOSED")

                #     left_gripper = last_gripper_state
               
                
                last_gripper_action = actions[i][7]
                action_buffer.append(last_gripper_action)
                if len(action_buffer) > SMOOTH_WINDOW:
                    action_buffer.pop(0)
                #连续 K 次一致
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
                    last_gripper_state = stable_gripper
                else:
                    last_gripper_state = last_gripper_state  # 保持上一帧
                
                # 多数投票：当 buffer 满时判断
                # stable_gripper = None
                # if len(action_buffer) == SMOOTH_WINDOW:
                #     votes_for_1 = sum(action_buffer)
                #     # 多数门槛（适配奇偶 window）
                #     majority = SMOOTH_WINDOW // 2 + 1  # e.g. window=5 -> majority=3

                #     if votes_for_1 >= majority:
                #         candidate = 1
                #     elif (SMOOTH_WINDOW - votes_for_1) >= majority:
                #         candidate = 0
                #     else:
                #         candidate = None  # 平票或无多数

                #     # 只有当 candidate 有明确多数且与当前状态不同时才更新
                #     if candidate is not None and candidate != last_gripper_state:
                #         last_gripper_state = candidate
                #         # 你可以在这里打印一条日志：
                #         print(f"Gripper state changed -> {last_gripper_state} (votes_for_1={votes_for_1})")
                #     # 否则保持 last_gripper_state 不变

                last_action = actions[i]
                # last_action[7] = smooth_gripper_action
                
                left_arm_joints = last_action[0:7].tolist()
                # left_arm_joints = None
                # print("=" * 30)
                # print("left_arm_joints:", left_arm_joints)
                # right_arm_joints = last_action[8:15].tolist()
                # print("right_arm_joints:", right_arm_joints)
                right_arm_joints = None
                torso_list = None
                control_time = 0.02
                # left_gripper = last_gripper_state
                # left_gripper=last_action[7]
                # left_gripper=None
                # print("=" * 30)
                # print("left_gripper:", left_gripper)
                # right_gripper=last_action[15]
                right_gripper=None
                # print("right_gripper:", right_gripper)
                robot_controller.control_joints(
                    left_arm=left_arm_joints,
                    right_arm=right_arm_joints,
                    torso=torso_list,
                    left_gripper=last_gripper_state,
                    right_gripper=right_gripper,
                    control_time=control_time)
                time.sleep(max(0, control_time - (time.time() - previous_time)))
                frame_left  = read_frame(video_left)
                # frame_right = read_frame(video_right)
                frame_top   = read_frame(video_top)
                num+=1
    except KeyboardInterrupt:
        print("Shutting down...")

    
if __name__ == "__main__":
    main()