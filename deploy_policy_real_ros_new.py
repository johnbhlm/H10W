import dataclasses
import sys
import time
import threading
from collections import deque
import torch
import numpy as np
import rclpy
from pynput import keyboard
import json
import cv2
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append("/home/diana/intern-vla_test/starVLA/")

from examples.H10W.model2h10w_interface import ModelClient
from examples.H10W.robot_interface import H10WInferfaceConfig, H10WInterface
from examples.H10W.robot_controller import RobotController
from vla.action import ActionVLA


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


# =========================
# 参数
# =========================
CONTROL_DT = 0.01
SMOOTH_WINDOW = 8

@dataclasses.dataclass
class Args:
    host: str = "10.8.26.53"
    # host: str = "192.168.0.99"
    # host: str = "192.168.1.77"
    port: int = 10093
    resize_size = [224, 224]
    # pretrained_path: str = "./results/Checkpoints/internvla_h10w_joint_real_vr_l500_r500/checkpoints/steps_60000_pytorch_model.pt"
    # pretrained_path: str = "./results/Checkpoints/cotrain_l1k_r1k_fixed_pre9k"
    # pretrained_path: str = "./results/Checkpoints/cotrain_full_data_sofa_tv_finetune"
    # pretrained_path: str = "./results/Checkpoints/internvla_no_tv_no_sofa_fixed"
    # pretrained_path: str = "./results/Checkpoints/internvla_fulldata_fixed_base10000"
    # pretrained_path: str = "./results/Checkpoints/gr00t_full_data_pre77k5"
    # pretrained_path: str = "./results/Checkpoints/gr00t_full_data_pre82k5"
    # pretrained_path: str = "./results/Checkpoints/gr00t_full_data_fixed_with_oft_vlm_pre50k"
    # pretrained_path: str = "./results/Checkpoints/gr00t_full_data_fixed_with_oft_vlm_pre50k"
    # pretrained_path: str = "./results/Checkpoints/internvla_fulldata_finetune_pre_no_sofa_tv"
    # pretrained_path: str = "./results/Checkpoints/internvla_l1000_r1000_no_instruction_preco18k9k/checkpoints/steps_6000_pytorch_model.pt"
    # pretrained_path: str = "./results/Checkpoints/internvla_1000l_1000r_algo_pre_co18k9k/checkpoints/steps_12000_pytorch_model.pt"
    # pretrained_path: str = "./results/Checkpoints/gr00t_water_bottle"
    # pretrained_path: str = "./results/Checkpoints/gr00t_full_data_fixed_update_round_table"
    # pretrained_path: str = "./results/Checkpoints/gr00t_mee_pre40k"
    pretrained_path: str = "./results/Checkpoints/gr00t_full_data_fixed_update_round_table_done_flag"
    # pretrained_path: str = "./results/Checkpoints/gr00t_action_noise_pre40k"
    # pretrained_path: str = "./results/Checkpoints/gr00t_baseline_pre85k"
    # pretrained_path: str = "./results/Checkpoints/gr00t_baseline_new"
    # pretrained_path: str = "./results/Checkpoints/gr00t_baseline_no_state_new"
    # pretrained_path: str = "./results/Checkpoints/gr00t_no_hand"
stats = load_stats("./results/Checkpoints/gr00t_full_data_fixed_update_round_table_done_flag/dataset_statistics.json")
normalizer = Normalizer("min_max", stats) 

# =========================
# 任务列表 & 键盘触发
# =========================
task_list = {
    # white desk 
    "1": "pick the plush green dinosaur toy from the white-desk with the left hand",
    "2": "place the plush green dinosaur toy on the white-desk with the left hand",
    "3": "pick the plush green dinosaur toy from the white-desk with the right hand",
    "4": "place the plush green dinosaur toy on the white-desk with the right hand",

    "5": "pick the plush yellow duck toy from the white-desk with the left hand",
    "6": "place the plush yellow duck toy on the white-desk with the left hand",
    "7": "pick the plush yellow duck toy from the white-desk with the right hand",
    "8": "place the plush yellow duck toy on the white-desk with the right hand",

    "9": "pick the plush yellow duck toy from the sofa with the right hand",
    "0": "place the plush yellow duck toy on the sofa with the right hand",
    "s": "pick the plush orange lion toy from the white-desk with the left hand",
    "w": "place the plush orange lion toy on the white-desk with the left hand",
    "r": "pick the plush orange lion toy from the white-desk with the right hand",
    "t": "place the plush orange lion toy on the white-desk with the right hand",

    # "1": "pick the plush green dinosaur toy from the white-desk",
    # "2": "place the plush green dinosaur toy on the white-desk",
    # "3": "pick the plush green dinosaur toy from the white-desk",
    # "4": "place the plush green dinosaur toy on the white-desk",

    # "5": "pick the plush yellow duck toy from the white-desk",
    # "6": "place the plush yellow duck toy on the white-desk",
    # "7": "pick the plush yellow duck toy from the white-desk",
    # "8": "place the plush yellow duck toy on the white-desk",

    # "9": "pick the plush yellow duck toy from the sofa",
    # "0": "place the plush yellow duck toy on the sofa",
    # "s": "pick the plush orange lion toy from the white-desk",
    # "w": "place the plush orange lion toy on the white-desk",
    # "r": "pick the plush orange lion toy from the white-desk",
    # "t": "place the plush orange lion toy on the white-desk",

    # # round table 
    # "1": "pick the plush green dinosaur toy from the round table with the left hand",
    # "2": "place the plush green dinosaur toy on the round table with the left hand",
    # "3": "pick the plush green dinosaur toy from the round table with the right hand",
    # "4": "place the plush green dinosaur toy on the round table with the right hand",

    # "5": "pick the plush yellow duck toy from the round table with the left hand",
    # "6": "place the plush yellow duck toy on the round table with the left hand",
    # "7": "pick the plush yellow duck toy from the round table with the right hand",
    # "8": "place the plush yellow duck toy on the round table with the right hand",

    # "s": "pick the plush orange lion toy from the round table with the left hand",
    # "w": "place the plush orange lion toy on the round table with the left hand",
    # "r": "pick the plush orange lion toy from the round table with the right hand",
    # "t": "place the plush orange lion toy on the round table with the right hand",

    # sofa
    # "1": "pick the plush green dinosaur toy from the sofa",
    # "2": "place the plush green dinosaur toy on the sofa",
    # "3": "pick the plush green dinosaur toy from the sofa",
    # "4": "place the plush green dinosaur toy on the sofa",

    # "5": "pick the plush yellow duck toy from the sofa",
    # "6": "place the plush yellow duck toy on the sofa",
    # "7": "pick the plush yellow duck toy from the sofa",
    # "8": "place the plush yellow duck toy on the sofa",

    # "s": "pick the plush orange lion toy from the sofa",
    # "w": "place the plush orange lion toy on the sofa",
    # "r": "pick the plush orange lion toy from the sofa",
    # "t": "place the plush orange lion toy on the sofa",

    # "a": "pick the plush brown dog toy from the white desk with the left hand",
    # "b": "place the plush brown dog toy from the white desk with the left hand",
    # "c": "pick the plush brown dog toy from the white desk with the right hand",
    # "d": "place the plush brown dog toy from the white desk with the right hand",
    # "e": "pick the plush yellow duck on the left",
    # "f": "pick the plush yellow duck on the right", 
    # "g": "pick the plush yellow duck toy", 
    # "k": "place the plush green dinosaur toy on the blue plate on the sofa with the left hand",
    # "h": "place the plush green dinosaur toy on the blue plate on the sofa with the right hand",
}
current_task_id = "1"  # 默认任务
task_switch_flag = False
stop_program = False

# =========================
# 键盘监听
# =========================
def on_press(key):
    global task_switch_flag, stop_program, current_task_id
    try:
        if key.char in task_list.keys():
            current_task_id = key.char
            task_switch_flag = True
            # print(f"\n>>> Task switched to [{current_task_id}] : {task_list[current_task_id]}\n")
        elif key.char == "q":
            stop_program = True
    except:
        pass

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener

# =========================
# 夹爪平滑工具
# =========================
class GripperSmoother:
    def __init__(self, window=SMOOTH_WINDOW):
        self.buffer = deque(maxlen=window)
        self.stable_value = 0

    def update(self, value):
        self.buffer.append(value)
        if len(self.buffer) == self.buffer.maxlen:
            if all(x == 1 for x in self.buffer):
                self.stable_value = 1
            elif all(x == 0 for x in self.buffer):
                self.stable_value = 0
        return self.stable_value
    
 # 机器人控制
robot_controller = RobotController()

# =========================
# 主控制循环
# =========================
def main():
    global task_switch_flag, current_task_id, stop_program

    args = Args()
    start_keyboard_listener()
    rclpy.init()

    # ROS接口
    interface = H10WInterface(H10WInferfaceConfig(),robot_controller)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(interface)
    threading.Thread(target=executor.spin, daemon=True).start()
    time.sleep(1.0)

    # 模型
    model = ModelClient(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

   

    # gripper smoother
    left_gripper_smoother = GripperSmoother()
    right_gripper_smoother = GripperSmoother()

    # print("[INFO] Starting control loop...")
    while not stop_program:
        robot_controller.h10w_motion.enableRealtimeCmd(True)

        # 键盘切换任务
        if task_switch_flag:
            task_switch_flag = False
            current_task = task_list[current_task_id]
            # print(f"[TASK] Switched to: {current_task}")
        else:
            current_task = task_list[current_task_id]

        # 获取相机观测
        obs = interface.get_observations()
        if obs is None:
            time.sleep(0.01)
            continue

        img_top = obs["head_rgb"]["data"]
        img_left = obs["left_rgb"]["data"]
        img_right = obs["right_rgb"]["data"]
        if img_top is None or img_left is None or img_right is None:
            time.sleep(0.01)
            continue
        robot_status = robot_controller.get_status()
        if robot_status is None:
            time.sleep(0.01)
            continue

        # print("robot_status:", robot_status) 
        left_arm_joints = robot_status['leftjoint'] 
        right_arm_joints = robot_status['rightjoint']
        left_gripper_state = robot_status['left_gripper']
        right_gripper_state = robot_status['right_gripper']
        left_tcp = robot_status['leftPose']
        right_tcp = robot_status['rightPose']
        # print("left_arm_joints:", left_arm_joints)
        # print("right_arm_joints:", right_arm_joints)
        state = np.array(left_arm_joints + left_gripper_state + right_arm_joints + right_gripper_state)
        # print("state:", state)
        # print("left_tcp", left_tcp)
        # print("right_tcp", right_tcp)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state = normalizer.forward(state_tensor)
        state = state.numpy().tolist()
        # print("normalized state:", state)

        # 构建模型输入
        example = {
            "image": [img_top, img_left, img_right],
            "lang": current_task,
            "state": state,
        }
        

        # 推理
        t1 = time.time()
        response = model.step(example)
        actions = response.get("raw_actions", None)
        # print("raw_actions:", actions)
        # print("*" * 80)
        # print("infer time:",time.time()-t1)

        if actions is None or len(actions) == 0:
            continue

        # 依次执行动作
        action_horizon = 16
        for a in actions[0:action_horizon]:
            # print("done_flag:", a[-1])
            # print("*" * 80)
            # 左右臂动作
            
            left_arm = a[0:7].tolist()
            # [+0.04, 0, +0.02, 0, 0, 0, 0]
            # left_arm[0] += 0.04
            # left_arm[2] += 0.02
            # left_arm = [-1.67145561, -1.13936652,  1.20873752,  1.60997681,  2.65293581,  1.66953755, 0.32388518]
            left_gripper = int(a[7])
            right_arm = a[8:15].tolist()
            # right_arm = [1.67145561, -1.13936652,  -1.20873752,  1.60997681,  -2.65293581,  1.66953755, -0.32388518]
            right_gripper = int(a[15])

            # 平滑夹爪
            stable_left_gripper = left_gripper_smoother.update(left_gripper)
            stable_right_gripper = right_gripper_smoother.update(right_gripper)

            # 控制机器人
            robot_controller.control_joints(
                left_arm=left_arm,
                right_arm=right_arm,
                torso=None,
                left_gripper=stable_left_gripper,
                right_gripper=stable_right_gripper,
                control_time=CONTROL_DT,
            )
            # robot_controller.control_pose(
            #     left_pose = [0.97090805, 0.21024241, 0.989829,  -3.09642911, 0.20732841, 2.33892322],
            #     right_pose = [ 0.97090328, -0.1902431,  0.98981893, -0.04514866, 2.93428946, 0.80268997],
            #     control_time=CONTROL_DT,
            # )

            time.sleep(max(0, CONTROL_DT - (time.time() - t1)))
            t1 = time.time()
            # time.sleep(0.01)
        # print("*" * 80)
        # print("execute time:",time.time()-t2)

if __name__ == "__main__":
    main()
