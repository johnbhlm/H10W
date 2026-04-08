import dataclasses
import sys
sys.path.append("/home/diana/intern-vla_test/starVLA/")
sys.stdout.flush()

import time
import threading
import logging
import numpy as np
import rclpy
import tyro
import torch
import json
import cv2

from loguru import logger

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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ===============================
# 参数
# ===============================
CONTROL_DT = 0.01
# CONTROL_DT = 0.008
SMOOTH_WINDOW = 10
GRASP_CONFIRM_STEPS = 500
TASK_TIMEOUT = 120.0  # 2 分钟

# ===============================
# gripper smoothing buffer
# ===============================
left_action_buffer = []
right_action_buffer = []

last_left_gripper = 0
last_right_gripper = 0

left_hand_holding = False
right_hand_holding = False

left_grasp_counter = 0
right_grasp_counter = 0

allow_left_release = False
allow_right_release = False

left_place_done = False
right_place_done = False

left_item_name = ""
right_item_name = ""

use_left = False
use_right = False

# ===============================
# 判断是否需要冻结某侧图像
# ===============================
freeze_left_image = False
freeze_right_image = False

def smooth_gripper(buffer, value):
    buffer.append(value)
    if len(buffer) > SMOOTH_WINDOW:
        buffer.pop(0)

    if len(buffer) < SMOOTH_WINDOW:
        return None

    if all(x == 1 for x in buffer):
        return 1
    if all(x == 0 for x in buffer):
        return 0
    return None

def resolve_gripper(stable, last, holding, allow_release):
    if holding and not allow_release:
        return 1
    if stable is not None:
        return stable
    return last

frozen_left_rgb = cv2.imread("/home/diana/intern-vla_test/starVLA/debug_images/left.png")
frozen_right_rgb = cv2.imread("/home/diana/intern-vla_test/starVLA/debug_images/right.png")

@dataclasses.dataclass
class Args:
    # host: str = "10.8.26.93"
    host: str = "10.8.26.53"
    # host: str = "192.168.1.77"
    port: int = 10093
    resize_size = [224, 224]
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/cotrain_l1k_r1k_fixed_pre9k"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_pre82k5"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_pre77k5"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_fixed_with_oft_vlm_pre50k"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_baseline_pre85k"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_fixed_update_round_table"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_with_oft_cotrain"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_mee_pre40k"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/internvla_fulldata_fixed_with_oft_vlm"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_fixed_with_oft_vlm"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_fixed_pre16k"
    #pretrained_path: str = "/home/diana/intern-vla_debug/starVLA/results/Checkpoints/internvla_full_data_finetune_pre18k9k"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_no_round_pre85k"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_baseline_no_round"
    # pretrained_path: str = "./results/Checkpoints/gr00t_baseline_pre85k"
    pretrained_path: str = "./results/Checkpoints/gr00t_baseline_new"
    # pretrained_path: str = "./results/Checkpoints/gr00t_baseline_no_state_new"
    

stats = load_stats("/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_baseline_new/dataset_statistics.json")
normalizer = Normalizer("min_max", stats) 

def wait_for_goal_handle(interface, timeout=2.0):
    start = time.time()
    while time.time() - start < timeout:
        with interface._lock:
            if interface._current_goal_handle is not None:
                return True
        time.sleep(0.005)
    return False

def main(args: Args):
    global left_hand_holding, right_hand_holding
    global allow_left_release, allow_right_release
    global last_left_gripper, last_right_gripper
    global left_grasp_counter, right_grasp_counter
    global left_place_done, right_place_done
    global left_item_name,right_item_name
    global use_left, use_right
    global freeze_left_image,freeze_right_image
    
    # ===============================
    # frozen wrist image buffer
    # ===============================

    hold_left_joint = None
    hold_right_joint = None

    rclpy.init()

    robot_controller = RobotController()
    interface = H10WInterface(H10WInferfaceConfig(),robot_controller)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(interface)
    threading.Thread(target=executor.spin, daemon=True).start()
    print("[INFO] ROS Executor started. Waiting for camera topics...")
    time.sleep(1.0)
    
    model = ModelClient(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    task_complate = False
    action_finished = False
    FINISH_HOLD_TIME = 0.5  # 连续 0.5 秒满足条件
    finish_start_time = None

    left_target_pose = np.array([0.60, 0.60, 1.21])

    try:
        while True:
            task_description = interface.latest_instruction
            if task_description is None:
                time.sleep(0.01)
                continue
            
            if not interface.new_task_flag:
                time.sleep(0.01)
                continue

            # ===============================
            # 非 pick / place 任务直接完成
            # ===============================
            if "pick" not in task_description and "place" not in task_description:
                if not wait_for_goal_handle(interface):
                    logger.error("execute_goal_callback not started yet, cannot send result")
                    continue
                torso = 0.54
                ret = robot_controller.h10w_system.enableController(True)
                robot_controller.control_torso(torso=torso)
                ret = robot_controller.h10w_system.enableController(False)
                # if interface._current_goal_handle is not None:
                # logger.info(f"{interface._current_goal_handle}")
                result = ActionVLA.Result()
                result.success = True
                result.final_state = 3
                result.error_code = 0
                result.result_msg = "Task finished"
                with interface._lock:
                    interface._result_msg = result
                    interface._done_event.set()
                interface.new_task_flag = False
                logger.info("*" * 80)
                logger.info("Task finished")
                continue

            if "sofa" in interface.target_location:
                torso = 0.26 - 0.01
                ret = robot_controller.h10w_system.enableController(True)
                robot_controller.control_torso(torso=torso)
                ret = robot_controller.h10w_system.enableController(False)
                left_target_pose[2]= 0.9
            elif "TV-cabinet" in interface.target_location:
                torso = 0.34
                ret = robot_controller.h10w_system.enableController(True)
                robot_controller.control_torso(torso=torso)
                ret = robot_controller.h10w_system.enableController(False)
                left_target_pose[2]= 1.0
            else:
                left_target_pose[2] = 1.21
            
            if action_finished:
                task_complate = False
                finish_start_time = None
                step_counter = 0
                left_action_buffer.clear()
                right_action_buffer.clear()
                action_finished = False
                
            # ===============================
            # place 任务：只允许对应手放
            # ===============================
            if "place" in task_description:
                allow_left_release = "left" in task_description
                allow_right_release = "right" in task_description
                if "left" not in task_description and "right" not in task_description:
                    allow_left_release = True
                    allow_right_release = True
                left_place_done = False
                right_place_done = False
            
            if "left" in task_description:
                use_left = True
            if "right" in task_description:
                use_right = True

            #left/right，默认双手都可用
            if not use_left and not use_right:
                use_left = True
                use_right = True
            task_start_time = time.time()

            # if "pick" in task_description:
            #     if "left" in task_description and last_right_gripper==1:
            #         ret = robot_controller.h10w_system.enableController(True)
            #         robot_controller.control_lpose()
            #         ret = robot_controller.h10w_system.enableController(False)
            #     if "right" in task_description and last_left_gripper==1:
            #         ret = robot_controller.h10w_system.enableController(True)
            #         robot_controller.control_rpose()
            #         ret = robot_controller.h10w_system.enableController(False)
                
            # 任务开始
            ret = robot_controller.h10w_system.enableController(True)
            ret = robot_controller.h10w_motion.enableRealtimeCmd(True)
            step_counter = 0

            # if "pick" in task_description:
            if "left" in task_description:
                freeze_right_image = True  
            if "right" in task_description:
                freeze_left_image = True 

            while not task_complate:
                    
                obs = interface.get_observations()
                if obs is None:
                    continue

                # images = [
                #     obs["head_rgb"]["data"],
                #     obs["left_rgb"]["data"],
                #     obs["right_rgb"]["data"],
                # ]

                head_img = obs["head_rgb"]["data"]
                left_img = obs["left_rgb"]["data"]
                right_img = obs["right_rgb"]["data"]

                # ===============================
                # 冻结逻辑
                # ===============================
                if freeze_left_image:
                    left_img = frozen_left_rgb

                if freeze_right_image:
                    right_img = frozen_right_rgb

                images = [head_img, left_img, right_img]

                robot_status = robot_controller.get_status()
                if robot_status is None:
                    time.sleep(0.01)
                    continue
                # print("robot_status:", robot_status) 
                left_arm_joints = robot_status['leftjoint'] 
                right_arm_joints = robot_status['rightjoint']
                left_gripper_state = robot_status['left_gripper']
                right_gripper_state = robot_status['right_gripper']
                
                state = np.array(left_arm_joints + left_gripper_state + right_arm_joints + right_gripper_state)
                # print("state src",state)
                state_tensor = torch.tensor(state, dtype=torch.float32)
                state = normalizer.forward(state_tensor)
                state = state.numpy().tolist()
                # print(state)

                example = {
                    "image": images,
                    "lang": task_description,
                    "state": state,
                }

                current_item = interface.current_item  
                
                robot_status = robot_controller.get_status()
                if robot_status is None:
                    time.sleep(0.001)
                    continue
                lpose = robot_status.get("leftPose", None)
                rpose = robot_status.get("rightPose", None)
                if lpose is None or rpose is None:
                    return False
                
                right_target_pose = left_target_pose.copy()
                right_target_pose[1] *= -1

                dl = np.linalg.norm(np.array(lpose[:3]) - left_target_pose[:3])
                dr = np.linalg.norm(np.array(rpose[:3]) - right_target_pose[:3])
                both_close = (dl < 0.1 and dr < 0.1)

                if both_close and step_counter > 300:
                    if finish_start_time is None:
                        finish_start_time = time.time()
                    elif time.time() - finish_start_time >= FINISH_HOLD_TIME:
                        print("[INFO] Task finished by end-effector pose")
                        interface.vla_status["status"] = 3   # 3 = 完成
                        task_complate = True
                        action_finished = True
                         # ===============================
                        # 根据夹爪状态刷新 holding 状态
                        # ===============================
                        if last_left_gripper == 1:
                            left_hand_holding = True
                        else:
                            left_hand_holding = False
                            
                        
                        if last_right_gripper == 1:
                            right_hand_holding = True
                        else:
                            right_hand_holding = False
                            

                        # ===============================
                        # 统一刷新 vla_status，用最终值
                        # ===============================
                        interface.vla_status["left_state"] = 1 if left_hand_holding else 0
                        interface.vla_status["right_state"] = 1 if right_hand_holding else 0
                        time.sleep(1)
                        
                        break
                # print("*" * 100)
                # print("task_description:", task_description)
                response = model.step(example)
                actions = response["raw_actions"]

                interface.vla_status["status"] = 2
                for i in range(actions.shape[0]- 0):
                    act = actions[i]
                    # for act in actions:
                    start = time.time()
                    step_counter += 1

                    # if step_counter < 15:
                    #     continue
                    left_arm = act[0:7].tolist()
                    # left_arm[1] += 0.04
                    # left_arm[0] += 0.00174
                    left_grip_raw = int(act[7])

                    right_arm = act[8:15].tolist()
                    right_grip_raw = int(act[15])
                    
                    stable_left = smooth_gripper(left_action_buffer, left_grip_raw)
                    stable_right = smooth_gripper(right_action_buffer, right_grip_raw)
                    
                    # print("stable_left:",stable_left)
                    # print("stable_right:",stable_right)

                    # print("left_hand_holding:",left_hand_holding)
                    # print("right_hand_holding:",right_hand_holding)

                    # print("allow_left_release:",allow_left_release)
                    # print("allow_right_release:",allow_right_release)

                    last_left_gripper = resolve_gripper(
                        stable_left, last_left_gripper,
                        left_hand_holding, allow_left_release
                    )

                    last_right_gripper = resolve_gripper(
                        stable_right, last_right_gripper,
                        right_hand_holding, allow_right_release
                    )
                    
                    # print("last_left_gripper:",last_left_gripper)
                    # print("last_right_gripper:",last_right_gripper)

                    
                    robot_controller.control_joints(
                    left_arm=left_arm,
                    right_arm=right_arm,
                    left_gripper=last_left_gripper,
                    right_gripper=last_right_gripper,
                    control_time=CONTROL_DT,
                    )

                    time.sleep(max(0.0, CONTROL_DT - (time.time() - start)))
                    
                    if not left_hand_holding:
                        if last_left_gripper == 1:
                            left_grasp_counter += 1
                            # 第一次从“没 holding” → “holding”的那一刻立即绑定 item_name
                            if (not left_hand_holding) and left_grasp_counter >= 50 and left_item_name == "":
                                left_item_name = current_item           # 立即绑定物品，不等 150 步
                            if left_grasp_counter >= GRASP_CONFIRM_STEPS:
                                left_hand_holding = True
                                left_grasp_counter = GRASP_CONFIRM_STEPS
                        else:
                            left_grasp_counter = 0
                    else:
                        # 已经 holding，绝对不更新 item
                        left_grasp_counter = GRASP_CONFIRM_STEPS

                    # if allow_left_release and left_hand_holding and last_left_gripper == 0:
                    if allow_left_release and last_left_gripper == 0:
                        left_hand_holding = False
                        # allow_left_release = False
                        left_place_done = True
                        left_item_name = ""
                        left_grasp_counter = 0  

                    if not right_hand_holding:
                        if last_right_gripper == 1:
                            right_grasp_counter += 1
                            # 第一次从“没 holding” → “holding”的那一刻立即绑定 item_name
                            if (not right_hand_holding) and right_grasp_counter >= 50 and right_item_name == "":
                                right_item_name = current_item           # 立即绑定物品，不等 150 步
                            if right_grasp_counter >= GRASP_CONFIRM_STEPS:
                                right_hand_holding = True
                                right_grasp_counter = GRASP_CONFIRM_STEPS
                        else:
                            right_grasp_counter = 0
                    else:
                        right_grasp_counter = GRASP_CONFIRM_STEPS


                    # if allow_right_release and right_hand_holding and last_right_gripper == 0:
                    if allow_right_release  and last_right_gripper == 0:
                        right_hand_holding = False
                        # allow_right_release = False
                        right_place_done = True
                        right_item_name = ""
                        right_grasp_counter = 0

                    
                    interface.vla_status["left_state"] = 1 if left_hand_holding else 0
                    interface.vla_status["right_state"] = 1 if right_hand_holding else 0
                    interface.vla_status["left_item"] = left_item_name
                    interface.vla_status["right_item"] = right_item_name
            
            robot_controller.init_left()
            robot_controller.init_right()
            robot_controller.h10w_motion.enableRealtimeCmd(False)
            robot_controller.h10w_system.enableController(False)

            if interface.vla_status["status"] == 3 and interface._current_goal_handle is not None:
                result = ActionVLA.Result()
                result.success = True
                result.final_state = 3
                result.error_code = 0
                result.result_msg = "Task finished"

                with interface._lock:
                    interface._result_msg = result
                    interface._done_event.set()

                interface.new_task_flag = False
                task_complate = False
                action_finished = False
                finish_start_time = None
                step_counter = 0
                use_left = False
                use_right = False
                allow_left_release = False
                allow_right_release = False
                freeze_left_image = False
                freeze_right_image = False

    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    tyro.cli(main)
