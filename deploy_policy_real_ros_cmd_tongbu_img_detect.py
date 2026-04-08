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

from detection_client_3d import DetectionClient3DSync

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
TASK_TIMEOUT = 40.0  # 

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
    host: str = "10.8.26.11"
    # host: str = "192.168.1.77"
    port: int = 10093
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
    # pretrained_path: str = "/home/diana/intern-vla_debug/starVLA/results/Checkpoints/internvla_full_data_finetune_pre18k9k"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_full_data_no_round_pre85k"
    # pretrained_path: str = "/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_baseline_no_round"
    # pretrained_path: str = "./results/Checkpoints/gr00t_baseline_pre85k"
    # pretrained_path: str = "./results/Checkpoints/gr00t_baseline_new"
    # pretrained_path: str = "./results/Checkpoints/gr00t_baseline_no_state_new"
    # pretrained_path: str = "./results/Checkpoints/gr00t_delta_qpos"
    # pretrained_path: str = "./results/Checkpoints/gr00t_full_data_with_unifolm_cotrain"
    # pretrained_path: str = "./results/Checkpoints/gr00t_baseline_new_pre50k"
    pretrained_path: str = "./results/Checkpoints/gr00t_data_latest_sofa_v2_finetune"
    

    if "unifolm" in pretrained_path:
        resize_size = [256, 256]
    else:
        resize_size = [224, 224]

stats = load_stats("/home/diana/intern-vla_test/starVLA/results/Checkpoints/gr00t_data_latest_sofa_v2_finetune/dataset_statistics.json")
normalizer = Normalizer("min_max", stats) 

def wait_for_goal_handle(interface, timeout=2.0):
    start = time.time()
    while time.time() - start < timeout:
        with interface._lock:
            if interface._current_goal_handle is not None:
                return True
        time.sleep(0.005)
    return False

def extract_target_object(task_description: str):
    """
    从任务文本里提取要检测的目标物体，映射到检测服务支持的名字
    """
    text = task_description.lower()

    mapping = [
        ("green dinosaur", "green dinosaur toy"),
        ("brown dog", "brown dog toy"),
        ("golden lion", "golden lion toy"),
        ("yellow duck", "yellow duck toy"),
        ("dinosaur", "green dinosaur toy"),
        ("dog", "brown dog toy"),
        ("lion", "golden lion toy"),
        ("duck", "yellow duck toy"),
    ]

    for key, value in mapping:
        if key in text:
            return value
    return None


def finish_task_with_failure(interface, msg: str, error_code: int = 1, timeout: float = 2.0):
    """
    直接结束当前任务，并通过 action 返回失败
    """
    if not wait_for_goal_handle(interface, timeout=timeout):
        logger.error(f"任务失败，但 goal handle 未就绪: {msg}")
        interface.new_task_flag = False
        return False

    result = ActionVLA.Result()
    result.success = False
    result.final_state = 3   # 失败，不要再用 3
    result.error_code = error_code
    result.result_msg = msg

    with interface._lock:
        interface._result_msg = result
        interface._done_event.set()

    interface.vla_status["status"] = 3
    interface.new_task_flag = False
    logger.error(msg)
    return True


def infer_hand_and_rewrite_instruction(
    interface,
    robot_controller,
    detector_client,
    task_description: str,
    target_location: str,
    save_debug: bool = False,
):
    """
    在真正进入 VLA 推理前，先根据 head_rgb + head_depth 判断目标在左/右。

    返回:
    {
        "success": True/False,
        "hand": "left"/"right",
        "object_name": "...",
        "new_instruction": "...",
        "detect_result": {...},
        "reason": "..."
    }
    """
    object_name = extract_target_object(task_description)
    if object_name is None:
        return {
            "success": False,
            "reason": f"无法从任务中解析目标物体: {task_description}"
        }

    # 拿一帧最新观测
    obs = interface.get_observations()
    if obs is None:
        return {
            "success": False,
            "reason": "无法获取 head_rgb/head_depth"
        }

    if "head_rgb" not in obs or "head_depth" not in obs:
        return {
            "success": False,
            "reason": "观测中缺少 head_rgb 或 head_depth"
        }

    head_rgb = obs["head_rgb"]["data"]
    head_depth = obs["head_depth"]["data"]

    if head_rgb is None or head_depth is None:
        return {
            "success": False,
            "reason": "head_rgb 或 head_depth 为空"
        }

    head_bgr = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)

    # 当前先沿用固定头部姿态参数
    # 如果后面你能从 robot_status 读到真实 pan/tilt，再替换这里
    torso_height = 0.56
    tilt_angle = -0.6
    pan_angle = 0.0

    try:
        result = detector_client.detect_hand(
            color_img=head_bgr,
            depth_img=head_depth,
            torso_height=torso_height,
            tilt_angle=tilt_angle,
            pan_angle=pan_angle,
            object_name=object_name,
            save_debug=save_debug,
        )
    except Exception as e:
        return {
            "success": False,
            "reason": f"检测服务调用失败: {e}"
        }

    if not result.get("success"):
        return {
            "success": False,
            "reason": f"未检测到目标物体: {object_name}",
            "debug": result.get("debug", {})
        }

    hand = result.get("hand", None)
    if hand not in ["left", "right"]:
        return {
            "success": False,
            "reason": f"检测结果没有有效 hand: {result}"
        }

    # 新增：根据当前双手占用情况做最终仲裁
    choose_ret = choose_available_hand(hand)
    if not choose_ret["success"]:
        return {
            "success": False,
            "reason": choose_ret["reason"],
            "debug": result.get("debug", {})
        }

    hand = choose_ret["hand"]

    # 改写指令：只在原指令没有 hand 信息时添加
    text = task_description
    if "left" not in text.lower() and "right" not in text.lower():
        new_instruction = f"{text} with the {hand} hand"
    else:
        new_instruction = text

    return {
        "success": True,
        "hand": hand,
        "object_name": object_name,
        "new_instruction": new_instruction,
        "detect_result": result,
    }


def infer_place_hand_and_rewrite_instruction(task_description: str):
    """
    对 place 任务，如果没有明确左右手，则根据当前 holding 状态补全左右手。

    返回:
    {
        "success": True/False,
        "hand": "left"/"right",
        "object_name": "...",
        "new_instruction": "...",
        "reason": "..."
    }
    """
    global left_hand_holding, right_hand_holding
    global left_item_name, right_item_name

    object_name = extract_target_object(task_description)
    if object_name is None:
        return {
            "success": False,
            "reason": f"无法从 place 指令中解析目标物体: {task_description}"
        }

    # 统一成简短 key，方便和 current_item / item_name 对齐
    object_key_map = {
        "green dinosaur toy": "dinosaur",
        "brown dog toy": "dog",
        "golden lion toy": "lion",
        "yellow duck toy": "duck",
    }
    object_key = object_key_map.get(object_name, object_name)

    left_match = left_hand_holding and (left_item_name == object_key)
    right_match = right_hand_holding and (right_item_name == object_key)

    if left_match and not right_match:
        hand = "left"
    elif right_match and not left_match:
        hand = "right"
    elif left_match and right_match:
        hand = "left"
    else:
        return {
            "success": False,
            "reason": (
                f"没有手正在持有目标物体 {object_key}，"
                f"left=({left_hand_holding}, {left_item_name}), "
                f"right=({right_hand_holding}, {right_item_name})"
            )
        }

    text = task_description
    if "left" not in text.lower() and "right" not in text.lower():
        new_instruction = f"{text} with the {hand} hand"
    else:
        new_instruction = text

    return {
        "success": True,
        "hand": hand,
        "object_name": object_name,
        "new_instruction": new_instruction,
    }

def choose_available_hand(detected_hand: str):
    """
    根据检测结果 + 当前持握状态，决定最终该用哪只手抓取

    返回:
    {
        "success": True/False,
        "hand": "left"/"right",
        "reason": "..."
    }
    """
    global left_hand_holding, right_hand_holding
    global left_item_name, right_item_name

    if detected_hand not in ["left", "right"]:
        return {
            "success": False,
            "reason": f"无效的检测手: {detected_hand}"
        }

    left_busy = bool(left_hand_holding)
    right_busy = bool(right_hand_holding)

    # 两只手都忙
    if left_busy and right_busy:
        return {
            "success": False,
            "reason": (
                f"双手都被占用，无法执行新的抓取。"
                f"left=({left_hand_holding}, {left_item_name}), "
                f"right=({right_hand_holding}, {right_item_name})"
            )
        }

    # 检测建议左手
    if detected_hand == "left":
        if not left_busy:
            return {
                "success": True,
                "hand": "left",
                "reason": "detected_left_and_left_free"
            }
        elif not right_busy:
            return {
                "success": True,
                "hand": "right",
                "reason": "detected_left_but_left_busy_switch_to_right"
            }

    # 检测建议右手
    if detected_hand == "right":
        if not right_busy:
            return {
                "success": True,
                "hand": "right",
                "reason": "detected_right_and_right_free"
            }
        elif not left_busy:
            return {
                "success": True,
                "hand": "left",
                "reason": "detected_right_but_right_busy_switch_to_left"
            }

    return {
        "success": False,
        "reason": "unexpected_hand_arbitration_state"
    }

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

    detector_client = DetectionClient3DSync("ws://10.8.26.11:8000")

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
                torso = 0.26 - 0.02
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
            if (
                "place" in task_description.lower()
                and "left" not in task_description.lower()
                and "right" not in task_description.lower()
            ):
                place_ret = infer_place_hand_and_rewrite_instruction(task_description)

                if not place_ret["success"]:
                    finish_task_with_failure(
                        interface,
                        msg=place_ret["reason"],
                        error_code=-1,
                    )
                    # result = ActionVLA.Result()
                    # result.success = False
                    # result.final_state = 3
                    # result.error_code = 0
                    # result.result_msg = "No traget object "
                    # with interface._lock:
                    #     interface._result_msg = result
                    #     interface._done_event.set()
                    # interface.new_task_flag = False
                    continue

                logger.info(
                    f"[PLACE HAND INFER] object={place_ret['object_name']}, "
                    f"hand={place_ret['hand']}, "
                    f"new_instruction={place_ret['new_instruction']}"
                )

                task_description = place_ret["new_instruction"]
                interface.latest_instruction = task_description
            
            if "place" in task_description:
                allow_left_release = "left" in task_description
                allow_right_release = "right" in task_description
                if "left" not in task_description and "right" not in task_description:
                    allow_left_release = True
                    allow_right_release = True
                left_place_done = False
                right_place_done = False
            
            if (
                "pick" in task_description.lower()
                and "left" not in task_description.lower()
                and "right" not in task_description.lower()
            ):
                infer_ret = infer_hand_and_rewrite_instruction(
                    interface=interface,
                    robot_controller=robot_controller,
                    detector_client=detector_client,
                    task_description=task_description,
                    target_location=interface.target_location if interface.target_location is not None else "",
                    save_debug=False,
                )

                if not infer_ret["success"]:
                    finish_task_with_failure(
                        interface,
                        msg=infer_ret["reason"],
                        error_code=-7,
                    )
                    # result = ActionVLA.Result()
                    # result.success = False
                    # result.final_state = 3
                    # result.error_code = 0
                    # result.result_msg = "No traget object "
                    # with interface._lock:
                    #     interface._result_msg = result
                    #     interface._done_event.set()
                    # interface.new_task_flag = False
                    
                    # finish_task_with_failure(
                    #     interface,
                    #     msg=infer_ret["reason"],
                    #     error_code=1001,
                    # )
                    continue

                logger.info(
                    f"[HAND DETECT] object={infer_ret['object_name']}, "
                    f"hand={infer_ret['hand']}, "
                    f"new_instruction={infer_ret['new_instruction']}"
                )

                time.sleep(1)

                task_description = infer_ret["new_instruction"]
                interface.latest_instruction = task_description

            
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
            
            task_start_time = time.time()
            while not task_complate:
                # ==========================
                # 任务超时判断
                # ==========================
                if time.time() - task_start_time > TASK_TIMEOUT:
                    print("[WARN] Task timeout! Returning to home position...")
                    robot_status = robot_controller.get_status()
                    lpose = robot_status.get("leftPose", None)
                    rpose = robot_status.get("rightPose", None)
                    
                    if "left" in task_description:
                        left_hand_holding = False
                        left_item_name = ""
                        left_grasp_counter = 0
                        allow_left_release = False
                        left_place_done = False 
                        robot_controller.control_lpose(lpose)
                        robot_controller.init_left()
                    
                    if "right" in task_description:
                        right_hand_holding = False
                        right_item_name = ""
                        right_grasp_counter = 0
                        allow_right_release = False
                        right_place_done = False
                        robot_controller.control_rpose(rpose)
                        robot_controller.init_right()

                    # 如果任务要求双手（如 pick-up apple）
                    if ("left" not in task_description) and ("right" not in task_description):
                        # 默认认为是双手任务
                        left_hand_holding = False
                        left_item_name = ""
                        left_grasp_counter = 0
                        allow_left_release = False
                        left_place_done = False
                        robot_controller.control_lpose(lpose)

                        right_hand_holding = False
                        right_item_name = ""
                        right_grasp_counter = 0
                        allow_right_release = False
                        right_place_done = False
                        robot_controller.control_rpose(rpose)
                        
                    interface.vla_status["left_state"] = 1 if left_hand_holding else 0
                    interface.vla_status["right_state"] = 1 if right_hand_holding else 0
                    interface.vla_status["left_item"] = left_item_name
                    interface.vla_status["right_item"] = right_item_name
                    
                    interface.vla_status["status"] = 3   # 3 = 完成
                    task_complate = True
                    action_finished = True
                    break
                    
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
                left_arm_joints = robot_status['leftjoint'] 
                if "white-desk" in interface.target_location:
                    left_arm_joints[2] += 0.1
                    left_arm_joints[3] += 0.02
                right_arm_joints = robot_status['rightjoint']
                left_gripper_state = robot_status['left_gripper']
                right_gripper_state = robot_status['right_gripper']
                
                state = np.array(left_arm_joints + left_gripper_state + right_arm_joints + right_gripper_state)
                state_tensor = torch.tensor(state, dtype=torch.float32)
                state = normalizer.forward(state_tensor)
                state = state.numpy().tolist()

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

                response = model.step(example)
                actions = response["raw_actions"]

                interface.vla_status["status"] = 2

                exec_steps = actions.shape[0]- 0
                for i in range(exec_steps):
                    act = actions[i]
                    start = time.time()
                    step_counter += 1
                    
                    # delta OR absolute
                    if "delta" in Args.pretrained_path:
                        left_delta = np.array(act[0:7])
                        right_delta = np.array(act[8:15])

                        left_arm = left_arm_joints + left_delta
                        right_arm = right_arm_joints + right_delta
                    else:  
                        left_arm = act[0:7]
                        right_arm = act[8:15]
                    
                    # white desk offset
                    if "white-desk" in interface.target_location:
                        left_arm[2] -= 0.1
                        left_arm[3] -= 0.02

                    left_arm = left_arm.tolist()
                    right_arm = right_arm.tolist()

                    left_grip_raw = int(act[7])
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
