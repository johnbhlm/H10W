import dataclasses
import sys
sys.path.append("/home/diana/intern-vla_test/starVLA/")
sys.stdout.flush()

import time
import threading
import logging
import queue
import os
import math
import numpy as np
import rclpy
import tyro
import torch
import json
import cv2

from examples.H10W.model2h10w_interface import ModelClient
from examples.H10W.robot_interface import H10WInferfaceConfig, H10WInterface
from examples.H10W.robot_controller import RobotController
from vla.action import ActionVLA

from detection_client_3d import DetectionClient3DSync

RUNNING = 2
SUCCEEDED = 3
FAILED = 4
TIMEOUT = 5

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
EARLY_HOLD_STEPS = 50
TASK_TIMEOUT = 40.0  # 
RTC_SWAP_GUARD_STEPS = 8
ASYNC_WAIT_TIMEOUT = 2.0
INITIAL_PREP_TIMEOUT = 3.0
MIN_FINISH_STEPS = 300

VALID_RUN_MODES = {"sync", "sync_rtc", "async", "async_rtc"}


def get_exec_mode_flags(exec_mode: str):
    mode = str(exec_mode).strip().lower()
    if mode == "sync":
        return False, False
    if mode == "sync_rtc":
        return False, True
    if mode == "async":
        return True, False
    if mode == "async_rtc":
        return True, True
    raise ValueError(f"Invalid EXEC_MODE={exec_mode}. Supported: sync, sync_rtc, async, async_rtc")

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
_warn_ts = {}

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


def _rate_limited_warning(key: str, message: str, interval_sec: float = 2.0):
    now = time.time()
    last = _warn_ts.get(key, 0.0)
    if now - last >= interval_sec:
        logger.warning(message)
        _warn_ts[key] = now


def set_running_if_not_terminal(interface):
    status = interface.vla_status.get("status")
    if status in (FAILED, TIMEOUT, SUCCEEDED):
        return False
    interface.vla_status["status"] = RUNNING
    return True


def reset_per_task_execution_state():
    global last_left_gripper, last_right_gripper
    global left_grasp_counter, right_grasp_counter
    global left_place_done, right_place_done
    global allow_left_release, allow_right_release

    left_action_buffer.clear()
    right_action_buffer.clear()
    last_left_gripper = 0
    last_right_gripper = 0
    left_grasp_counter = 0
    right_grasp_counter = 0
    left_place_done = False
    right_place_done = False
    allow_left_release = False
    allow_right_release = False


def check_task_completion(
    use_left,
    use_right,
    left_pose,
    right_pose,
    left_target_pose,
    right_target_pose,
    step_counter,
    finish_start_time,
    finish_hold_time,
):
    dl = np.linalg.norm(np.array(left_pose[:3]) - left_target_pose[:3])
    dr = np.linalg.norm(np.array(right_pose[:3]) - right_target_pose[:3])
    left_close = dl < 0.1
    right_close = dr < 0.1
    if use_left and use_right:
        completed = left_close and right_close
    elif use_left:
        completed = left_close
    elif use_right:
        completed = right_close
    else:
        completed = left_close and right_close

    if completed and step_counter > MIN_FINISH_STEPS:
        if finish_start_time is None:
            finish_start_time = time.time()
        elif time.time() - finish_start_time >= finish_hold_time:
            return True, finish_start_time
    else:
        finish_start_time = None
    return False, finish_start_time


class AsyncRTCInferenceWorker:
    """异步推理 worker，带 task_epoch/request_id 隔离，防止串任务与旧结果污染。"""
    def __init__(self, model, wait_timeout: float = ASYNC_WAIT_TIMEOUT, request_queue_size: int = 2, result_queue_size: int = 2):
        self.model = model
        self.wait_timeout = wait_timeout
        self._request_q = queue.Queue(maxsize=max(1, int(request_queue_size)))
        self._result_q = queue.Queue(maxsize=max(1, int(result_queue_size)))
        self._stop_event = threading.Event()
        self._meta_lock = threading.Lock()
        self._closed = False
        self._generation_id = 0
        self._task_epoch = 0
        self._latency_ema = None
        self._stale_inflight_dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _flush_queue(self, q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def start_new_task(self):
        with self._meta_lock:
            self._task_epoch += 1
            task_epoch = self._task_epoch
        self._flush_queue(self._request_q)
        self._flush_queue(self._result_q)
        return task_epoch

    def get_recommended_frozen(self, control_dt: float, action_horizon: int, default_frozen: int):
        with self._meta_lock:
            latency = self._latency_ema
        if latency is None or control_dt <= 0:
            return int(default_frozen)
        estimated = int(math.ceil(max(0.0, latency) / control_dt))
        estimated = max(1, min(estimated, max(1, action_horizon - 1)))
        return estimated

    def _run(self):
        while not self._stop_event.is_set():
            try:
                request = self._request_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if request is None:
                return
            req_generation, req_epoch, request_id, example = request

            try:
                t0 = time.time()
                response = self.model.step(example)
                latency = time.time() - t0
                with self._meta_lock:
                    if self._latency_ema is None:
                        self._latency_ema = latency
                    else:
                        self._latency_ema = 0.8 * self._latency_ema + 0.2 * latency
                    closed = self._closed
                    current_generation = self._generation_id
                    current_epoch = self._task_epoch
                if closed or req_generation != current_generation or req_epoch != current_epoch:
                    with self._meta_lock:
                        self._stale_inflight_dropped += 1
                    logger.debug(
                        f"[ASYNC INFER] drop stale in-flight result req={request_id}, "
                        f"req_gen={req_generation}, cur_gen={current_generation}, "
                        f"req_epoch={req_epoch}, cur_epoch={current_epoch}"
                    )
                    continue
                actions = response["raw_actions"]
                while self._result_q.full():
                    try:
                        self._result_q.get_nowait()
                    except queue.Empty:
                        break
                self._result_q.put((req_epoch, request_id, actions))
            except Exception as e:
                logger.exception(f"[ASYNC INFER] inference failed for request_id={request_id}: {e}")

    def request(self, example, request_id: int, task_epoch: int):
        with self._meta_lock:
            if self._closed:
                logger.debug(f"[ASYNC INFER] reject request_id={request_id}: worker already closed")
                return False
            gen = self._generation_id
        while self._request_q.full():
            try:
                self._request_q.get_nowait()
            except queue.Empty:
                break
        self._request_q.put((gen, task_epoch, request_id, example))
        return True

    def get_latency_ema(self):
        with self._meta_lock:
            return self._latency_ema

    def get_stale_inflight_dropped(self):
        with self._meta_lock:
            return self._stale_inflight_dropped

    def get_blocking_for_request(self, request_id: int, task_epoch: int, timeout: float = None):
        timeout = self.wait_timeout if timeout is None else timeout
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            try:
                got_epoch, got_req, actions = self._result_q.get(timeout=remaining)
            except queue.Empty:
                break
            if got_epoch == task_epoch and got_req == request_id:
                return got_req, actions
            logger.debug(
                f"[ASYNC INFER] discard non-matching result, got(epoch={got_epoch}, req={got_req}), "
                f"expect(epoch={task_epoch}, req={request_id})"
            )
        raise queue.Empty(f"timeout waiting request_id={request_id}, task_epoch={task_epoch}")

    def get_latest_matching_request(self, request_id: int, task_epoch: int):
        latest = None
        while not self._result_q.empty():
            try:
                got_epoch, got_req, actions = self._result_q.get_nowait()
            except queue.Empty:
                break
            if got_epoch == task_epoch and got_req == request_id:
                latest = (got_req, actions)
            else:
                logger.debug(
                    f"[ASYNC INFER] drop stale latest result got(epoch={got_epoch}, req={got_req}), "
                    f"expect(epoch={task_epoch}, req={request_id})"
                )
        return latest

    def stop(self):
        with self._meta_lock:
            self._closed = True
            self._generation_id += 1
        self._stop_event.set()
        self._flush_queue(self._request_q)
        self._flush_queue(self._result_q)
        try:
            self._request_q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=0.5)


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
    run_mode: str = "sync"
    control_dt: float = 0.01
    async_queue_size: int = 2
    prefetch_lead_steps: int = 6
    rtc_overlap: int = 8
    rtc_frozen: int = 6
    rtc_alpha: float = 0.5
    enable_action_smoothing: bool = True
    chunk_min_for_rtc_warn: int = 32

    if "unifolm" in pretrained_path:
        resize_size = [256, 256]
    else:
        resize_size = [224, 224]

normalizer = None

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
    result.final_state = FAILED
    result.error_code = error_code
    result.result_msg = msg

    with interface._lock:
        interface._result_msg = result
        interface._done_event.set()

    interface.vla_status["status"] = FAILED
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


def prepare_policy_example(interface, robot_controller, task_description: str):
    global normalizer
    obs = interface.get_observations()
    if obs is None:
        _rate_limited_warning("prepare_obs_none", "prepare_policy_example skipped: obs is None")
        return None

    required_keys = ["head_rgb", "left_rgb", "right_rgb"]
    for key in required_keys:
        if key not in obs or obs[key] is None or obs[key].get("data", None) is None:
            _rate_limited_warning("prepare_obs_missing", f"prepare_policy_example skipped: missing {key}.data")
            return None

    head_img = obs["head_rgb"]["data"].copy()
    left_img = obs["left_rgb"]["data"].copy()
    right_img = obs["right_rgb"]["data"].copy()

    if freeze_left_image:
        if frozen_left_rgb is None:
            _rate_limited_warning("freeze_left_missing", "left frozen image not available, keep live image")
        else:
            left_img = frozen_left_rgb.copy()
    if freeze_right_image:
        if frozen_right_rgb is None:
            _rate_limited_warning("freeze_right_missing", "right frozen image not available, keep live image")
        else:
            right_img = frozen_right_rgb.copy()

    images = [head_img, left_img, right_img]

    robot_status = robot_controller.get_status()
    if robot_status is None:
        _rate_limited_warning("prepare_robot_status_none", "prepare_policy_example skipped: robot status is None")
        return None
    required = ["leftjoint", "rightjoint", "left_gripper", "right_gripper"]
    for key in required:
        if key not in robot_status or robot_status[key] is None:
            _rate_limited_warning("prepare_robot_status_missing", f"prepare_policy_example skipped: missing {key}")
            return None

    left_arm_joints = list(robot_status["leftjoint"])
    right_arm_joints = list(robot_status["rightjoint"])
    left_gripper_state = list(robot_status["left_gripper"])
    right_gripper_state = list(robot_status["right_gripper"])
    if len(left_arm_joints) != 7 or len(right_arm_joints) != 7:
        _rate_limited_warning("prepare_joint_len_invalid", f"prepare_policy_example skipped: invalid joint len L={len(left_arm_joints)} R={len(right_arm_joints)}")
        return None
    if len(left_gripper_state) < 1 or len(right_gripper_state) < 1:
        _rate_limited_warning("prepare_gripper_len_invalid", f"prepare_policy_example skipped: invalid gripper len L={len(left_gripper_state)} R={len(right_gripper_state)}")
        return None

    target_location_snapshot = str(interface.target_location or "")
    if "white-desk" in target_location_snapshot:
        left_arm_joints[2] += 0.1
        left_arm_joints[3] += 0.02
    current_item_snapshot = str(interface.current_item) if interface.current_item is not None else ""
    task_description_snapshot = str(task_description)

    state = np.array(left_arm_joints + left_gripper_state + right_arm_joints + right_gripper_state)
    state_tensor = torch.tensor(state, dtype=torch.float32)
    state = normalizer.forward(state_tensor)
    state = state.numpy().tolist()

    example = {
        "image": images,
        "lang": task_description_snapshot,
        "state": state,
    }

    return {
        "example": example,
        "left_arm_joints": left_arm_joints,
        "right_arm_joints": right_arm_joints,
        "left_gripper_state": left_gripper_state,
        "right_gripper_state": right_gripper_state,
        "current_item_snapshot": current_item_snapshot,
        "task_description_snapshot": task_description_snapshot,
        "target_location_snapshot": target_location_snapshot,
    }




def get_current_control_state(robot_controller, target_location: str):
    robot_status = robot_controller.get_status()
    if robot_status is None:
        _rate_limited_warning("robot_status_none", "robot status is None")
        return None
    required = ["leftjoint", "rightjoint", "left_gripper", "right_gripper"]
    for key in required:
        if key not in robot_status or robot_status[key] is None:
            _rate_limited_warning("robot_status_missing", f"robot status missing key: {key}")
            return None
    left_arm_joints = list(robot_status["leftjoint"])
    right_arm_joints = list(robot_status["rightjoint"])
    left_gripper_state = list(robot_status["left_gripper"])
    right_gripper_state = list(robot_status["right_gripper"])
    if len(left_arm_joints) != 7 or len(right_arm_joints) != 7:
        _rate_limited_warning("robot_joint_len_invalid", f"robot status invalid joint len: L={len(left_arm_joints)}, R={len(right_arm_joints)}")
        return None
    if len(left_gripper_state) < 1 or len(right_gripper_state) < 1:
        _rate_limited_warning("robot_gripper_len_invalid", f"robot status invalid gripper len: L={len(left_gripper_state)}, R={len(right_gripper_state)}")
        return None
    if "white-desk" in target_location:
        left_arm_joints[2] += 0.1
        left_arm_joints[3] += 0.02
    return {
        "left_arm_joints": left_arm_joints,
        "right_arm_joints": right_arm_joints,
        "left_gripper_state": left_gripper_state,
        "right_gripper_state": right_gripper_state,
        "leftPose": robot_status.get("leftPose", None),
        "rightPose": robot_status.get("rightPose", None),
    }


def sanitize_action(left_arm, right_arm, left_gripper, right_gripper, target_location):
    left = np.asarray(left_arm, dtype=np.float32).reshape(-1)
    right = np.asarray(right_arm, dtype=np.float32).reshape(-1)
    if left.shape[0] != 7 or right.shape[0] != 7:
        return {"valid": False, "reason": "invalid arm dof"}
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return {"valid": False, "reason": "non-finite arm action"}
    if not np.isfinite(float(left_gripper)) or not np.isfinite(float(right_gripper)):
        return {"valid": False, "reason": "non-finite gripper action"}

    try:
        lg = int(float(left_gripper) >= 0.5)
        rg = int(float(right_gripper) >= 0.5)
    except Exception:
        return {"valid": False, "reason": "invalid gripper action"}

    return {
        "valid": True,
        "left_arm": left,
        "right_arm": right,
        "left_gripper": lg,
        "right_gripper": rg,
    }




def fuse_chunks_rtc(prev_chunk, next_chunk, overlap: int, frozen: int, alpha: float = 0.5):
    prev = np.asarray(prev_chunk, dtype=np.float32)
    nxt = np.asarray(next_chunk, dtype=np.float32)
    if prev.ndim != 2 or nxt.ndim != 2 or prev.shape[1] < 16 or nxt.shape[1] < 16:
        return nxt
    actual_frozen = max(0, min(int(frozen), len(prev)))
    actual_overlap = max(1, min(int(overlap), len(prev) - actual_frozen, len(nxt)))
    if actual_overlap <= 0:
        return nxt
    frozen_keep = prev[-actual_frozen:] if actual_frozen > 0 else np.zeros((0, prev.shape[1]), dtype=np.float32)
    prev_overlap = prev[-(actual_frozen + actual_overlap):-actual_frozen if actual_frozen > 0 else None].copy()
    next_overlap = nxt[:actual_overlap].copy()
    alphas = np.linspace(max(0.0, alpha - 0.5), min(1.0, alpha + 0.5), actual_overlap, dtype=np.float32)
    for i, a in enumerate(alphas):
        prev_overlap[i, 0:7] = (1.0 - a) * prev_overlap[i, 0:7] + a * next_overlap[i, 0:7]
        prev_overlap[i, 8:15] = (1.0 - a) * prev_overlap[i, 8:15] + a * next_overlap[i, 8:15]
        prev_overlap[i, 7] = prev_overlap[i, 7] if i < actual_frozen else next_overlap[i, 7]
        prev_overlap[i, 15] = prev_overlap[i, 15] if i < actual_frozen else next_overlap[i, 15]
    return np.concatenate([frozen_keep, prev_overlap, nxt[actual_overlap:]], axis=0)


def execute_single_task(
    args: Args,
    interface,
    robot_controller,
    model,
    detector_client,
    task_description: str,
):
    global left_hand_holding, right_hand_holding
    global allow_left_release, allow_right_release
    global last_left_gripper, last_right_gripper
    global left_grasp_counter, right_grasp_counter
    global left_place_done, right_place_done
    global left_item_name, right_item_name
    global use_left, use_right
    global freeze_left_image, freeze_right_image

    task_complete = False
    action_finished = False
    finish_hold_time = 0.5
    finish_start_time = None
    left_target_pose = np.array([0.60, 0.60, 1.21])
    target_location = interface.target_location or ""
    use_async_prefetch, use_rtc_fusion = get_exec_mode_flags(args.run_mode)
    allow_mid_chunk_switch = use_async_prefetch and use_rtc_fusion
    async_worker = AsyncRTCInferenceWorker(
        model,
        wait_timeout=ASYNC_WAIT_TIMEOUT,
        request_queue_size=args.async_queue_size,
        result_queue_size=args.async_queue_size,
    ) if use_async_prefetch else None
    current_request_id = 0

    def is_valid_chunk(chunk):
        return chunk is not None and len(chunk) > 0

    def wait_prepare(task_desc: str, timeout: float):
        prepared = None
        t0 = time.time()
        while prepared is None and (time.time() - t0) < timeout:
            prepared = prepare_policy_example(interface, robot_controller, task_desc)
            if prepared is None:
                time.sleep(0.001)
        return prepared

    try:
        interface.vla_status["status"] = RUNNING

        if "pick" not in task_description and "place" not in task_description:
            robot_controller.h10w_system.enableController(True)
            robot_controller.control_torso(torso=0.54)
            robot_controller.h10w_system.enableController(False)
            interface.vla_status["status"] = SUCCEEDED
            return True, SUCCEEDED, "Task finished"

        if "sofa" in target_location:
            robot_controller.h10w_system.enableController(True)
            robot_controller.control_torso(torso=0.24)
            robot_controller.h10w_system.enableController(False)
            left_target_pose[2] = 0.9
        elif "TV-cabinet" in target_location:
            robot_controller.h10w_system.enableController(True)
            robot_controller.control_torso(torso=0.34)
            robot_controller.h10w_system.enableController(False)
            left_target_pose[2] = 1.0
        else:
            left_target_pose[2] = 1.21
        
        if action_finished:
            task_complete = False
            finish_start_time = None
            step_counter = 0
            left_action_buffer.clear()
            right_action_buffer.clear()
            action_finished = False

        if "place" in task_description.lower() and "left" not in task_description.lower() and "right" not in task_description.lower():
            place_ret = infer_place_hand_and_rewrite_instruction(task_description)
            if not place_ret["success"]:
                finish_task_with_failure(interface, msg=place_ret["reason"], error_code=-1)
                return False, FAILED, place_ret["reason"]
            task_description = place_ret["new_instruction"]
            interface.latest_instruction = task_description

        if "pick" in task_description.lower() and "left" not in task_description.lower() and "right" not in task_description.lower():
            infer_ret = infer_hand_and_rewrite_instruction(
                interface=interface,
                robot_controller=robot_controller,
                detector_client=detector_client,
                task_description=task_description,
                target_location=target_location,
                save_debug=False,
            )
            if not infer_ret["success"]:
                finish_task_with_failure(interface, msg=infer_ret["reason"], error_code=-7)
                return False, FAILED, infer_ret["reason"]
            time.sleep(1)
            task_description = infer_ret["new_instruction"]
            interface.latest_instruction = task_description

        # reset_per_task_execution_state()
        if "place" in task_description:
            allow_left_release = "left" in task_description
            allow_right_release = "right" in task_description
            if "left" not in task_description and "right" not in task_description:
                allow_left_release = True
                allow_right_release = True
            left_place_done = False
            right_place_done = False

        use_left = "left" in task_description
        use_right = "right" in task_description
        if not use_left and not use_right:
            use_left = True
            use_right = True

        if "left" in task_description:
            freeze_right_image = True
        if "right" in task_description:
            freeze_left_image = True

        robot_controller.h10w_system.enableController(True)
        robot_controller.h10w_motion.enableRealtimeCmd(True)
        step_counter = 0
        task_start_time = time.time()

        if args.run_mode in ("sync", "sync_rtc"):
            prev_tail = None
            while not task_complete:
                if time.time() - task_start_time > TASK_TIMEOUT:
                    # interface.vla_status["status"] = TIMEOUT
                    # return False, TIMEOUT, "Task timeout"
                    
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
                    
                    interface.vla_status["status"] = TIMEOUT   # 3 = 完成
                    task_complete = True
                    action_finished = True
                    break

                obs = interface.get_observations()
                if obs is None:
                    continue

                head_img = obs["head_rgb"]["data"]
                left_img = obs["left_rgb"]["data"]
                right_img = obs["right_rgb"]["data"]
                if freeze_left_image and frozen_left_rgb is not None:
                    left_img = frozen_left_rgb
                if freeze_right_image and frozen_right_rgb is not None:
                    right_img = frozen_right_rgb
                images = [head_img, left_img, right_img]

                robot_status = robot_controller.get_status()
                if robot_status is None:
                    time.sleep(0.01)
                    continue
                left_arm_joints = list(robot_status["leftjoint"])
                if "white-desk" in target_location:
                    left_arm_joints[2] += 0.1
                    left_arm_joints[3] += 0.02
                right_arm_joints = list(robot_status["rightjoint"])
                left_gripper_state = robot_status["left_gripper"]
                right_gripper_state = robot_status["right_gripper"]

                state = np.array(left_arm_joints + left_gripper_state + right_arm_joints + right_gripper_state)
                state_tensor = torch.tensor(state, dtype=torch.float32)
                state = normalizer.forward(state_tensor)
                state = state.numpy().tolist()
                example = {"image": images, "lang": task_description, "state": state}
                current_item = interface.current_item

                robot_status = robot_controller.get_status()
                if robot_status is None:
                    time.sleep(0.001)
                    continue
                left_pose = robot_status.get("leftPose", None)
                right_pose = robot_status.get("rightPose", None)
                if left_pose is None or right_pose is None:
                    time.sleep(0.001)
                    continue

                right_target_pose = left_target_pose.copy()
                right_target_pose[1] *= -1
                completed, finish_start_time = check_task_completion(
                    use_left=use_left,
                    use_right=use_right,
                    left_pose=left_pose,
                    right_pose=right_pose,
                    left_target_pose=left_target_pose,
                    right_target_pose=right_target_pose,
                    step_counter=step_counter,
                    finish_start_time=finish_start_time,
                    finish_hold_time=finish_hold_time,
                )
                if completed:
                    interface.vla_status["status"] = SUCCEEDED
                    task_complete = True
                    interface.vla_status["left_state"] = 1 if last_left_gripper == 1 else 0
                    interface.vla_status["right_state"] = 1 if last_right_gripper == 1 else 0
                    break

                response = model.step(example)
                actions = response["raw_actions"]
                if args.run_mode == "sync_rtc" and prev_tail is not None and len(prev_tail) > 0:
                    actions = fuse_chunks_rtc(prev_tail, actions, overlap=args.rtc_overlap, frozen=min(args.rtc_frozen, 1), alpha=args.rtc_alpha)
                interface.vla_status["status"] = RUNNING

                exec_steps = actions.shape[0]
                for i in range(exec_steps):
                    act = actions[i]
                    start = time.time()
                    step_counter += 1

                    if "delta" in args.pretrained_path:
                        left_arm = np.array(left_arm_joints) + np.array(act[0:7])
                        right_arm = np.array(right_arm_joints) + np.array(act[8:15])
                    else:
                        left_arm = np.array(act[0:7])
                        right_arm = np.array(act[8:15])

                    if "white-desk" in target_location:
                        left_arm[2] -= 0.1
                        left_arm[3] -= 0.02

                    left_grip_raw = int(act[7])
                    right_grip_raw = int(act[15])
                    stable_left = smooth_gripper(left_action_buffer, left_grip_raw)
                    stable_right = smooth_gripper(right_action_buffer, right_grip_raw)
                    last_left_gripper = resolve_gripper(stable_left, last_left_gripper, left_hand_holding, allow_left_release)
                    last_right_gripper = resolve_gripper(stable_right, last_right_gripper, right_hand_holding, allow_right_release)

                    safe_action = sanitize_action(left_arm, right_arm, last_left_gripper, last_right_gripper, target_location)
                    if not safe_action["valid"]:
                        finish_task_with_failure(interface, f"Unsafe action rejected: {safe_action['reason']}", error_code=-10)
                        return False, FAILED, safe_action["reason"]

                    robot_controller.control_joints(
                        left_arm=safe_action["left_arm"].tolist(),
                        right_arm=safe_action["right_arm"].tolist(),
                        left_gripper=safe_action["left_gripper"],
                        right_gripper=safe_action["right_gripper"],
                        control_time=args.control_dt,
                    )

                    time.sleep(max(0.0, args.control_dt - (time.time() - start)))

                    if not left_hand_holding:
                        if last_left_gripper == 1:
                            left_grasp_counter += 1
                            if left_grasp_counter >= 50 and left_item_name == "":
                                left_item_name = current_item
                            if left_grasp_counter >= GRASP_CONFIRM_STEPS:
                                left_hand_holding = True
                                left_grasp_counter = GRASP_CONFIRM_STEPS
                        else:
                            left_grasp_counter = 0
                    else:
                        left_grasp_counter = GRASP_CONFIRM_STEPS
                    if allow_left_release and last_left_gripper == 0:
                        left_hand_holding = False
                        left_place_done = True
                        left_item_name = ""
                        left_grasp_counter = 0

                    if not right_hand_holding:
                        if last_right_gripper == 1:
                            right_grasp_counter += 1
                            if right_grasp_counter >= 50 and right_item_name == "":
                                right_item_name = current_item
                            if right_grasp_counter >= GRASP_CONFIRM_STEPS:
                                right_hand_holding = True
                                right_grasp_counter = GRASP_CONFIRM_STEPS
                        else:
                            right_grasp_counter = 0
                    else:
                        right_grasp_counter = GRASP_CONFIRM_STEPS
                    if allow_right_release and last_right_gripper == 0:
                        right_hand_holding = False
                        right_place_done = True
                        right_item_name = ""
                        right_grasp_counter = 0

                    interface.vla_status["left_state"] = 1 if left_hand_holding else 0
                    interface.vla_status["right_state"] = 1 if right_hand_holding else 0
                    interface.vla_status["left_item"] = left_item_name
                    interface.vla_status["right_item"] = right_item_name

                prev_tail = actions[-1:, ...]
        else:
            prepared = wait_prepare(task_description, INITIAL_PREP_TIMEOUT)
            if prepared is None:
                finish_task_with_failure(interface, "Initial observation unavailable", error_code=-11)
                return False, FAILED, "Initial observation unavailable"

            current_item_snapshot = prepared["current_item_snapshot"]
            current_chunk_left_snapshot = np.asarray(prepared["left_arm_joints"], dtype=np.float32)
            current_chunk_right_snapshot = np.asarray(prepared["right_arm_joints"], dtype=np.float32)
            task_epoch = async_worker.start_new_task()
            current_request_id += 1
            if not async_worker.request(prepared["example"], request_id=current_request_id, task_epoch=task_epoch):
                finish_task_with_failure(interface, "Initial inference request rejected", error_code=-12)
                return False, FAILED, "Initial inference request rejected"
            try:
                _, current_actions = async_worker.get_blocking_for_request(current_request_id, task_epoch=task_epoch)
            except Exception:
                finish_task_with_failure(interface, "Initial inference failed/timeout", error_code=-13)
                return False, FAILED, "Initial inference failed/timeout"

            while not task_complete:
                actions = current_actions
                exec_steps = actions.shape[0]
                i = 0
                async_requested = False
                pending_request_id = None
                next_chunk_item_snapshot = current_item_snapshot
                next_chunk_left_snapshot = current_chunk_left_snapshot
                next_chunk_right_snapshot = current_chunk_right_snapshot
                switched_mid_chunk = False

                if allow_mid_chunk_switch:
                    rtc_frozen = async_worker.get_recommended_frozen(args.control_dt, actions.shape[0], args.rtc_frozen)
                    rtc_overlap = max(args.rtc_overlap, rtc_frozen + 4)
                    rtc_overlap = max(1, min(rtc_overlap, max(1, exec_steps - 1)))
                    swap_guard = min(max(RTC_SWAP_GUARD_STEPS, rtc_frozen), max(1, exec_steps - 1))
                    lead_steps = min(max(args.prefetch_lead_steps, rtc_frozen + 2), max(1, exec_steps - 1))
                    rtc_swap_idx = max(0, exec_steps - swap_guard)
                    rtc_trigger_idx = max(0, rtc_swap_idx - lead_steps)
                else:
                    rtc_frozen = args.rtc_frozen
                    rtc_overlap = args.rtc_overlap
                    lead_steps = min(max(1, args.prefetch_lead_steps), max(1, exec_steps - 1))
                    rtc_trigger_idx = max(0, exec_steps - lead_steps)
                    rtc_swap_idx = exec_steps

                while i < exec_steps:
                    step_start = time.time()
                    if time.time() - task_start_time > TASK_TIMEOUT:
                        interface.vla_status["status"] = TIMEOUT
                        return False, TIMEOUT, "Task timeout"

                    control_state = get_current_control_state(robot_controller, target_location)
                    if control_state is None:
                        time.sleep(0.005)
                        continue

                    left_pose = control_state.get("leftPose")
                    right_pose = control_state.get("rightPose")
                    if left_pose is None or right_pose is None:
                        time.sleep(0.005)
                        continue

                    act = actions[i]
                    if "delta" in args.pretrained_path:
                        left_arm = current_chunk_left_snapshot + np.array(act[0:7], dtype=np.float32)
                        right_arm = current_chunk_right_snapshot + np.array(act[8:15], dtype=np.float32)
                    else:
                        left_arm = np.array(act[0:7], dtype=np.float32)
                        right_arm = np.array(act[8:15], dtype=np.float32)
                    if "white-desk" in target_location:
                        left_arm[2] -= 0.1
                        left_arm[3] -= 0.02

                    left_grip_raw = int(act[7])
                    right_grip_raw = int(act[15])
                    stable_left = smooth_gripper(left_action_buffer, left_grip_raw)
                    stable_right = smooth_gripper(right_action_buffer, right_grip_raw)
                    last_left_gripper = resolve_gripper(stable_left, last_left_gripper, left_hand_holding, allow_left_release)
                    last_right_gripper = resolve_gripper(stable_right, last_right_gripper, right_hand_holding, allow_right_release)

                    safe_action = sanitize_action(left_arm, right_arm, last_left_gripper, last_right_gripper, target_location)
                    if not safe_action["valid"]:
                        finish_task_with_failure(interface, f"Unsafe action rejected: {safe_action['reason']}", error_code=-10)
                        return False, FAILED, safe_action["reason"]

                    robot_controller.control_joints(
                        left_arm=safe_action["left_arm"].tolist(),
                        right_arm=safe_action["right_arm"].tolist(),
                        left_gripper=safe_action["left_gripper"],
                        right_gripper=safe_action["right_gripper"],
                        control_time=args.control_dt,
                    )

                    step_counter += 1
                    i += 1
                    last_left_gripper = safe_action["left_gripper"]
                    last_right_gripper = safe_action["right_gripper"]

                    right_target_pose = left_target_pose.copy()
                    right_target_pose[1] *= -1
                    completed, finish_start_time = check_task_completion(
                        use_left=use_left,
                        use_right=use_right,
                        left_pose=left_pose,
                        right_pose=right_pose,
                        left_target_pose=left_target_pose,
                        right_target_pose=right_target_pose,
                        step_counter=step_counter,
                        finish_start_time=finish_start_time,
                        finish_hold_time=finish_hold_time,
                    )
                    if completed:
                        interface.vla_status["status"] = SUCCEEDED
                        task_complete = True

                    if not left_hand_holding:
                        if last_left_gripper == 1:
                            left_grasp_counter += 1
                            if left_grasp_counter >= 50 and left_item_name == "":
                                left_item_name = current_item_snapshot
                            if "pick" in task_description.lower() and left_grasp_counter >= EARLY_HOLD_STEPS:
                                left_hand_holding = True
                            if left_grasp_counter >= GRASP_CONFIRM_STEPS:
                                left_grasp_counter = GRASP_CONFIRM_STEPS
                        else:
                            left_grasp_counter = 0
                    else:
                        left_grasp_counter = GRASP_CONFIRM_STEPS
                    if allow_left_release and last_left_gripper == 0:
                        left_hand_holding = False
                        left_place_done = True
                        left_item_name = ""
                        left_grasp_counter = 0

                    if not right_hand_holding:
                        if last_right_gripper == 1:
                            right_grasp_counter += 1
                            if right_grasp_counter >= 50 and right_item_name == "":
                                right_item_name = current_item_snapshot
                            if "pick" in task_description.lower() and right_grasp_counter >= EARLY_HOLD_STEPS:
                                right_hand_holding = True
                            if right_grasp_counter >= GRASP_CONFIRM_STEPS:
                                right_grasp_counter = GRASP_CONFIRM_STEPS
                        else:
                            right_grasp_counter = 0
                    else:
                        right_grasp_counter = GRASP_CONFIRM_STEPS
                    if allow_right_release and last_right_gripper == 0:
                        right_hand_holding = False
                        right_place_done = True
                        right_item_name = ""
                        right_grasp_counter = 0

                    interface.vla_status["left_state"] = 1 if left_hand_holding else 0
                    interface.vla_status["right_state"] = 1 if right_hand_holding else 0
                    interface.vla_status["left_item"] = left_item_name
                    interface.vla_status["right_item"] = right_item_name
                    set_running_if_not_terminal(interface)

                    if use_async_prefetch and (not async_requested) and i >= rtc_trigger_idx:
                        next_prepared = prepare_policy_example(interface, robot_controller, task_description)
                        if next_prepared is not None:
                            current_request_id += 1
                            pending_request_id = current_request_id
                            next_chunk_item_snapshot = next_prepared["current_item_snapshot"]
                            next_chunk_left_snapshot = np.asarray(next_prepared["left_arm_joints"], dtype=np.float32)
                            next_chunk_right_snapshot = np.asarray(next_prepared["right_arm_joints"], dtype=np.float32)
                            async_requested = async_worker.request(next_prepared["example"], pending_request_id, task_epoch=task_epoch)

                    if task_complete:
                        break

                    if allow_mid_chunk_switch and async_requested and pending_request_id is not None and i >= rtc_swap_idx:
                        got = async_worker.get_latest_matching_request(request_id=pending_request_id, task_epoch=task_epoch)
                        if got is not None:
                            _, next_actions = got
                            if is_valid_chunk(next_actions):
                                current_actions = fuse_chunks_rtc(actions[i:], next_actions, overlap=rtc_overlap, frozen=rtc_frozen, alpha=args.rtc_alpha)
                                current_item_snapshot = next_chunk_item_snapshot
                                current_chunk_left_snapshot = next_chunk_left_snapshot
                                current_chunk_right_snapshot = next_chunk_right_snapshot
                                switched_mid_chunk = True
                                break

                    time.sleep(max(0.0, args.control_dt - (time.time() - step_start)))

                if task_complete:
                    break
                if switched_mid_chunk:
                    continue

                if (not async_requested):
                    next_prepared = prepare_policy_example(interface, robot_controller, task_description)
                    if next_prepared is not None:
                        current_request_id += 1
                        pending_request_id = current_request_id
                        next_chunk_item_snapshot = next_prepared["current_item_snapshot"]
                        next_chunk_left_snapshot = np.asarray(next_prepared["left_arm_joints"], dtype=np.float32)
                        next_chunk_right_snapshot = np.asarray(next_prepared["right_arm_joints"], dtype=np.float32)
                        async_requested = async_worker.request(next_prepared["example"], pending_request_id, task_epoch=task_epoch)

                if async_requested and pending_request_id is not None:
                    try:
                        _, next_actions = async_worker.get_blocking_for_request(pending_request_id, task_epoch=task_epoch, timeout=ASYNC_WAIT_TIMEOUT)
                    except Exception:
                        next_actions = None
                    if is_valid_chunk(next_actions):
                        if use_rtc_fusion:
                            current_actions = fuse_chunks_rtc(actions[-1:, ...], next_actions, overlap=rtc_overlap, frozen=rtc_frozen, alpha=args.rtc_alpha)
                        else:
                            current_actions = next_actions
                        current_item_snapshot = next_chunk_item_snapshot
                        current_chunk_left_snapshot = next_chunk_left_snapshot
                        current_chunk_right_snapshot = next_chunk_right_snapshot
                    else:
                        current_actions = actions[-1:, ...]
                else:
                    current_actions = actions[-1:, ...]

        if interface.vla_status.get("status") == SUCCEEDED:
            return True, SUCCEEDED, "Task finished"
        if interface.vla_status.get("status") == TIMEOUT:
            return False, TIMEOUT, "Task timeout"
        return False, FAILED, "Task failed"
    finally:
        if async_worker is not None:
            async_worker.stop()
        robot_controller.init_left()
        robot_controller.init_right()
        robot_controller.h10w_motion.enableRealtimeCmd(False)
        robot_controller.h10w_system.enableController(False)
        use_left = False
        use_right = False
        allow_left_release = False
        allow_right_release = False
        freeze_left_image = False
        freeze_right_image = False

def main(args: Args):
    global normalizer
    args.run_mode = str(args.run_mode).strip().lower()
    if args.run_mode not in VALID_RUN_MODES:
        raise ValueError(f"run_mode must be one of {sorted(VALID_RUN_MODES)}, got: {args.run_mode}")
    if args.rtc_overlap <= args.rtc_frozen:
        logger.warning(f"rtc_overlap({args.rtc_overlap}) <= rtc_frozen({args.rtc_frozen}), auto fix to frozen+1")
        args.rtc_overlap = args.rtc_frozen + 1
    rclpy.init()
    stats_path = os.path.join(args.pretrained_path, "dataset_statistics.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"dataset statistics not found: {stats_path}")
    stats = load_stats(stats_path)
    normalizer = Normalizer("min_max", stats)

    robot_controller = RobotController()
    interface = H10WInterface(H10WInferfaceConfig(), robot_controller)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(interface)
    threading.Thread(target=executor.spin, daemon=True).start()
    logger.info("ROS Executor started. Waiting for camera topics...")
    time.sleep(1.0)

    model = ModelClient(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )
    detector_client = DetectionClient3DSync("ws://10.8.26.11:8000")
    mode_async, mode_rtc = get_exec_mode_flags(args.run_mode)
    logger.info(
        f"Runtime config: mode={args.run_mode}, control_dt={args.control_dt}, action_horizon=unknown, "
        f"async_enabled={mode_async}, rtc_enabled={mode_rtc}, "
        f"RTC_OVERLAP={args.rtc_overlap}, RTC_FROZEN={args.rtc_frozen}, RTC_REQUEST_LEAD_STEPS={args.prefetch_lead_steps}, "
        f"RTC_SWAP_GUARD_STEPS={RTC_SWAP_GUARD_STEPS}, ASYNC_WAIT_TIMEOUT={ASYNC_WAIT_TIMEOUT}"
    )

    while True:
        task_description = interface.latest_instruction
        if task_description is None or not interface.new_task_flag:
            time.sleep(0.01)
            continue
        with interface._lock:
            interface._done_event.clear()
            interface._result_msg = None
        success, final_state, result_msg = execute_single_task(
            args=args,
            interface=interface,
            robot_controller=robot_controller,
            model=model,
            detector_client=detector_client,
            task_description=task_description,
        )
        should_write_result = True
        with interface._lock:
            if interface._done_event.is_set() and interface._result_msg is not None:
                should_write_result = False
        if should_write_result and wait_for_goal_handle(interface):
            result = ActionVLA.Result()
            result.success = bool(final_state == SUCCEEDED and success)
            result.final_state = final_state
            result.error_code = 0 if result.success else 1
            result.result_msg = result_msg
            with interface._lock:
                interface._result_msg = result
                interface._done_event.set()
        interface.vla_status["status"] = final_state
        interface.new_task_flag = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tyro.cli(main)
