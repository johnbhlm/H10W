import dataclasses
import sys
import time
import json
import threading
from collections import deque

import numpy as np
import torch
import rclpy
from pynput import keyboard

from unitree_sdk2py.utils.thread import RecurrentThread

sys.path.append("/home/maintenance/workspace/starVLA/")

from examples.H10W.model2h10w_interface import ModelClient
from examples.H10W.robot_interface import H10WInferfaceConfig, H10WInterface
from examples.H10W.robot_controller import RobotController


# =========================
# 参数
# =========================
CONTROL_DT = 0.01          # 100 Hz
INFER_INTERVAL = 0.13      # 推理线程频率（秒）
ACTION_DROP_N = 2          # 丢弃 rollout 前几个 action
ACTION_QUEUE_SIZE = 64


@dataclasses.dataclass
class Args:
    host: str = "10.8.8.78"
    port: int = 10093
    resize_size = [224, 224]
    pretrained_path: str = (
        "./results/Checkpoints/"
        "internvla_l1000_r1000_36k_finetune_no_instruction/"
        "checkpoints/steps_6000_pytorch_model.pt"
    )


# =========================
# 主类
# =========================
class H10W:
    def __init__(self):
        # -------- 任务 --------
        self.task_list = {
            "1": "pick the plush green dinosaur from the desk with the left hand",
            "2": "place the plush green dinosaur toy with the left hand",
            "3": "pick the plush yellow duck toy from the desk with the right hand",
            "4": "place the plush yellow duck toy on the desk with the right hand",
        }
        self.current_task_id = "1"

        # -------- 标志 --------
        self.stop_program = False

        # -------- Action Buffer（新增） --------
        self.action_queue = deque(maxlen=ACTION_QUEUE_SIZE)
        self.action_lock = threading.Lock()
        self.last_action = None

        # -------- ROS --------
        rclpy.init()
        self.interface = H10WInterface(H10WInferfaceConfig())
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self.interface)
        threading.Thread(target=executor.spin, daemon=True).start()

        time.sleep(1.0)

        # -------- Model --------
        self.args = Args()
        self.model = ModelClient(
            policy_ckpt_path=self.args.pretrained_path,
            host=self.args.host,
            port=self.args.port,
            image_size=self.args.resize_size,
        )

        # -------- Robot --------
        self.robot_controller = RobotController()

        # -------- Inference Thread（修改） --------
        self.inference_thread = RecurrentThread(
            interval=INFER_INTERVAL,
            target=self.inference,
            name="inference",
        )

    # =========================
    # 键盘
    # =========================
    def on_press(self, key):
        try:
            if key.char == "q":
                self.stop_program = True
            elif key.char in self.task_list:
                self.current_task_id = key.char
                print(f"[TASK] {self.task_list[key.char]}")
        except Exception:
            pass

    def start_keyboard_listener(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.daemon = True
        listener.start()

    # =========================
    # 推理线程（只产出 action）
    # =========================
    def inference(self):
        obs = self.interface.get_observations()
        if obs is None:
            return

        img_top = obs["head_rgb"]["data"]
        img_left = obs["left_rgb"]["data"]
        img_right = obs["right_rgb"]["data"]

        if img_top is None or img_left is None or img_right is None:
            return

        example = {
            "image": [img_top, img_left, img_right[:, :, ::-1]],
            "lang": self.task_list[self.current_task_id],
        }

        response = self.model.step(example=example)
        actions = response["raw_actions"]

        if actions is None or len(actions) == 0:
            return

        # -------- 丢弃前几个 rollout action（关键） --------
        actions = actions[ACTION_DROP_N:]

        with self.action_lock:
            for a in actions:
                self.action_queue.append(a)

        print(f"[INFER] push {len(actions)} actions, queue={len(self.action_queue)}")

    # =========================
    # 控制主循环（100Hz）
    # =========================
    def run(self):
        self.start_keyboard_listener()
        self.inference_thread.Start()

        print("[INFO] Control loop started (100Hz)")

        try:
            while not self.stop_program:
                t0 = time.time()

                with self.action_lock:
                    if len(self.action_queue) > 0:
                        action = self.action_queue.popleft()
                        self.last_action = action
                    else:
                        action = self.last_action

                if action is not None:
                    left_arm = action[0:7].tolist()
                    left_gripper = int(action[7])
                    right_arm = action[8:15].tolist()
                    right_gripper = int(action[15])

                    self.robot_controller.control_joints(
                        left_arm=left_arm,
                        right_arm=right_arm,
                        torso=None,
                        left_gripper=left_gripper,
                        right_gripper=right_gripper,
                        control_time=CONTROL_DT,
                    )

                time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))

        finally:
            print("[EXIT] Shutdown")


# =========================
# 入口
# =========================
if __name__ == "__main__":
    app = H10W()
    app.run()
