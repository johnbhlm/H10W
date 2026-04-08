import dataclasses
import sys
import time
import threading
from collections import deque

import numpy as np
import rclpy
from unitree_sdk2py.utils.thread import RecurrentThread

sys.path.append("/home/diana/intern-vla/starVLA/")

from examples.H10W.model2h10w_interface import ModelClient
from examples.H10W.robot_interface_temp import H10WInferfaceConfig, H10WInterface
from examples.H10W.robot_controller import RobotController
from vla.action import ActionVLA

# =========================
# 参数
# =========================
CONTROL_DT = 0.01
INFER_INTERVAL = 0.13
ACTION_DROP_N = 3
ACTION_QUEUE_SIZE = 64
FINISH_HOLD_TIME = 0.5
SMOOTH_WINDOW = 10


@dataclasses.dataclass
class Args:
    host: str = "192.168.0.10"
    port: int = 10093
    resize_size = [224, 224]
    pretrained_path: str = ""

# =========================
# 主控制类（双臂异步版）
# =========================
class AsyncH10WVLA:
    def __init__(self, args: Args):
        self.args = args

        # ---------- 状态 ----------
        self.current_task = None
        self.task_active = False
        self.stop_program = False

        self.step_counter = 0
        self.finish_start_time = None

        # ---------- target ----------
        self.left_target = None
        self.right_target = None

        # ---------- gripper smoothing（双臂） ----------
        self.action_buffer = {"left": [], "right": []}
        self.last_gripper_action = {"left": 0, "right": 0}
        
        self.grasp_state = {
            "left": {
                "holding": False,
                "item": None,
            },
            "right": {
                "holding": False,
                "item": None,
            }
        }
        
        self.allow_release = {
            "left": False,
            "right": False,
        }


        # ---------- action queue ----------
        self.action_queue = deque(maxlen=ACTION_QUEUE_SIZE)
        self.action_lock = threading.Lock()
        self.last_action = None
        
         # ---------- Robot ----------
        self.robot_controller = RobotController()

        # ---------- ROS ----------
        rclpy.init()
        self.interface = H10WInterface(H10WInferfaceConfig(),self.robot_controller)
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self.interface)
        threading.Thread(target=executor.spin, daemon=True).start()
        time.sleep(1.0)

        # ---------- Model ----------
        self.model = ModelClient(
            policy_ckpt_path=args.pretrained_path,
            host=args.host,
            port=args.port,
            image_size=args.resize_size,
        )


        # ---------- 推理线程 ----------
        self.inference_thread = RecurrentThread(
            interval=INFER_INTERVAL,
            target=self.inference_loop,
            name="inference",
        )

    # =========================================================
    # 推理线程（只产出 action）
    # =========================================================
    def inference_loop(self):
        if not self.task_active or self.current_task is None:
            return

        obs = self.interface.get_observations()
        if obs is None:
            return

        img_top = obs["head_rgb"]["data"]
        img_left = obs["left_rgb"]["data"]
        img_right = obs["right_rgb"]["data"]

        if img_top is None or img_left is None or img_right is None:
            return

        example = {
            "image": [img_top, img_left, img_right],
            "lang": self.current_task,
        }

        response = self.model.step(example=example)
        actions = response.get("raw_actions", None)
        if actions is None or len(actions) == 0:
            return

        actions = actions[ACTION_DROP_N:]

        with self.action_lock:
            for a in actions:
                self.action_queue.append(a)

    # =========================================================
    # 新任务处理
    # =========================================================
    def on_new_task(self, task_description: str):
        print(f"[TASK] {task_description}")
        
        # ---------- 语义短路 ----------
        for side in ["left", "right"]:
            if "pick" in task_description and f"{side} hand" in task_description:
                if self.grasp_state[side]["holding"]:
                    print(
                        f"[WARN] {side} already holding "
                        f"{self.grasp_state[side]['item']}, ignore task: {task_description}"
                    )
                    self.finish_task_immediately()
                    return

            if "place" in task_description:
                if "left" in task_description:
                    self.allow_release["left"] = True
                if "right" in task_description:
                    self.allow_release["right"] = True
                held_item = self.grasp_state[side]["item"]
                if held_item and held_item not in task_description:
                    print(
                        f"[WARN] {side} holding {held_item}, "
                        f"but task wants to place different object"
                    )
                    self.finish_task_immediately()
                    return

        if "pick" not in task_description and "place" not in task_description:
            self.finish_task_immediately()
            return

        self.current_task = task_description
        self.task_active = True
        self.step_counter = 0
        self.finish_start_time = None

        self.action_buffer["left"].clear()
        self.action_buffer["right"].clear()

        with self.action_lock:
            self.action_queue.clear()
            self.last_action = None

        self.robot_controller.h10w_system.enableController(True)

        # ---- target 逻辑（左臂给定，右臂镜像）----
        if (
            "sofa" in self.interface.target_location
            or "TV-cabinet" in self.interface.target_location
        ):  
            
            self.robot_controller.control_torso(torso=0.34)
            left = np.array([0.60, 0.60, 1.01])
        else:
            left = np.array([0.60, 0.60, 1.21])

        right = left.copy()
        right[1] *= -1

        self.left_target = left
        self.right_target = right
        
        self.robot_controller.h10w_motion.enableRealtimeCmd(True)

        self.interface.vla_status["status"] = 2

    # =========================================================
    # gripper smoothing
    # =========================================================
    def update_gripper(self, side: str, value: int):
        buf = self.action_buffer[side]
        buf.append(value)
        if len(buf) > SMOOTH_WINDOW:
            buf.pop(0)

        if len(buf) == SMOOTH_WINDOW:
            if all(x == 1 for x in buf):
                self.last_gripper_action[side] = 1
            elif all(x == 0 for x in buf):
                self.last_gripper_action[side] = 0
    
    def update_grasp_memory(self):
        current_item = self.interface.current_item

        for side in ["left", "right"]:
            # 只有在“允许释放 + 实际执行了 open”时，才清空
            if self.last_gripper_action[side] == 1:
                if not self.grasp_state[side]["holding"]:
                    self.grasp_state[side]["holding"] = True
                    self.grasp_state[side]["item"] = current_item
            else:
                if self.allow_release[side]:
                    self.grasp_state[side]["holding"] = False
                    self.grasp_state[side]["item"] = None

    def apply_gripper_guard(self, side, proposed):
        if self.grasp_state[side]["holding"] and not self.allow_release[side]:
            return 1  # 强制闭合
        return proposed

    
    def update_world_state(self):
        for side in ["left", "right"]:
            # 一旦真的松开，release 权限自动失效
            if not self.grasp_state[side]["holding"]:
                self.allow_release[side] = False
              
    def publish_grasp_status(self):
        for side in ["left", "right"]:
            if self.grasp_state[side]["holding"]:
                self.interface.vla_status[f"{side}_state"] = 1
                self.interface.vla_status[f"{side}_item"] = self.grasp_state[side]["item"]
            else:
                self.interface.vla_status[f"{side}_state"] = 0
                self.interface.vla_status[f"{side}_item"] = ""

    # =========================================================
    # 完成判断（双臂）
    # =========================================================
    def check_task_finished(self):
        status = self.robot_controller.get_status()
        if status is None:
            return False

        lpose = status.get("leftPose", None)
        rpose = status.get("rightPose", None)
        if lpose is None or rpose is None:
            return False

        dl = np.linalg.norm(np.array(lpose[:3]) - self.left_target[:3])
        dr = np.linalg.norm(np.array(rpose[:3]) - self.right_target[:3])

        self.step_counter += 1

        # ⚠️ 如果你只想看左臂：
        # both_close = dl < 0.1
        both_close = (dl < 0.1 and dr < 0.1)

        if both_close and self.step_counter > 500:
            if self.finish_start_time is None:
                self.finish_start_time = time.time()
            elif time.time() - self.finish_start_time >= FINISH_HOLD_TIME:
                return True
        else:
            self.finish_start_time = None

        return False

    # =========================================================
    # 控制主循环（100Hz）
    # =========================================================
    def control_loop(self):
        print("[INFO] Control loop started (100Hz)")
        self.inference_thread.Start()

        while not self.stop_program:
            t0 = time.time()

            if getattr(self.interface, "new_task_flag", False):
                self.on_new_task(self.interface.latest_instruction)
                self.interface.new_task_flag = False

            with self.action_lock:
                if self.action_queue:
                    action = self.action_queue.popleft()
                    self.last_action = action
                else:
                    action = self.last_action

            if action is not None and self.task_active:
                left_arm = action[0:7].tolist()
                left_gripper = int(action[7])
                right_arm = action[8:15].tolist()
                right_gripper = int(action[15])

                # 1. 模型原始意图
                self.update_gripper("left", left_gripper)
                self.update_gripper("right", right_gripper)

                # 2. smoothing 后的意图
                smoothed_l = self.last_gripper_action["left"]
                smoothed_r = self.last_gripper_action["right"]

                # 3. guard 最终裁决
                lg = self.apply_gripper_guard("left", smoothed_l)
                rg = self.apply_gripper_guard("right", smoothed_r)

                self.robot_controller.control_joints(
                    left_arm=left_arm,
                    right_arm=right_arm,
                    torso=None,
                    left_gripper=lg,
                    right_gripper=rg,
                    control_time=CONTROL_DT,
                )

                self.update_grasp_memory()
                self.update_world_state()
                self.publish_grasp_status()

            if self.task_active and self.check_task_finished():
                self.finish_task()

            time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))

    # =========================================================
    # 立即完成
    # =========================================================
    def finish_task_immediately(self):
        self.task_active = False

        result = ActionVLA.Result()
        result.success = True
        result.final_state = 3
        result.error_code = 0
        result.result_msg = "Task finished"

        with self.interface._lock:
            self.interface._result_msg = result
            self.interface._done_event.set()

    # =========================================================
    # 正常完成
    # =========================================================
    def finish_task(self):
        print("[INFO] Task finished")

        self.task_active = False
        self.interface.vla_status["status"] = 3

        self.robot_controller.h10w_motion.enableRealtimeCmd(False)
        self.robot_controller.control_torso(torso=0.54)
        self.robot_controller.h10w_system.enableController(False)

        result = ActionVLA.Result()
        result.success = True
        result.final_state = 3
        result.error_code = 0
        result.result_msg = "Task finished"

        with self.interface._lock:
            self.interface._result_msg = result
            self.interface._done_event.set()

        with self.action_lock:
            self.action_queue.clear()
            self.last_action = None


# =========================
# 入口
# =========================
def main():
    args = Args()
    app = AsyncH10WVLA(args)
    app.control_loop()


if __name__ == "__main__":
    main()
