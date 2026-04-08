from collections import deque
from dataclasses import dataclass, field
from functools import partial
import threading
import time
from typing import Dict, Literal, Optional, List

import cv2
import numpy as np
import datetime

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from loguru import logger
from sensor_msgs.msg import CompressedImage, JointState,Image

from vla.action import ActionVLA,ResetVLA
from vla.msg import VLAStatus
from std_msgs.msg import Header
from rclpy.action import ActionClient, ActionServer
from rclpy.task import Future
from rclpy.callback_groups import (
    CallbackGroup,
    ReentrantCallbackGroup,
    MutuallyExclusiveCallbackGroup
)
import asyncio


def get_readable_timestamp(stamp):
    return datetime.datetime.fromtimestamp(
                    stamp
                ).strftime("%H:%M:%S.%f")[:-3]

@dataclass
class H10WTopicsConfig:
    image_input: Dict[str, str] = field(
        default_factory=lambda: {
            "head_rgb": "/h10_w/head/color/image_raw",
            # "head_rgb": "/camera/camera/color/image_raw/compressed",
            "left_rgb": "/h10_w/left_wrist/color/image_rect_raw",
            "right_rgb": "/h10_w/right_wrist/color/image_rect_raw",
        }
    )

    depth_input: Dict[str, str] = field(
        default_factory=lambda: {
            "head_depth": "/h10_w/head/aligned_depth_to_color/image_raw",
        }
    )
    
@dataclass
class H10WInferfaceConfig:
    topic: H10WTopicsConfig = field(default_factory=H10WTopicsConfig)
    msg_time_diff_threshold: float = 0.5

    with_torso: bool = False
    with_chassis: bool = False

    camera_deque_length: int = 10
    deque_length: int = 200

    control_freq: int = 50
    dry_run: bool = False
    torso_chassis_thres: float = 0.01
    use_direct_control: bool = True

class H10WInterface(Node):
    def __init__(self, config: H10WInferfaceConfig,robot_controller):
        # ROS2节点初始化
        super().__init__("H10W_real")
        self.config = config
        self.robot_controller = robot_controller
        self.br = CvBridge()
        self.inputs_dict = {}
        self.me_subscribers = {}
        self.me_publishers = {}
        self.last_camera_time = 0.
        self.latest_instruction = ""
        
        # QoS配置
        self.qos_best_effort = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.qos_reliable = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        ) 
        
        self._init_topics()
        time.sleep(1)
        
        self._action_server = ActionServer(
            self,
            ActionVLA,
            'vla_actions',
            execute_callback=self._execute_goal_callback,
            goal_callback=self._handle_goal_callback,
            cancel_callback=self._handle_cancel_callback
        )
        
        # ===== reset action server =====
        self._reset_action_server = ActionServer(
            self,
            ResetVLA,
            'reset_action',
            execute_callback=self._reset_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.latest_instruction = None  # 初始化为空

        # 发布 VLA 状态
        self.vla_status_pub = self.create_publisher(
            VLAStatus,
            "vla_status",
            self.qos_reliable
        )
        # 每 0.2 秒发布一次状态
        self.status_timer = self.create_timer(
            0.5,
            self.publish_vla_status
        )
        
        # 内部状态变量（可随时更新）
        self.vla_status = {
            "status": 1,        # 1=ready
            "left_state": 0,
            "right_state": 0,
            "left_item": "",
            "right_item": ""
        }
        
        self.last_action_id = None
        self._current_goal_handle = None
        
        # 用于 execute callback 和主线程之间同步
        self._result_msg = None              
        self._done_event = threading.Event() # 主线程完成时 set()，execute_callback 在等待
        self._lock = threading.Lock()
        
        self.current_item = None
        self.target_location = None
        self.new_task_flag = False
        
    def _reset_callback(self, goal_handle):
        self.get_logger().info("[RESET] Received reset command")

        try:
            # 调用 RobotController 复位函数
            self.robot_controller.init_position_home()  # 你需要在 RobotController 中实现这个函数
            # 2. 同步更新内部 VLA 状态
            self.vla_status["status"] = 1
            self.vla_status["left_state"] = 0
            self.vla_status["right_state"] = 0
            self.vla_status["left_item"] = ""
            self.vla_status["right_item"] = ""
            time.sleep(1)
            self.get_logger().info("[RESET] Robot reset to initial pose done")
        except Exception as e:
            self.get_logger().error(f"[RESET] Error resetting robot: {e}")
            result_msg = ResetVLA.Result()
            result_msg.success = False
            result_msg.final_state = -1
            result_msg.result_msg = str(e)
            goal_handle.abort()
            return result_msg

        # 构造返回结果
        result_msg = ResetVLA.Result()
        result_msg.success = True
        result_msg.final_state = 1
        result_msg.result_msg = "Reset successful"
        goal_handle.succeed()
        return result_msg


    def _camera_callback(self, msg: Image, que: deque, topic: str):
        try:
            is_depth = "depth" in topic
            # 16UC1 depth 
            if is_depth:
                img = self.br.imgmsg_to_cv2(msg,desired_encoding="16UC1")  # depth 保持原格式
            else:
                img_bgr = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            que.append(
                dict(
                    data=img,
                    message_time=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    is_depth=is_depth
                )
            )
        except Exception as e:
            logger.error(f"Camera callback error: {e}")
    
    def build_instruction(
        self,
        action_type: str,
        detailed_item: str,
        target_location: str,
        gripper_hand: str,
        fixed_hand: int,
    ):
        """
        构造给 VLA 的 language 指令
        """
        # 基础指令
        if action_type == "pick":
            if target_location:
                instruction = f"pick {detailed_item} from the {target_location}"
            else:
                instruction = f"pick {detailed_item}"
        elif action_type == "place":
            instruction = f"place {detailed_item} to the {target_location}"
        else:
            instruction = f"{action_type}"

        # 是否强制手
        if fixed_hand == 1 and gripper_hand:
            # gripper_hand: "left_hand" / "right_hand"
            hand_str = "left hand" if "left" in gripper_hand else "right hand"
            instruction = f"{instruction} with the {hand_str}"

        return instruction
    
    def _handle_goal_callback(self, goal_request):
        # if goal_request.action_type not in ["pick", "place"]:
        #     logger.warning(
        #         f"[Action Server] Reject action_type={goal_request.action_type}, "
        #         "only support [pick, place]"
        #     )
        #     return rclpy.action.GoalResponse.REJECT
        logger.info(f"[Action Server] Received Goal: {goal_request.action_type} {goal_request.target_object}")
        object_map = {
            "dog": "the brown dog toy",
            "duck": "the plush yellow duck toy",
            "dinosaur": "the plush green dinosaur toy",
            "lion": "the orange lion toy",
            "pepper": "the red pepper",
            "eggplant": "the eggplant",
            "rabbit": "the pink rabbit toy"
        }

        if goal_request.target_object in object_map:
            detailed_item = object_map[goal_request.target_object]
        else:
            detailed_item = f"the {goal_request.target_object}"
            
        self.latest_instruction = self.build_instruction(
            action_type=goal_request.action_type,
            detailed_item=detailed_item,
            target_location=goal_request.target_location,
            gripper_hand=goal_request.gripper_hand,
            fixed_hand=goal_request.fixed_hand,
        )

        logger.info(f"lastest instruction: {self.latest_instruction}")

        # self.latest_instruction = f"{goal_request.action_type} {detailed_item}"
        self.last_action_id = goal_request.action_id
        # self.latest_instruction = f"{goal_request.action_type} {goal_request.target_object}"
        self._cancel_requested = False
        self.current_item = goal_request.target_object
        self.target_location = goal_request.target_location
        self.new_task_flag = True
        return rclpy.action.GoalResponse.ACCEPT

    def _handle_cancel_callback(self, goal_handle):
        logger.warning(f"[Action Server] Goal canceled: {goal_handle.request.action_id}")
        self._cancel_requested = True
        self.vla_status["status"] = 1  
        # If an execute callback is waiting, set a canceled result and wake it up
        with self._lock:
            if self._current_goal_handle is not None and self._current_goal_handle.goal_id == goal_handle.goal_id:
                result_msg = ActionVLA.Result()
                result_msg.success = False
                result_msg.final_state = 0
                result_msg.error_code = -3
                result_msg.result_msg = "canceled"
                self._result_msg = result_msg
                self._done_event.set()
                # do not call goal_handle.canceled() here — execute_callback will return the result
        return rclpy.action.CancelResponse.ACCEPT
    
    def _execute_goal_callback(self, goal_handle):
        # NOTE: 这是同步函数（不是 async def），会在 executor 的线程里运行并阻塞等待主线程完成任务。
        self.get_logger().info("goal accepted, executing outside")
        with self._lock:
            self._current_goal_handle = goal_handle
            # 清除之前的结果/事件
            self._result_msg = None
            self._done_event.clear()
            self._cancel_requested = False
            
        feedback_msg = ActionVLA.Feedback()
        # 等待主线程把结果设置并 set() event
        # 超时时间可按需调整；这里用循环 + timeout 以便可在等待期间检查节点何时关闭
        while rclpy.ok() and not self._done_event.wait(timeout=0.1):
            # 你也可以在这里发布周期性的 feedback（如果需要）
            # Publish feedback from current status
            try:
                with self._lock:
                    # use available vla_status snapshot
                    status_snapshot = dict(self.vla_status)
                feedback_msg.current_state = int(status_snapshot.get("status", 1))
                # progress float in [0,1] if available else 0.0
                feedback_msg.progress = float(status_snapshot.get("progress", 0.0))
                goal_handle.publish_feedback(feedback_msg)
            except Exception as e:
                logger.debug(f"Failed to publish feedback: {e}")

            # if client requested cancel via goal_handle API, set canceled result and exit
            if goal_handle.is_cancel_requested or self._cancel_requested:
                logger.info("Goal cancel detected in execute loop")
                with self._lock:
                    result_msg = ActionVLA.Result()
                    result_msg.success = False
                    result_msg.final_state = 0
                    result_msg.error_code = -3
                    result_msg.result_msg = "canceled"
                    self._result_msg = result_msg
                    self._done_event.set()
                break

        # 如果节点被要求关闭或出现异常，返回一个默认失败结果
        if not rclpy.ok():
            res = ActionVLA.Result()
            res.success = False
            res.final_state = 0
            res.error_code = -1
            res.result_msg = "node shutdown"
            return res

        # 返回主线程设置好的结果对象
        with self._lock:
            result = self._result_msg
            self._current_goal_handle = None

        # 保证不返回 None（Action Server 会断言类型），如果为空，则返回失败 result
        if result is None:
            res = ActionVLA.Result()
            res.success = False
            res.final_state = 0
            res.error_code = -2
            res.result_msg = "no result set"
            return res

        # 正常返回 result
        goal_handle.succeed()
        return result
    def publish_vla_status(self):
        msg = VLAStatus()

        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.status = self.vla_status["status"]
        msg.left_state = self.vla_status["left_state"]
        msg.right_state = self.vla_status["right_state"]
        msg.left_item = self.vla_status["left_item"]
        msg.right_item = self.vla_status["right_item"]

        self.vla_status_pub.publish(msg)
        
    def _joint_states_callback(self, msg: JointState, que: deque, topic: str):
        try:
            que.append(
                dict(
                    position=np.array(msg.position, dtype=np.float32),
                    velocity=np.array(msg.velocity, dtype=np.float32),
                    message_time=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                )
            )
        except Exception as e:
            logger.error(f"Joint state callback error: {e}")

    def _init_topics(self):
        config_topic = self.config.topic
        
        # 动态构建订阅配置
        subscription_configs = []
        
        # 只有存在话题时才添加到配置中
        if hasattr(config_topic, 'image_input') and config_topic.image_input:
            subscription_configs.append({
                'topic_dict': config_topic.image_input,
                'topic_type': Image,  # 或者 Image，根据实际话题类型
                'callback': self._camera_callback,
                'deque_maxlen': self.config.camera_deque_length
            })
                 
        if hasattr(config_topic, 'depth_input') and config_topic.depth_input:
            subscription_configs.append({
                'topic_dict': config_topic.depth_input,
                'topic_type': Image,  # 深度图像类型
                'callback': self._camera_callback,
                'deque_maxlen': self.config.camera_deque_length
            })
            
        if hasattr(config_topic, 'joint_state_input') and config_topic.joint_state_input:
            subscription_configs.append({
                'topic_dict': config_topic.joint_state_input,
                'topic_type': JointState,
                'callback': self._joint_states_callback,
                'deque_maxlen': self.config.deque_length
            })
        self.group : Optional[CallbackGroup] = ReentrantCallbackGroup()
        # 初始化订阅者
        for config in subscription_configs:
            for topic, topic_name in config['topic_dict'].items():
                torso_flag = "torso" not in topic or self.config.with_torso
                chassis_flag = "chassis" not in topic or self.config.with_chassis
                if torso_flag and chassis_flag:
                    self.inputs_dict[topic] = deque(maxlen=config['deque_maxlen'])
                    
                    self.me_subscribers[topic] = self.create_subscription(
                        config['topic_type'],
                        topic_name,
                        partial(config['callback'], que=self.inputs_dict[topic], topic=topic),
                        self.qos_best_effort,
                        callback_group=self.group
                    )
                    logger.info(f"Subscriber {topic} created.")

        # 动态构建发布配置
        publish_configs = []
        
        if hasattr(config_topic, 'joint_state_output') and config_topic.joint_state_output:
            publish_configs.append({
                'topic_dict': config_topic.joint_state_output,
                'topic_type': JointState
            })
        
        if hasattr(config_topic, 'twist_output') and config_topic.twist_output:
            publish_configs.append({
                'topic_dict': config_topic.twist_output,
                'topic_type': TwistStamped
            })
        
        # 初始化发布者
        for config in publish_configs:
            for topic, topic_name in config['topic_dict'].items():
                torso_flag = "torso" not in topic or self.config.with_torso
                chassis_flag = "chassis" not in topic or self.config.with_chassis
                if torso_flag and chassis_flag:
                    self.me_publishers[topic] = self.create_publisher(
                        config['topic_type'], topic_name, self.qos_reliable
                    )
                    logger.info(f"Publisher {topic} created.")

    def find_nearest_message(self, topic_name: str, timestamp: float) -> Optional[dict]:
        min_diff = 100.0
        nearest_msg = None
        data_queue = self.inputs_dict[topic_name].copy()
        
        for msg in data_queue:
            diff = abs(msg['message_time'] - timestamp)
            if diff < min_diff:
                nearest_msg = msg
                min_diff = diff
        return nearest_msg

    def lookup_by_camera_under_tolerance(self, camera_timestamp: float, threshold: float) -> Optional[Dict]:
        msgs = {}
        for topic, que in self.inputs_dict.items():
            if len(que) == 0:
                logger.warning(f'Channel: {topic} has no messages')
                return None
                
            if topic == "head_rgb":  # 修正为head_rgb
                msgs[topic] = que[-1]
            else:
                msg = self.find_nearest_message(topic, camera_timestamp)
                if msg is None:
                    logger.warning(f'Channel: {topic} does not have any message')
                    return None
                    
                time_diff = abs(msg['message_time'] - camera_timestamp)
                # print("^" * 80)
                # print("hand_time",msg['message_time'])
                # print(time_diff)
                if time_diff > threshold:    
                    if topic not in ['left_rgb', 'right_rgb']:
                        logger.warning(f'Channel: {topic} does not satisfy threshold: {threshold} > {time_diff}')
                        return None
                msgs[topic] = msg
        return msgs

    def get_observations(self) -> Optional[Dict]:
        if "head_rgb" not in self.inputs_dict or len(self.inputs_dict["head_rgb"]) == 0:
            logger.warning('No camera_head message')
            return None

        if "head_depth" not in self.inputs_dict or len(self.inputs_dict["head_depth"]) == 0:
            logger.warning("No head_depth message")
            return None
        
        latest_camera_msg = self.inputs_dict["head_rgb"][-1]
        latest_camera_time = latest_camera_msg['message_time']
        
        obs = self.lookup_by_camera_under_tolerance(latest_camera_time, self.config.msg_time_diff_threshold)

        if obs is None:
            logger.warning('Failed to get latest_msgs')
            return None

        self.last_camera_time = latest_camera_time
        return obs

    def is_close(self) -> bool:
        return not rclpy.ok()

    def get_latest_instruction(self) -> str:
        return self.latest_instruction

    def destroy(self):
        super().destroy_node()

from robot_controller import RobotController
def main():
    rclpy.init()
    
    config = H10WInferfaceConfig()
    interface = H10WInterface(config,robot_controller=RobotController())
    print("=" * 30)
    print(interface.inputs_dict)
    
    try:
        # 先等待一段时间让消息到达
        print("Waiting for initial messages...")
        wait_start = time.time()
        while time.time() - wait_start < 3.0:  # 等待3秒
            rclpy.spin_once(interface, timeout_sec=0.1)
            # 检查是否有数据
            if any(len(que) > 1 for que in interface.inputs_dict.values()):
                print("Messages received!")
                break
            print(".", end="", flush=True)
    except :
        pass
        print("\n" + "=" * 30)
    # 再次打印队列状态
    print("After waiting:")
    for topic, que in interface.inputs_dict.items():
        print(f"  {topic}: {len(que)} messages")
        
    try:
        # 测试循环
        for i in range(10):
            if interface.is_close():
                break
                
            # time.sleep(0.1)
            obs = interface.get_observations()
            if obs:
                for k, v in obs.items():
                    if 'data' in v:
                        print(k, v['data'].shape, end=" ")
                    else:
                        print(k, v['position'].shape, v['velocity'].shape, end=" ")
                print("")
            
            # 处理ROS2事件
            rclpy.spin_once(interface, timeout_sec=0.1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        interface.destroy()
        rclpy.shutdown()

if __name__ == "__main__":
    main()