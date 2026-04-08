import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from vla.action import ActionVLA
from vla.msg import VLAStatus

class VLAClient(Node):
    def __init__(self):
        super().__init__('vla_client')

        # 创建 ActionClient
        self.client = ActionClient(self, ActionVLA, 'vla_actions')

        # 待执行的动作队列
        self.task_list = [
            dict(action_id=1, action_type="pick",   target_object="the plush green dinosaur toy",  target_location="sofa",   gripper_hand="right_hand", fixed_hand=0),
            dict(action_id=2, action_type="place",  target_object="the green dinosaur",  target_location="box",     gripper_hand="right_hand", fixed_hand=0),
            dict(action_id=3, action_type="pick",   target_object="the plush yellow duck toy", target_location="desk", gripper_hand="left_hand",  fixed_hand=0),
            dict(action_id=4, action_type="place",   target_object="the yellow duck", target_location="desk", gripper_hand="left_hand",  fixed_hand=0),
            dict(action_id=5, action_type="pick",   target_object="the orange lion toy", target_location="desk", gripper_hand="left_hand",  fixed_hand=0),
            dict(action_id=6, action_type="place",   target_object="the orange lion", target_location="desk", gripper_hand="left_hand",  fixed_hand=0),
        ]
        self.task_index = 0

        self.get_logger().info("Waiting for action server...")
        self.client.wait_for_server()

        self.send_next_goal()

    # -------------------------------------------------------
    #  发送下一条指令
    # -------------------------------------------------------
    def send_next_goal(self):

        if self.task_index >= len(self.task_list):
            self.get_logger().info("🎉 All tasks finished!")
            return

        task = self.task_list[self.task_index]

        self.get_logger().info(
            f"🔵 Sending task {task['action_id']}: "
            f"{task['action_type']} {task['target_object']}"
        )

        goal_msg = ActionVLA.Goal()
        goal_msg.action_id        = task["action_id"]
        goal_msg.action_type      = task["action_type"]
        goal_msg.target_object    = task["target_object"]
        goal_msg.target_location  = task["target_location"]
        goal_msg.gripper_hand     = task["gripper_hand"]
        goal_msg.fixed_hand       = task["fixed_hand"]

        self.future = self.client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self.future.add_done_callback(self.goal_response_callback)

    # -------------------------------------------------------
    #  服务器收到 goal 的回调
    # -------------------------------------------------------
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("❌ Action rejected by server!")
            self.task_index += 1
            self.send_next_goal()
            return

        self.get_logger().info("🟢 Action accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    # -------------------------------------------------------
    #  接收执行过程反馈
    # -------------------------------------------------------
    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f"📢 Feedback: state={fb.current_state}, "
            f"progress={fb.progress:.2f}, "
            f"msg={fb.feedback_msg}"
        )

    # -------------------------------------------------------
    #  接收完成结果
    # -------------------------------------------------------
    def result_callback(self, future):
        result = future.result().result

        self.get_logger().info(
            f"🏁 Result received: success={result.success}, "
            f"state={result.final_state}, "
            f"msg={result.result_msg}"
        )

        # 下一条指令
        self.task_index += 1
        self.send_next_goal()


def main(args=None):
    rclpy.init(args=args)
    node = VLAClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
