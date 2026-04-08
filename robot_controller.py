import time
from typing import List, Optional, Union
import numpy as np
from loguru import logger

# from examples.H10W_robot.vla_action_ros.build.vla.rosidl_generator_py.vla import action
import humanoid_sdk_py
import humanoid_sdk_py.h10w as h10w

import TekkenD
from TekkenD import TekkenDInterface

class RobotController:
    
    def __init__(self):
        self.initialized = False
        self.h10w_motion = h10w.H10wMotion()
        self.h10w_status = h10w.H10wStatus()
        self.h10w_system = h10w.H10wSystem()
        
        self.init_robot()
        self.init_gripper()
        self.init_position()
        
        self.last_left_gripper = None
        self.last_right_gripper = None
        self.dist = 0.0
        self.last_dist = 0.0
        self.gripper_done = False
    
    def check_ret(self, ret, msg):
        if ret != 0:
            print(msg)
            exit(-1)
    
    def init_robot(self) -> bool:
        self.check_ret(self.h10w_system.clearError(), "清除错误失败")
        self.check_ret(self.h10w_system.controlPower(humanoid_sdk_py.PowerState.POWER_ON), "上电失败")
        self.check_ret(self.h10w_system.clearError(), "清除错误失败")
        time.sleep(2)
        self.check_ret(self.h10w_system.clearError(), "清除错误失败")
        self.check_ret(self.h10w_system.controlBrake(humanoid_sdk_py.BrakeState.BRAKE_ON), "打开抱闸失败")
        self.check_ret(self.h10w_system.enableController(True),"打开失败")
        self.check_ret(self.h10w_motion.enableRealtimeCmd(False), "关闭实时指令失败")
        self.initialized = True
        logger.info("Robot controller initialized")
    
    def init_position(self) -> bool:
        # self.check_ret(self.h10w_system.clearError(), "清除错误失败")
        # self.check_ret(self.h10w_system.controlBrake(humanoid_sdk_py.BrakeState.BRAKE_ON), "打开抱闸失败")
        self.check_ret(self.h10w_motion.enableRealtimeCmd(False), "关闭实时指令失败")
        self.check_ret(self.h10w_system.enableController(True),"打开失败")
        #init_joints = [[0.034906678, -1.5009823, 0.17453226, 2.530736, 1.6525322, 1.5434607, -0.20944083], [-0.035274446, -1.4983919, -0.17670366, 2.536254, -1.660838, 1.5351231, 0.19762026]]
        init_joints = [[0.035274446, -1.4983919, 0.17670366, 2.536254, 1.660838, 1.5351231, -0.19762026], [-0.035274446, -1.4983919, -0.17670366, 2.536254, -1.660838, 1.5351231, 0.19762026]]
                    #[-0.035274446, -1.4983919, -0.17670366, 2.536254, -1.660838, 1.5351231, 0.19762026]

        left_arm = init_joints[0]
        right_arm = init_joints[1] 
        
        torso_height = 0.54
        head_pitch   = -0.6 # 低头一点
        head_yaw     = 0.0 

        joint_targets = []
        for i, angle in enumerate(left_arm):
            joint_targets.append((getattr(h10w.JointIndex, f"L_ARM_JOINT{i+1}"), float(angle), 0.1))
        for i, angle in enumerate(right_arm):
            joint_targets.append((getattr(h10w.JointIndex, f"R_ARM_JOINT{i+1}"), float(angle), 0.1))
        
        joint_targets.append(
            (h10w.JointIndex.ELEVATOR_MOTOR, float(torso_height), 0.1)
        )
        joint_targets.append(
            (h10w.JointIndex.HEAD_PITCH, float(head_pitch), 0.1)
        )
        # joint_targets.append(
        #     (h10w.JointIndex.HEAD_YAW, float(head_yaw), 0.05)
        # )

        ret = self.h10w_motion.requestMultiJointsMove(joint_targets)
        self.check_ret(ret, "移动到初始位置失败")
        self.h10w_motion.waitMove(100000)
        print("已到达初始姿态")
        # self.check_ret(self.h10w_system.enableController(False),"关闭失败")
        
        self.left_gripper.move_gripper_by_position(0, 80, 100)
        self.right_gripper.move_gripper_by_position(0, 80, 100)
        
    def init_left(self) -> bool:
        # self.check_ret(self.h10w_system.clearError(), "清除错误失败")
        # self.check_ret(self.h10w_system.controlBrake(humanoid_sdk_py.BrakeState.BRAKE_ON), "打开抱闸失败")
        self.check_ret(self.h10w_motion.enableRealtimeCmd(False), "关闭实时指令失败")
        self.check_ret(self.h10w_system.enableController(True),"打开失败")
        #init_joints = [[0.034906678, -1.5009823, 0.17453226, 2.530736, 1.6525322, 1.5434607, -0.20944083], [-0.035274446, -1.4983919, -0.17670366, 2.536254, -1.660838, 1.5351231, 0.19762026]]
        init_joints =[[0.035274446, -1.4983919, 0.17670366, 2.536254, 1.660838, 1.5351231, -0.19762026]]
        left_arm = init_joints[0]
        #right_arm = init_joints[1] 
        
        joint_targets = []
        for i, angle in enumerate(left_arm):
            joint_targets.append((getattr(h10w.JointIndex, f"L_ARM_JOINT{i+1}"), float(angle), 0.05))

        ret = self.h10w_motion.requestMultiJointsMove(joint_targets)
        self.check_ret(ret, "左臂重置失败")
        self.h10w_motion.waitMove(100000)
        print("左臂重置成功")
        
        # self.left_gripper.stop_gripper_movement()
        # self.left_gripper.move_gripper_by_position(0, 80, 100)
    
    def init_right(self) -> bool:
        # self.check_ret(self.h10w_system.clearError(), "清除错误失败")
        # self.check_ret(self.h10w_system.controlBrake(humanoid_sdk_py.BrakeState.BRAKE_ON), "打开抱闸失败")
        self.check_ret(self.h10w_motion.enableRealtimeCmd(False), "关闭实时指令失败")
        self.check_ret(self.h10w_system.enableController(True),"打开失败")
        init_joints = [[0.034906678, -1.5009823, 0.17453226, 2.530736, 1.6525322, 1.5434607, -0.20944083], [-0.035274446, -1.4983919, -0.17670366, 2.536254, -1.660838, 1.5351231, 0.19762026]]
        left_arm = init_joints[0]
        right_arm = init_joints[1] 
        
        joint_targets = []
        for i, angle in enumerate(right_arm):
            joint_targets.append((getattr(h10w.JointIndex, f"R_ARM_JOINT{i+1}"), float(angle), 0.05))
        
        ret = self.h10w_motion.requestMultiJointsMove(joint_targets)
        self.check_ret(ret, "右臂重置失败")
        self.h10w_motion.waitMove(100000)
        print("右臂重置成功")
        
        # self.right_gripper.stop_gripper_movement()
        # self.right_gripper.move_gripper_by_position(0, 80, 100)

    def init_position_home(self) -> bool:
        # self.check_ret(self.h10w_system.clearError(), "清除错误失败")
        # self.check_ret(self.h10w_system.controlBrake(humanoid_sdk_py.BrakeState.BRAKE_ON), "打开抱闸失败")
        self.check_ret(self.h10w_system.enableController(True),"打开失败")
        self.check_ret(self.h10w_motion.enableRealtimeCmd(False), "关闭实时指令失败")
        init_joints = [[0.034906678, -1.5009823, 0.17453226, 2.530736, 1.6525322, 1.5434607, -0.20944083], [-0.035274446, -1.4983919, -0.17670366, 2.536254, -1.660838, 1.5351231, 0.19762026]]
        left_arm = init_joints[0]
        right_arm = init_joints[1] 
        
        torso_height = 0.54
        head_pitch   = -0.6          # 低头一点
        head_yaw     = 0.0 

        joint_targets = []
        for i, angle in enumerate(left_arm):
            joint_targets.append((getattr(h10w.JointIndex, f"L_ARM_JOINT{i+1}"), float(angle), 0.1))
        for i, angle in enumerate(right_arm):
            joint_targets.append((getattr(h10w.JointIndex, f"R_ARM_JOINT{i+1}"), float(angle), 0.1))
        
        joint_targets.append(
            (h10w.JointIndex.ELEVATOR_MOTOR, float(torso_height), 0.1)
        )
        joint_targets.append(
            (h10w.JointIndex.HEAD_PITCH, float(head_pitch), 0.1)
        )
        # joint_targets.append(
        #     (h10w.JointIndex.HEAD_YAW, float(head_yaw), 0.05)
        # )

        ret = self.h10w_motion.requestMultiJointsMove(joint_targets)
        self.check_ret(ret, "移动到初始位置失败")
        self.h10w_motion.waitMove(100000)
        print("已到达初始姿态")
        # self.check_ret(self.h10w_system.enableController(False),"关闭失败")
        self.left_gripper.stop_gripper_movement()
        self.left_gripper.move_gripper_by_position(0, 80, 100)
        self.right_gripper.stop_gripper_movement()
        self.right_gripper.move_gripper_by_position(0, 80, 100)
        
    def init_gripper(self) -> bool:
        TekkenD.initialize()

        right_port = "/dev/ttyUSB1"
        left_port  = "/dev/ttyUSB0"

        self.right_gripper = TekkenDInterface(right_port)
        self.left_gripper  = TekkenDInterface(left_port)

        ret_r = self.right_gripper.initialize_gripper(0, 0.0)
        if ret_r == 0:
            print(f"[RIGHT] Init Success ({right_port})")
        else:
            print(f"[RIGHT] Init Failed ({right_port})")
        status_r = self.right_gripper.get_gripper_init_status()
        if status_r == 0:
            print("[RIGHT] Gripper Init finished")
        else:
            print("[RIGHT] Gripper Init failed")
            
        self.right_gripper.release_brake()
        self.right_gripper.switch_gripper_control_mode(0)
        self.right_gripper.set_gripper_max_force(50)

        ret_l = self.left_gripper.initialize_gripper(0, 0.0)
        if ret_l == 0:
            print(f"[LEFT] Init Success ({left_port})")
        else:
            print(f"[LEFT] Init Failed ({left_port})")

        status_l = self.left_gripper.get_gripper_init_status()
        if status_l == 0:
            print("[LEFT] Gripper Init finished")
        else:
            print("[LEFT] Gripper Init failed")

        self.left_gripper.release_brake()
        self.left_gripper.switch_gripper_control_mode(0)
        self.left_gripper.set_gripper_max_force(50)

        print("Both grippers initialized.")
        return True

    def get_status(self) -> Optional[dict]:
        """获取当前机器人状态"""
        if not self.initialized:
            return None
        
        try:
            ret, move_msg = self.h10w_status.getMoveMessage()
            if ret != 0 or move_msg is None:
                return None
            
            leftjoint = move_msg.position[:7]
            rightjoint = move_msg.position[7:14]
            headjoint = move_msg.position[14:16]
            torsojoint = move_msg.position[16]
            
            leftPose = rightPose = torsoPose = None

            for pose in move_msg.tcp_pose:# h10w.CartIndex.LEFT_ARM   h10w.CartIndex.RIGHT_ARM     h10w.CartIndex.TORSO（头）
                if pose.type == h10w.CartIndex.LEFT_ARM:
                    leftPose = pose.pose
                elif pose.type == h10w.CartIndex.RIGHT_ARM:
                    rightPose = pose.pose
                    
                elif pose.type == h10w.CartIndex.TORSO:
                    torsoPose = pose.pose
            
            ret, pos_left_gripper = self.left_gripper.get_gripper_position()
            ret, pos_right_gripper = self.right_gripper.get_gripper_position()
            ret, distance_left = self.left_gripper.get_gripper_distance()
            ret, distance_right = self.right_gripper.get_gripper_distance()
            # 假设夹爪最大开距为 40mm，则闭合度为 (40 - distance) / 40 映射到 [0.0, 1.0]
            # 这与 _ros_to_mm 是反向映射
            pos_left = max(0.0, min(1.0, (40.0 - distance_left) / 40.0))
            pos_right = max(0.0, min(1.0, (40.0 - distance_right) / 40.0))
            
            if pos_left > 0.4:
                pos_left = 1.0
            else:
                pos_left = 0
            if pos_right > 0.4:
                pos_right = 1.0
            else:
                pos_right = 0
            pos_left_gripper = [pos_left] 
            pos_right_gripper = [pos_right]
            
            status = {
                "leftjoint": leftjoint,
                "rightjoint": rightjoint,
                "headjoint": headjoint,
                "torsojoint": torsojoint,
                "leftPose": leftPose,
                "rightPose": rightPose,
                "torsoPose": torsoPose,
                "left_gripper": pos_left_gripper,
                "right_gripper": pos_right_gripper,
            }
            return status
        except Exception as e:
            logger.error(f"Failed to get robot status: {e}")
            return None
    
    def control_torso(self, torso: float, control_time: float = 0.001) -> bool:
        
        try:
            joint_target = (17, torso, 0.3)

            ret = self.h10w_motion.requestSingleJointMove(joint_target)
            if ret != 0:
                print("运动指令发送失败")
                exit(-1)
            else:
                print("运动指令发送成功")
                
            ret = self.h10w_motion.waitMove(10000)

            return True

        except Exception as e:
            logger.error(f"Torso control failed: {e}")
            return False
        
    def control_lpose(self,pose) -> bool:
        try:
            self.check_ret(self.h10w_motion.enableRealtimeCmd(False), "关闭实时指令失败")
            self.check_ret(self.h10w_system.enableController(True),"打开失败")
            pose_target = pose.copy()
            pose_target[0] -= 0.2
            pose_target[2] += 0.15
            # pose_target[1] -= 0.1
            target = h10w.LinearTarget()
            target.type = h10w.CartIndex.LEFT_ARM
            target.pose = pose_target
            target.velocityPercent = 0.3  # 10% of max velocity
            target.accelerationPercent = 0.2  # 10% of max acceleration
            linear_targets = [target]
            ret = self.h10w_motion.requestLinearMove(linear_targets)
            if ret != 0:
                print("运动指令发送失败")
                exit(-1)
            else:
                print("运动指令发送成功")

            ret = self.h10w_motion.waitMove(10000)
            return True
        except Exception as e:
            logger.error(f"Torso control failed: {e}")
            return False
        
    def control_rpose(self, pose) -> bool:
        try:
            self.check_ret(self.h10w_motion.enableRealtimeCmd(False), "关闭实时指令失败")
            self.check_ret(self.h10w_system.enableController(True),"打开失败")
            pose_target = pose.copy()
            pose_target[0] -= 0.2
            pose_target[2] += 0.15
            # pose_target[1] += 0.1
            target = h10w.LinearTarget()
            target.type = h10w.CartIndex.RIGHT_ARM
            target.pose = pose_target
            target.velocityPercent = 0.3  # 10% of max velocity
            target.accelerationPercent = 0.2  # 10% of max acceleration
            linear_targets = [target]
            ret = self.h10w_motion.requestLinearMove(linear_targets)
            if ret != 0:
                print("运动指令发送失败")
                exit(-1)
            else:
                print("运动指令发送成功")

            ret = self.h10w_motion.waitMove(10000)
            return True
        except Exception as e:
            logger.error(f"Torso control failed: {e}")
            return False

    # def control_lpose(self) -> bool:
    #     left_arm = [-0.5253599882125854, -0.9871698617935181, 0.43689918518066406, 2.1320202350616455, 1.9078527688980103, 1.5360443592071533, 0.04143471270799637]
    #     joint_targets = []
    #     for i, angle in enumerate(left_arm):
    #         joint_targets.append((getattr(h10w.JointIndex, f"L_ARM_JOINT{i+1}"), float(angle), 0.1))
        
    #     ret = self.h10w_motion.requestMultiJointsMove(joint_targets)
    #     self.check_ret(ret, "移动到初始位置失败")
    #     self.h10w_motion.waitMove(100000)
        
    # def control_rpose(self) -> bool:
    #     right_arm = [0.5253599882125854, -0.9871698617935181, -0.43689918518066406, 2.1320202350616455, -1.9078527688980103, 1.5360443592071533, -0.04143471270799637]

    #     joint_targets = []
    #     for i, angle in enumerate(right_arm):
    #         joint_targets.append((getattr(h10w.JointIndex, f"R_ARM_JOINT{i+1}"), float(angle), 0.1))
        
    #     ret = self.h10w_motion.requestMultiJointsMove(joint_targets)
    #     self.check_ret(ret, "移动到初始位置失败")
    #     self.h10w_motion.waitMove(100000)

    def control_joints(self, 
                  left_arm: Optional[List[float]] = None,
                  right_arm: Optional[List[float]] = None,
                  torso: Optional[float] = None,
                  left_gripper: Optional[float] = None,
                  right_gripper: Optional[float] = None,
                  control_time: float = 0.001) -> bool:
        try:
            # self.check_ret(self.h10w_system.enableController(True),"打开失败")
            # ret = self.h10w_motion.enableRealtimeCmd(True)
            # if ret != 0:
            #     print("打开实时指令失败")
            #     exit(-1)
            joints = h10w.RealtimeJointsParams()
            joints.time = control_time
            
            if left_arm is not None:
                if len(left_arm) != 7:
                    logger.error(f"left_arm expects 7 dims, got {len(left_arm)}")
                    return False
                joints.left_arm = left_arm
                joints.left_arm_valid = True

            if right_arm is not None:
                if len(right_arm) != 7:
                    logger.error(f"right_arm expects 7 dims, got {len(right_arm)}")
                    return False
                joints.right_arm = right_arm
                joints.right_arm_valid = True

            if torso is not None:
                joints.torso = float(torso)
                joints.torso_valid = True

            ret = self.h10w_motion.servoJoint(joints)
            if ret != 0:
                logger.error("servoJoint failed")
                return False

            if left_gripper is not None:
                target_action = float(left_gripper)
                if target_action > 0.5:
                    target_position=37
                else:
                    target_position=0
                if self.last_left_gripper != target_action:
                    self.left_gripper.stop_gripper_movement()
                    self.left_gripper.move_gripper_by_position(target_position, 50, 50)
                    self.last_left_gripper = target_action
                    # arrived = 0
                    # while arrived != 1:
                    #     arrived = self.left_gripper.get_gripper_arriving_signal()
                        # time.sleep(0.005)
                    
            if right_gripper is not None:
                target_action = float(right_gripper)
                if target_action > 0.5:
                    target_position=37
                else:
                    target_position=0
                if self.last_right_gripper != target_action:
                    self.right_gripper.stop_gripper_movement()
                    self.right_gripper.move_gripper_by_position(target_position, 50, 50)
                    self.last_right_gripper = target_action
                    # arrived = 0
                    # while arrived != 1:
                    #     arrived = self.right_gripper.get_gripper_arriving_signal()
                        # time.sleep(0.005)

            return True

        except Exception as e:
            logger.error(f"Joint or gripper control failed: {e}")
            return False 
    def control_pose(self,
                    left_pose: Optional[List[float]] = None,
                    right_pose: Optional[List[float]] = None,
                    control_time: float = 0.001) -> bool:
        try:
            pose = h10w.RealtimePoseParams()
            pose.time = control_time
            
            # 设置左臂末端位姿
            if left_pose is not None and len(left_pose) == 7:
                pose.left_pose = h10w.Pose(
                    left_pose[0], left_pose[1], left_pose[2],  # x, y, z
                    left_pose[3], left_pose[4], left_pose[5], left_pose[6]  # qx, qy, qz, qw
                )
                pose.left_pose_valid = True
            
            # 设置右臂末端位姿
            if right_pose is not None and len(right_pose) == 7:
                pose.right_pose = h10w.Pose(
                    right_pose[0], right_pose[1], right_pose[2],  # x, y, z
                    right_pose[3], right_pose[4], right_pose[5], right_pose[6]  # qx, qy, qz, qw
                )
                pose.right_pose_valid = True
            
            ret = self.h10w_motion.servoPose(pose)
            return ret == 0
            
        except Exception as e:
            logger.error(f"Pose control failed: {e}")
            return False
    
    def get_current_joints(self) -> Optional[List[float]]:
        """获取当前所有关节角度"""
        if not self.initialized:
            return None
        
        try:
            ret, move_msg = self.h10w_status.getMoveMessage()
            if ret == 0 and move_msg is not None:
                return move_msg.position
            return None
        except:
            return None
    
    def cleanup(self):
        """清理资源"""
        if self.initialized:
            self.enable_realtime(False)
            logger.info("Robot controller cleaned up")


if __name__== '__main__':
    import numpy as np
    robot_controller = RobotController()
    try:
        while True:
            status = robot_controller.get_status()
            if status:
                print("Left Joint Angles:", status["leftjoint"])
                print("Right Joint Angles:", status["rightjoint"])
                print("Left Gripper Position:", status["left_gripper"])
                print("Right Gripper Position:", status["right_gripper"])
                left_arm = status["leftjoint"]
                right_arm = status["rightjoint"]
                left_gripper = status["left_gripper"]
                right_gripper = status["right_gripper"]
                state = np.array(left_arm + left_gripper + right_arm + right_gripper)

                print(state)
                print("Shape =", state.shape)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        robot_controller.cleanup()