import time
import threading
import numpy as np
import rclpy
import cv2
import matplotlib.pyplot as plt

# ROS接口
from examples.H10W_robot.robot_interface import H10WInferfaceConfig, H10WInterface
# 模型推理
from examples.H10W_robot.controller_dual import M1Inference
#相机数据
from examples.H10W_robot.realsense import Camera
# 机器人控制
from examples.H10W_robot.robot_controller import RobotController

def read_frame(cap):
    """读取一帧，如果失败就循环播放"""
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame

def main():
    rclpy.init()

    # 初始化 ROS 接口
    interface = H10WInterface(H10WInferfaceConfig())

    # -----------------------------------------------------
    # 🔥 关键：启动 ROS 回调线程，否则永远收不到图像
    # -----------------------------------------------------
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(interface)

    exec_thread = threading.Thread(
        target=executor.spin,
        daemon=True
    )
    exec_thread.start()
    # -----------------------------------------------------

    print("[INFO] Executor thread started, waiting for messages...")

    # 等待订阅稳定
    time.sleep(1.0)
    
    # serial_number1_d455 = "333422304763"  # d455
    # serial_number1_d405_right = "218722271112" # d405-right
    # serial_number1_d405_left = "218622271722" # d405-left

    # camera_d455 = Camera(width=640, height=480, fps=30, serial_number=serial_number1_d455, name='455', calibrate_done=False)
    # camera_d405_right = Camera(width=640, height=480, fps=30, serial_number=serial_number1_d405_right, name='405_right', calibrate_done=False)
    # camera_d405_left = Camera(width=640, height=480, fps=30, serial_number=serial_number1_d405_left, name='405_left', calibrate_done=False)
    # video_top   = cv2.VideoCapture("/home/maintenance/H10_w/vla_code/InternVLA-M1/examples/H10W_robot/video/head.mp4")  
    # video_left  = cv2.VideoCapture("/home/maintenance/H10_w/vla_code/InternVLA-M1/examples/H10W_robot/video/left_hand.mp4")
    # video_right = cv2.VideoCapture("/home/maintenance/H10_w/vla_code/InternVLA-M1/examples/H10W_robot/video/right_hand.mp4")
    
    robot_controller = RobotController()
    # 初始化模型
    # agent = M1Inference(
    #     saved_model_path="/home/maintenance/Downloads/act_freezeqwen_h10w_q_pre1/checkpoints/steps_60000_pytorch_model.pt",
    #     policy_setup="real",
    #     use_bf16=False,
    #     action_ensemble=False,
    #     dual_freq_mode=False,
    #     planning_freq=2.0,
    # )
    model = M1Inference(
        policy_ckpt_path="/home/maintenance/vla_ws/InternVLA-M1/results/Checkpoints/act_freezeqwen_h10w_q_real_sl_pre5bl6sim/checkpoints/steps_20000_pytorch_model.pt", # to get unnormalization stats
        image_size=[224,224],
    )

    try:
        while True:
            # 从队列获取观测（此时回调线程已经持续填充数据）
            obs_dict = interface.get_observations()

            if obs_dict is None:
                time.sleep(0.01)
                continue

            print("=" * 30)
            print("Received:", obs_dict.keys())

            # 构造模型输入
            head_img = obs_dict.get("left_rgb", obs_dict.get("head_rgb"))
            left_img = obs_dict.get("left_rgb", obs_dict.get("left_rgb"))
            right_img = obs_dict.get("right_rgb", obs_dict.get("right_rgb"))

            # if head_img is None or right_img is None:
            #     print("[WARN] Missing image data, skip")
            #     continue
            
            # model_input = {
            #     "obs_camera_top": {"color_image": head_img["data"]}, 
            #     "obs_camera_left": {"color_image": right_img["data"]},
            #     "obs_camera_right": {"color_image": right_img["data"]},
            #     "robot": {"ee_pose_state": np.zeros(7, dtype=np.float32)}
            # }
            
            # color_455,depth_455 = camera_d455.get_data()
            # color_405,depth_405 = camera_d405.get_data()
            
            # img_top,depth_455 = camera_d455.get_data()
            # img_right,depth_405_right = camera_d405_right.get_data()
            # img_left,depth_405_left = camera_d405_left.get_data()
            
            # plt.imshow(img_left[..., ::-1])  # BGR → RGB
            # plt.show()
            
            # print("img_left shape:", img_left.shape)
            # print("img_right shape:", img_right.shape)
            # print("img_top shape:", img_top.shape)
            
            # model_input = {
            #     "obs_camera_left": {"color_image": img_left[:,:, ::-1]},  # BGR → RGB
            #     "obs_camera_right": {"color_image": img_right[:,:, ::-1]},  # BGR → RGB
            #     "obs_camera_top": {"color_image": img_top[:,:, ::-1]},  # BGR → RGB
            # }
            
            # plt.imshow(frame_top[..., ::-1])  # BGR → RGB
            # plt.show()
            
            # 转换成模型需要的格式
            # img_left  = cv2.cvtColor(image_left_resized,  cv2.COLOR_BGR2RGB)
            # img_right = cv2.cvtColor(image_right_resized, cv2.COLOR_BGR2RGB)
            # img_top   = cv2.cvtColor(image_top_resized,   cv2.COLOR_BGR2RGB)
            # print("img_left shape:", img_left.shape)
            # print("img_right shape:", img_right.shape)
            # print("img_top shape:", img_top.shape)
            # model_input = {
            #     "obs_camera_left": {"color_image": img_left},
            #     "obs_camera_right": {"color_image": img_right},
            #     "obs_camera_top": {"color_image": img_top}, 
            #     "robot": {"ee_pose_state": np.zeros(7, dtype=np.float32)}
            # }

            instruction = "Pick up the yellow duck."

            # 模型推理
            actions = model.step([head_img, left_img, right_img], instruction)["raw_actions"]
            print("actions:", actions)
            # print("actions:", actions.shape)
            
            for i in range(actions.shape[0]):
                last_action = actions[i]
                left_arm_joints = last_action[0:7].tolist()
                print("=" * 30)
                print("left_arm_joints:", left_arm_joints)
                # right_arm_joints = last_action[8:15].tolist()
                right_arm_joints = None
                torso_list = None
                control_time = 0.02
                robot_controller.control_joints(
                    left_arm=left_arm_joints,
                    right_arm=right_arm_joints,
                    torso=torso_list,
                    control_time=control_time)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
