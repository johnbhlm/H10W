import time
import pandas as pd
import numpy as np
import humanoid_sdk_py
import humanoid_sdk_py.h10w as h10w

# === 1. 初始化系统 ===
h10w_motion = h10w.H10wMotion()
h10w_system = h10w.H10wSystem()
time.sleep(1)

print("=== LeRobot state 回放控制 ===")

# === 2. 上电 + 清错 + 开抱闸 ===
def check_ret(ret, msg):
    if ret != 0:
        print(msg)
        exit(-1)
check_ret(h10w_system.controlBrake(humanoid_sdk_py.BrakeState.BRAKE_OFF), "关闭抱闸失败")
check_ret(h10w_system.clearError(), "清除错误失败")
check_ret(h10w_system.controlPower(humanoid_sdk_py.PowerState.POWER_ON), "上电失败")
check_ret(h10w_system.controlBrake(humanoid_sdk_py.BrakeState.BRAKE_ON), "打开抱闸失败")

# print("")
# === 3. 读取 LeRobot parquet 数据 ===
parquet_path = "/home/diana/intern-vla_test/starVLA/playground/data/lerobot_dataset_273/data/chunk-000/episode_000004.parquet"
# parquet_path = "episode_000000.parquet"
print(parquet_path)
df = pd.read_parquet(parquet_path)
print("read success")
if "observation.state" not in df.columns:
    print("❌ 文件中没有 'observation.state' 字段")
    exit(-1)

# state 结构： [L1, L2, L3, L4, L5, L6, L7, gripper, R1, R2, R3, R4, R5, R6, R7, ...]
states = np.stack(df["observation.state"].apply(np.array).to_numpy())  # shape (N, D)
print(f"✅ 成功读取 {len(states)} 帧状态，state 维度 {states.shape}")
# === 7. 关闭实时控制 ===
check_ret(h10w_motion.enableRealtimeCmd(False), "关闭实时指令失败")
# === 4. 先移动到初始位置（第一个 state） ===
first_state = states[0]
left_arm = first_state[:7]
right_arm = first_state[8:15]  # 跳过第8位夹爪

joint_targets = []
for i, angle in enumerate(left_arm):
    joint_targets.append((getattr(h10w.JointIndex, f"L_ARM_JOINT{i+1}"), float(angle), 0.05))
for i, angle in enumerate(right_arm):
    joint_targets.append((getattr(h10w.JointIndex, f"R_ARM_JOINT{i+1}"), float(angle), 0.05))

ret = h10w_motion.requestMultiJointsMove(joint_targets)
check_ret(ret, "移动到初始位置失败")
print("✅ 已到达初始姿态")
h10w_motion.waitMove(10000)

# === 5. 启用实时控制模式 ===
check_ret(h10w_motion.enableRealtimeCmd(True), "打开实时指令失败")

fps = 50   # 录制时帧率（可根据 lerobot 数据采样率调整）
dt = 1.0 / fps
# === 6. 按帧播放 state 轨迹 ===
joints = h10w.RealtimeJointsParams()
joints.left_arm_valid = True
joints.right_arm_valid = True
joints.torso_valid = False
joints.time = dt  # 每帧时间间隔，单位秒

for idx, state in enumerate(states):
    print(f"▶️ 播放第 {idx+1}/{len(states)} 帧", end='\r')
    print(state)
    left_arm_ = state[:7]
    right_arm_ = state[8:15]
    print("&" * 80)
    print("Left Arm Joints:", left_arm_)
    print("Right Arm Joints:", right_arm_)


    joints.left_arm = left_arm_.tolist()
    joints.right_arm = right_arm_.tolist()

    ret = h10w_motion.servoJoint(joints)
    if ret != 0:
        print(f"⚠️ 第 {idx} 帧 servoJoint 失败")
        break
    time.sleep(dt * 3)

    # ret, move_msg = h10w.H10wStatus().getMoveMessage()
    # leftjoint = move_msg.position[:7]
    # rightjoint = move_msg.position[7:14]
    # print("Left Arm status:", leftjoint)
    # print("Right Arm status:", rightjoint)

print("✅ 播放完成")

# === 7. 关闭实时控制 ===
# check_ret(h10w_motion.enableRealtimeCmd(False), "关闭实时指令失败")

