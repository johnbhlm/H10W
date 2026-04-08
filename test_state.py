

import humanoid_sdk_py.h10w as h10w

h10w_motion = h10w.H10wMotion()
h10w_status = h10w.H10wStatus()
h10w_system = h10w.H10wSystem()

def get_status():
        
        try:
            ret, move_msg = h10w_status.getMoveMessage()
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
            
            status = {
                "leftjoint": leftjoint,
                "rightjoint": rightjoint,
                "headjoint": headjoint,
                "torsojoint": torsojoint,
                "leftPose": leftPose,
                "rightPose": rightPose,
                "torsoPose": torsoPose,
            }
            return status
        except Exception as e:
            print(f"Failed to get robot status: {e}")
            return None

result = get_status()
print(result)