import numpy as np

WHITE_DESK_OFFSET_JOINT3 = 0.0
WHITE_DESK_OFFSET_JOINT4 = 0.0

TV_OFFSET_JOINT3 = 0.0
TV_OFFSET_JOINT4 = 0.0

SOFA_OFFSET_JOINT3 = 0.0
SOFA_OFFSET_JOINT4 = 0.0

def get_location_name(target_location: str) -> str | None:
    if target_location == "white-desk":
        return "white-desk"
    if target_location == "TV-cabinet":
        return "TV-cabinet"
    if target_location == "sofa":
        return "sofa"
    return None


def get_left_arm_offset(target_location: str) -> tuple[float, float]:
    location = get_location_name(target_location)

    if location == "white-desk":
        return WHITE_DESK_OFFSET_JOINT3, WHITE_DESK_OFFSET_JOINT4
    elif location == "TV-cabinet":
        return TV_OFFSET_JOINT3, TV_OFFSET_JOINT4
    elif location == "sofa":
        return SOFA_OFFSET_JOINT3, SOFA_OFFSET_JOINT4

    return 0.0, 0.0


def add_left_arm_offset(joints, target_location: str):
    offset_j3, offset_j4 = get_left_arm_offset(target_location)

    if isinstance(joints, np.ndarray):
        joints = joints.copy()
    else:
        joints = list(joints)

    joints[2] += offset_j3
    joints[3] += offset_j4
    return joints


def subtract_left_arm_offset(joints, target_location: str):
    offset_j3, offset_j4 = get_left_arm_offset(target_location)

    if isinstance(joints, np.ndarray):
        joints = joints.copy()
    else:
        joints = list(joints)

    joints[2] -= offset_j3
    joints[3] -= offset_j4
    return joints