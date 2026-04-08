#!/usr/bin/env python3
"""
3D检测客户端 - 在Jetson上使用

功能：
1. 获取图像、深度、关节角度
2. 发送到服务器进行检测
3. 接收hand判断结果（基于世界坐标系）
4. 支持 save_debug 按需触发服务端保存调试信息

集成到VLA流程：
  hand = client.detect_hand(color, depth, torso_height, tilt_angle, pan_angle, "yellow duck toy")
  prompt = f"Use your {hand} hand to pick the yellow duck toy"
"""

import asyncio
import base64
import logging
from typing import Optional

import cv2
import numpy as np
import websockets.asyncio.client

try:
    import msgpack_numpy as msgpack
except ImportError:
    import msgpack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionClient3D:
    """3D检测客户端"""

    def __init__(self, server_url: str = "ws://localhost:8000", timeout: float = 30.0):
        self.server_url = server_url
        self.timeout = timeout

    async def detect_hand(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        torso_height: float,
        tilt_angle: float,
        pan_angle: float,
        object_name: str,
        table_prompt: Optional[str] = None,
        save_debug: bool = False,
    ) -> dict:
        """
        检测并返回建议的手

        Args:
            color_img: BGR彩色图 (H, W, 3)
            depth_img: 深度图 (H, W), uint16, 单位mm
            torso_height: torso高度
            tilt_angle: 头部tilt角度（弧度）
            pan_angle: 头部pan角度（弧度）
            object_name: 目标物体名称
            table_prompt: 可选，桌子描述
            save_debug: 是否请求服务端保存本次调试信息

        Returns:
            {
                "success": bool,
                "hand": "left" or "right",
                "object_xy": [x, y],
                "object_bbox": [x1, y1, x2, y2],
                ...
            }
        """
        # 编码彩色图
        ok, color_encoded = cv2.imencode(".jpg", color_img)
        if not ok:
            return {
                "success": False,
                "error": "彩色图编码失败"
            }

        # 深度图必须保持原始 uint16
        if depth_img.dtype != np.uint16:
            logger.warning(f"depth_img dtype is {depth_img.dtype}, converting to uint16")
            depth_img = depth_img.astype(np.uint16)

        depth_bytes = depth_img.tobytes()

        request = {
            "type": "detect",
            "color": base64.b64encode(color_encoded.tobytes()).decode(),
            "depth": base64.b64encode(depth_bytes).decode(),
            "object": object_name,
            "torso_height": float(torso_height),
            "tilt_angle": float(tilt_angle),
            "pan_angle": float(pan_angle),
            "save_debug": bool(save_debug),
        }

        if table_prompt:
            request["table_prompt"] = table_prompt

        logger.info(
            f"发送检测请求: object={object_name}, "
            f"save_debug={save_debug}, "
            f"color_shape={color_img.shape}, depth_shape={depth_img.shape}"
        )

        async with websockets.asyncio.client.connect(
            self.server_url,
            compression=None,
            max_size=None
        ) as websocket:

            await websocket.send(msgpack.packb(request))
            response_data = await asyncio.wait_for(
                websocket.recv(),
                timeout=self.timeout
            )
            response = msgpack.unpackb(response_data)

            if response.get("success"):
                hand = response.get("hand")
                xy = response.get("object_xy")
                if xy is not None:
                    logger.info(
                        f"✅ 检测成功: hand={hand}, world_xy=({xy[0]:.3f}, {xy[1]:.3f})"
                    )
                else:
                    logger.info(f"✅ 检测成功: hand={hand}")
            else:
                logger.error(f"❌ 检测失败: {response.get('error')}")

            return response

    async def ping(self) -> bool:
        """测试服务器连接"""
        try:
            async with websockets.asyncio.client.connect(
                self.server_url,
                compression=None,
                max_size=None
            ) as websocket:
                await websocket.send(msgpack.packb({"type": "ping"}))
                response = msgpack.unpackb(await websocket.recv())
                return response.get("type") == "pong"
        except Exception as e:
            logger.error(f"Ping失败: {e}")
            return False


class DetectionClient3DSync:
    """同步封装版"""

    def __init__(self, server_url: str = "ws://localhost:8000", timeout: float = 30.0):
        self.async_client = DetectionClient3D(server_url, timeout)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def detect_hand(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        torso_height: float,
        tilt_angle: float,
        pan_angle: float,
        object_name: str,
        table_prompt: Optional[str] = None,
        save_debug: bool = False,
    ) -> dict:
        """同步接口"""
        return self.loop.run_until_complete(
            self.async_client.detect_hand(
                color_img=color_img,
                depth_img=depth_img,
                torso_height=torso_height,
                tilt_angle=tilt_angle,
                pan_angle=pan_angle,
                object_name=object_name,
                table_prompt=table_prompt,
                save_debug=save_debug,
            )
        )

    def ping(self) -> bool:
        """测试连接"""
        return self.loop.run_until_complete(self.async_client.ping())

    def close(self):
        """关闭"""
        if not self.loop.is_closed():
            self.loop.close()


# ============ 集成到ROS/VLA的示例 ============

class VLADetector:
    """
    VLA检测集成类
    在你的 graspservice_state_machine.py 中使用
    """

    def __init__(self, server_url: str = "ws://192.168.1.215:8000"):
        self.client = DetectionClient3DSync(server_url)

    def detect_for_vla(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        torso_height: float,
        tilt_angle: float,
        pan_angle: float,
        object_name: str,
        save_debug: bool = False,
    ) -> Optional[str]:
        """
        为VLA检测hand

        返回: "left" 或 "right"，失败返回None
        """
        result = self.client.detect_hand(
            color_img=color_img,
            depth_img=depth_img,
            torso_height=torso_height,
            tilt_angle=tilt_angle,
            pan_angle=pan_angle,
            object_name=object_name,
            save_debug=save_debug,
        )

        if result.get("success"):
            hand = result.get("hand")
            xy = result.get("object_xy")
            if xy is not None:
                logger.info(
                    f"VLA检测: {object_name} at ({xy[0]:.2f}, {xy[1]:.2f}) -> {hand} hand"
                )
            else:
                logger.info(f"VLA检测: {object_name} -> {hand} hand")
            return hand
        else:
            logger.error(f"VLA检测失败: {result.get('error')}")
            return None

    def format_vla_prompt(self, action: str, object_name: str, location: str, hand: str) -> str:
        """格式化VLA提示词"""
        return f"Use your {hand} hand to {action} the {object_name} from the {location}"


# ============ 测试代码 ============

def test_with_simulated_data():
    """使用模拟数据测试"""
    color_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.putText(color_img, "Test", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    depth_img = np.ones((480, 640), dtype=np.uint16) * 1000  # 1米

    torso_height = 0.5
    tilt_angle = 0.0
    pan_angle = 0.0

    client = DetectionClient3DSync("ws://192.168.1.215:8000")

    try:
        if not client.ping():
            print("❌ 无法连接服务器")
            return

        print("✅ 连接成功")

        result = client.detect_hand(
            color_img=color_img,
            depth_img=depth_img,
            torso_height=torso_height,
            tilt_angle=tilt_angle,
            pan_angle=pan_angle,
            object_name="yellow duck toy",
            save_debug=True,
        )

        print(f"\n结果: {result}")

    finally:
        client.close()


if __name__ == "__main__":
    test_with_simulated_data()