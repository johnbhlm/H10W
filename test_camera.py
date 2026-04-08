# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# import numpy as np
# import time
# import sys


# class CameraSyncMonitor(Node):

#     def __init__(self):

#         super().__init__('camera_sync_monitor')

#         self.head_stamp = None
#         self.left_stamp = None
#         self.right_stamp = None

#         self.head_delay = 0
#         self.left_delay = 0
#         self.right_delay = 0
        
#         # 存储历史数据用于统计
#         self.hl_history = []
#         self.hr_history = []
#         self.lr_history = []
#         self.delay_history = {
#             'head': [],
#             'left': [],
#             'right': []
#         }
        
#         self.start_time = time.time()
#         self.msg_count = 0
#         self.last_print_time = time.time()

#         self.create_subscription(Image, "/h10_w/head/color/image_raw", self.head_cb, 10)
#         self.create_subscription(Image, "/h10_w/left_wrist/color/image_rect_raw", self.left_cb, 10)
#         self.create_subscription(Image, "/h10_w/right_wrist/color/image_rect_raw", self.right_cb, 10)
        
#         # 启动定时器定期打印统计信息
#         self.create_timer(1.0, self.print_statistics)
        
#         print("相机同步监控器已启动，等待消息...")
#         print("-" * 80)

#     def stamp_to_sec(self, stamp):
#         return stamp.sec + stamp.nanosec * 1e-9

#     def now_sec(self):
#         now = self.get_clock().now().seconds_nanoseconds()
#         return now[0] + now[1] * 1e-9

#     def head_cb(self, msg):
#         self.head_stamp = self.stamp_to_sec(msg.header.stamp)
#         self.head_delay = self.now_sec() - self.head_stamp
#         self.delay_history['head'].append(self.head_delay * 1000)
#         if len(self.delay_history['head']) > 100:
#             self.delay_history['head'].pop(0)
#         self.update()

#     def left_cb(self, msg):
#         self.left_stamp = self.stamp_to_sec(msg.header.stamp)
#         self.left_delay = self.now_sec() - self.left_stamp
#         self.delay_history['left'].append(self.left_delay * 1000)
#         if len(self.delay_history['left']) > 100:
#             self.delay_history['left'].pop(0)
#         self.update()

#     def right_cb(self, msg):
#         self.right_stamp = self.stamp_to_sec(msg.header.stamp)
#         self.right_delay = self.now_sec() - self.right_stamp
#         self.delay_history['right'].append(self.right_delay * 1000)
#         if len(self.delay_history['right']) > 100:
#             self.delay_history['right'].pop(0)
#         self.update()

#     def update(self):
#         if self.head_stamp and self.left_stamp and self.right_stamp:
            
#             self.msg_count += 1
            
#             hl = abs(self.head_stamp - self.left_stamp) * 1000
#             hr = abs(self.head_stamp - self.right_stamp) * 1000
#             lr = abs(self.left_stamp - self.right_stamp) * 1000
            
#             # 保存历史数据
#             self.hl_history.append(hl)
#             self.hr_history.append(hr)
#             self.lr_history.append(lr)
            
#             # 限制历史数据长度
#             max_history = 100
#             if len(self.hl_history) > max_history:
#                 self.hl_history.pop(0)
#                 self.hr_history.pop(0)
#                 self.lr_history.pop(0)
            
#             # 实时显示当前值
#             current_time = time.time()
#             if current_time - self.last_print_time >= 0.5:  # 每0.5秒更新一次显示
#                 self.print_current_status(hl, hr, lr)
#                 self.last_print_time = current_time

#     def print_current_status(self, hl, hr, lr):
#         """打印当前状态"""
#         sys.stdout.write('\r' + ' ' * 100 + '\r')  # 清除当前行
#         sys.stdout.write(
#             f"HL:{hl:6.2f}ms | HR:{hr:6.2f}ms | LR:{lr:6.2f}ms | "
#             f"延迟(ms) 头:{self.head_delay*1000:6.1f} 左:{self.left_delay*1000:6.1f} 右:{self.right_delay*1000:6.1f} | "
#             f"消息数:{self.msg_count:4d}"
#         )
#         sys.stdout.flush()

#     def print_statistics(self):
#         """打印统计信息"""
#         if len(self.hl_history) > 0:
#             print("\n" + "=" * 80)
#             print(f"统计信息 (基于最近 {len(self.hl_history)} 个样本):")
#             print("-" * 80)
            
#             # 相机间同步延迟统计
#             print("相机间同步延迟(ms):")
#             print(f"  Head-Left : 平均={np.mean(self.hl_history):6.2f}  "
#                   f"最小={np.min(self.hl_history):6.2f}  "
#                   f"最大={np.max(self.hl_history):6.2f}  "
#                   f"标准差={np.std(self.hl_history):6.2f}")
#             print(f"  Head-Right: 平均={np.mean(self.hr_history):6.2f}  "
#                   f"最小={np.min(self.hr_history):6.2f}  "
#                   f"最大={np.max(self.hr_history):6.2f}  "
#                   f"标准差={np.std(self.hr_history):6.2f}")
#             print(f"  Left-Right: 平均={np.mean(self.lr_history):6.2f}  "
#                   f"最小={np.min(self.lr_history):6.2f}  "
#                   f"最大={np.max(self.lr_history):6.2f}  "
#                   f"标准差={np.std(self.lr_history):6.2f}")
            
#             # 传输延迟统计
#             print("\n传输延迟(ms):")
#             for cam in ['head', 'left', 'right']:
#                 if len(self.delay_history[cam]) > 0:
#                     print(f"  {cam.capitalize():5}: 平均={np.mean(self.delay_history[cam]):6.2f}  "
#                           f"最小={np.min(self.delay_history[cam]):6.2f}  "
#                           f"最大={np.max(self.delay_history[cam]):6.2f}  "
#                           f"标准差={np.std(self.delay_history[cam]):6.2f}")
            
#             # 总体评估
#             print("-" * 80)
#             avg_sync = (np.mean(self.hl_history) + np.mean(self.hr_history) + np.mean(self.lr_history)) / 3
#             max_delay = max(np.mean(self.delay_history['head']) if self.delay_history['head'] else 0,
#                           np.mean(self.delay_history['left']) if self.delay_history['left'] else 0,
#                           np.mean(self.delay_history['right']) if self.delay_history['right'] else 0)
            
#             if avg_sync < 10 and max_delay < 50:
#                 status = "✅ 良好"
#             elif avg_sync < 30 and max_delay < 100:
#                 status = "⚠️ 一般"
#             else:
#                 status = "❌ 较差"
            
#             print(f"系统状态: {status}")
#             print(f"运行时间: {time.time() - self.start_time:.1f} 秒")
#             print("=" * 80)


# def main():
#     rclpy.init()
#     node = CameraSyncMonitor()
    
#     print("\n" + "=" * 80)
#     print("相机同步监控器")
#     print("实时显示: HL=头-左延迟, HR=头-右延迟, LR=左-右延迟")
#     print("统计信息每秒更新一次")
#     print("=" * 80 + "\n")
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         print("\n\n用户中断")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
        
#         # 打印最终统计
#         print("\n最终统计:")
#         node.print_statistics()
#         print("\n程序退出")


# if __name__ == "__main__":
#     main()



# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# import numpy as np
# import time
# from typing import Optional, Dict, List
# from collections import deque


# class ThreeCameraSyncTester(Node):

#     def __init__(self):

#         super().__init__('three_camera_sync_tester')

#         # 存储三个相机的消息队列
#         self.inputs_dict = {
#             'head_rgb': deque(maxlen=200),
#             'left_rgb': deque(maxlen=200),
#             'right_rgb': deque(maxlen=200)
#         }
        
#         self.config = type('Config', (), {})()
#         self.config.msg_time_diff_threshold = 0.03  # 30ms阈值
        
#         # 统计信息
#         self.stats = {
#             'head_left_diff': [],    # 头-左时间差
#             'head_right_diff': [],   # 头-右时间差
#             'left_right_diff': [],   # 左-右时间差
#             'success_count': 0,
#             'fail_count': 0,
#             'fail_reasons': {
#                 'no_head_msg': 0,
#                 'no_left_msg': 0,
#                 'no_right_msg': 0,
#                 'left_threshold': 0,
#                 'right_threshold': 0,
#                 'all_threshold': 0
#             }
#         }
        
#         self.start_time = time.time()
#         self.msg_count = 0

#         # 订阅三个相机话题
#         self.create_subscription(Image, "/h10_w/head/color/image_raw", 
#                                 lambda msg: self.camera_cb(msg, 'head_rgb'), 10)
#         self.create_subscription(Image, "/h10_w/left_wrist/color/image_rect_raw", 
#                                 lambda msg: self.camera_cb(msg, 'left_rgb'), 10)
#         self.create_subscription(Image, "/h10_w/right_wrist/color/image_rect_raw", 
#                                 lambda msg: self.camera_cb(msg, 'right_rgb'), 10)
        
#         # 定时打印统计
#         self.create_timer(1.0, self.print_statistics)
        
#         print("=" * 80)
#         print("三相机同步测试器")
#         print(f"阈值: {self.config.msg_time_diff_threshold*1000:.1f}ms")
#         print("=" * 80)

#     def stamp_to_sec(self, stamp):
#         return stamp.sec + stamp.nanosec * 1e-9

#     def camera_cb(self, msg, topic_name):
#         """相机回调函数"""
#         msg_data = {
#             'message_time': self.stamp_to_sec(msg.header.stamp),
#             'recv_time': time.time(),
#             'msg': msg
#         }
#         self.inputs_dict[topic_name].append(msg_data)
#         self.msg_count += 1
        
#         # 每次收到头相机消息时检查三相机同步
#         if topic_name == 'head_rgb':
#             self.check_three_camera_sync()

#     def find_nearest_message(self, topic_name: str, timestamp: float) -> Optional[dict]:
#         """找到最接近的消息"""
#         min_diff = 100.0
#         nearest_msg = None
#         data_queue = list(self.inputs_dict[topic_name])
        
#         for msg in data_queue:
#             diff = abs(msg['message_time'] - timestamp)
#             if diff < min_diff:
#                 nearest_msg = msg
#                 min_diff = diff
#         return nearest_msg

#     def check_three_camera_sync(self):
#         """检查三相机同步状态"""
#         # 检查所有队列是否都有数据
#         for cam in ['head_rgb', 'left_rgb', 'right_rgb']:
#             if len(self.inputs_dict[cam]) == 0:
#                 self.stats['fail_reasons'][f'no_{cam.split("_")[0]}_msg'] += 1
#                 self.stats['fail_count'] += 1
#                 return
        
#         # 获取头相机最新消息
#         head_msg = self.inputs_dict['head_rgb'][-1]
#         head_time = head_msg['message_time']
        
#         # 为左右相机找最近消息
#         left_msg = self.find_nearest_message('left_rgb', head_time)
#         right_msg = self.find_nearest_message('right_rgb', head_time)
        
#         if left_msg is None or right_msg is None:
#             self.stats['fail_count'] += 1
#             return
        
#         # 计算三组时间差
#         head_left_diff = abs(left_msg['message_time'] - head_time) * 1000  # ms
#         head_right_diff = abs(right_msg['message_time'] - head_time) * 1000  # ms
#         left_right_diff = abs(left_msg['message_time'] - right_msg['message_time']) * 1000  # ms
        
#         # 记录统计
#         self.stats['head_left_diff'].append(head_left_diff)
#         self.stats['head_right_diff'].append(head_right_diff)
#         self.stats['left_right_diff'].append(left_right_diff)
        
#         # 限制历史数据长度
#         max_history = 1000
#         for key in ['head_left_diff', 'head_right_diff', 'left_right_diff']:
#             if len(self.stats[key]) > max_history:
#                 self.stats[key].pop(0)
        
#         # 检查阈值
#         threshold_ms = self.config.msg_time_diff_threshold * 1000
#         left_ok = head_left_diff <= threshold_ms
#         right_ok = head_right_diff <= threshold_ms
#         lr_ok = left_right_diff <= threshold_ms
        
#         if left_ok and right_ok and lr_ok:
#             self.stats['success_count'] += 1
#         else:
#             self.stats['fail_count'] += 1
#             if not left_ok:
#                 self.stats['fail_reasons']['left_threshold'] += 1
#             if not right_ok:
#                 self.stats['fail_reasons']['right_threshold'] += 1
#             if not lr_ok:
#                 self.stats['fail_reasons']['all_threshold'] += 1
        
#         # 实时显示
#         self.print_current_status(head_left_diff, head_right_diff, left_right_diff)

#     def print_current_status(self, hl, hr, lr):
#         """实时显示当前状态"""
#         threshold_ms = self.config.msg_time_diff_threshold * 1000
#         status_line = (f"HL:{hl:6.2f}ms {'✓' if hl<=threshold_ms else '✗'} | "
#                       f"HR:{hr:6.2f}ms {'✓' if hr<=threshold_ms else '✗'} | "
#                       f"LR:{lr:6.2f}ms {'✓' if lr<=threshold_ms else '✗'} | "
#                       f"成功/失败:{self.stats['success_count']}/{self.stats['fail_count']}")
        
#         # 使用\r实现单行刷新
#         print(f"\r{status_line}", end='', flush=True)

#     def print_statistics(self):
#         """打印详细统计信息"""
#         if len(self.stats['head_left_diff']) == 0:
#             return
            
#         print("\n" + "=" * 80)
#         print(f"三相机同步统计 (基于最近 {len(self.stats['head_left_diff'])} 个样本)")
#         print("-" * 80)
        
#         # 头-左相机
#         hl_diffs = self.stats['head_left_diff']
#         print("头-左相机同步延迟(ms):")
#         print(f"  平均值: {np.mean(hl_diffs):8.2f}ms")
#         print(f"  最小值: {np.min(hl_diffs):8.2f}ms")
#         print(f"  最大值: {np.max(hl_diffs):8.2f}ms")
#         print(f"  标准差: {np.std(hl_diffs):8.2f}ms")
        
#         # 头-右相机
#         hr_diffs = self.stats['head_right_diff']
#         print("\n头-右相机同步延迟(ms):")
#         print(f"  平均值: {np.mean(hr_diffs):8.2f}ms")
#         print(f"  最小值: {np.min(hr_diffs):8.2f}ms")
#         print(f"  最大值: {np.max(hr_diffs):8.2f}ms")
#         print(f"  标准差: {np.std(hr_diffs):8.2f}ms")
        
#         # 左-右相机
#         lr_diffs = self.stats['left_right_diff']
#         print("\n左-右相机同步延迟(ms):")
#         print(f"  平均值: {np.mean(lr_diffs):8.2f}ms")
#         print(f"  最小值: {np.min(lr_diffs):8.2f}ms")
#         print(f"  最大值: {np.max(lr_diffs):8.2f}ms")
#         print(f"  标准差: {np.std(lr_diffs):8.2f}ms")
        
#         # 阈值达标率
#         threshold_ms = self.config.msg_time_diff_threshold * 1000
#         hl_success = sum(1 for d in hl_diffs if d <= threshold_ms)
#         hr_success = sum(1 for d in hr_diffs if d <= threshold_ms)
#         lr_success = sum(1 for d in lr_diffs if d <= threshold_ms)
#         total_samples = len(hl_diffs)
        
#         print("\n阈值达标率:")
#         print(f"  头-左相机: {hl_success/total_samples*100:5.1f}% ({hl_success}/{total_samples})")
#         print(f"  头-右相机: {hr_success/total_samples*100:5.1f}% ({hr_success}/{total_samples})")
#         print(f"  左-右相机: {lr_success/total_samples*100:5.1f}% ({lr_success}/{total_samples})")
        
#         # 三相机完全同步率（三组都在阈值内）
#         perfect_sync = sum(1 for i in range(total_samples) 
#                           if hl_diffs[i] <= threshold_ms 
#                           and hr_diffs[i] <= threshold_ms 
#                           and lr_diffs[i] <= threshold_ms)
#         print(f"\n三相机完全同步率: {perfect_sync/total_samples*100:5.1f}% ({perfect_sync}/{total_samples})")
        
#         # 失败原因
#         print("\n失败原因统计:")
#         total_fails = sum(self.stats['fail_reasons'].values())
#         if total_fails > 0:
#             for reason, count in self.stats['fail_reasons'].items():
#                 if count > 0:
#                     pct = count/total_fails*100
#                     print(f"  {reason}: {count} ({pct:.1f}%)")
        
#         # 整体评估
#         print("-" * 80)
#         avg_sync = (np.mean(hl_diffs) + np.mean(hr_diffs) + np.mean(lr_diffs)) / 3
#         perfect_rate = perfect_sync/total_samples*100
        
#         if avg_sync < 10 and perfect_rate > 90:
#             status = "✅ 优秀"
#         elif avg_sync < 30 and perfect_rate > 70:
#             status = "⚠️ 一般"
#         else:
#             status = "❌ 较差"
        
#         print(f"系统状态: {status}")
#         print(f"运行时间: {time.time() - self.start_time:.1f}秒")
#         print("=" * 80)


# def main():
#     rclpy.init()
#     node = ThreeCameraSyncTester()
    
#     print("\n三相机同步测试开始...")
#     print("实时显示: HL=头-左, HR=头-右, LR=左-右")
#     print("详细统计每秒更新一次")
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         print("\n\n用户中断")
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
        
#         # 打印最终统计
#         print("\n最终统计:")
#         node.print_statistics()
#         print("\n测试结束")


# if __name__ == "__main__":
#     main()



import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import time
from typing import Optional, Dict, List
from collections import deque


class ThreeCameraSyncTester(Node):

    def __init__(self):

        super().__init__('three_camera_sync_tester')

        # 存储三个相机的消息队列（保留用于其他功能）
        self.inputs_dict = {
            'head_rgb': deque(maxlen=200),
            'left_rgb': deque(maxlen=200),
            'right_rgb': deque(maxlen=200)
        }
        
        self.config = type('Config', (), {})()
        self.config.msg_time_diff_threshold = 0.03  # 30ms阈值

        # frame interval统计
        self.last_stamp = {
            'head': None,
            'left': None,
            'right': None
        }

        self.frame_intervals = {
            'head': deque(maxlen=200),
            'left': deque(maxlen=200),
            'right': deque(maxlen=200)
        }

        # frame phase统计
        self.frame_phases = {
            'head': deque(maxlen=200),
            'left': deque(maxlen=200),
            'right': deque(maxlen=200)
        }

        # 相机周期（15Hz）
        self.frame_period = 1.0 / 15.0
        
        # 统计信息
        self.stats = {
            'head_left_diff': [],    # 头-左时间差
            'head_right_diff': [],   # 头-右时间差
            'left_right_diff': [],   # 左-右时间差
            'three_camera_diff': [], # 三相机最大时间差（新增）
            'success_count': 0,
            'fail_count': 0,
            'fail_reasons': {
                'no_head_msg': 0,
                'no_left_msg': 0,
                'no_right_msg': 0,
                'left_threshold': 0,
                'right_threshold': 0,
                'all_threshold': 0, 
                'three_camera_threshold': 0,  # 新增三相机阈值失败原因
                'sync_timeout': 0
            }
        }
        
        self.start_time = time.time()
        self.msg_count = 0
        self.sync_count = 0

        # 创建三个订阅者（用于message_filters）
        self.sub_head = Subscriber(self, Image, "/h10_w/head/color/image_raw")
        self.sub_left = Subscriber(self, Image, "/h10_w/left_wrist/color/image_rect_raw")
        self.sub_right = Subscriber(self, Image, "/h10_w/right_wrist/color/image_rect_raw")
        
        # 创建近似时间同步器
        # slop=0.05 表示50ms的时间容差，队列大小设为10
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_head, self.sub_left, self.sub_right],
            queue_size=50,
            slop=0.1  # 50ms的时间容差
        )
        self.sync.registerCallback(self.sync_callback)
        
        # 仍然保留原始订阅用于消息计数（可选）
        # 注意：如果同时使用message_filters和直接订阅，消息会被处理两次
        # 所以这里注释掉直接订阅，或者可以保留但需要注意
        """
        self.create_subscription(Image, "/h10_w/head/color/image_raw", 
                                lambda msg: self.camera_cb(msg, 'head_rgb'), 10)
        self.create_subscription(Image, "/h10_w/left_wrist/color/image_rect_raw", 
                                lambda msg: self.camera_cb(msg, 'left_rgb'), 10)
        self.create_subscription(Image, "/h10_w/right_wrist/color/image_rect_raw", 
                                lambda msg: self.camera_cb(msg, 'right_rgb'), 10)
        """
        
        # 为了保留消息计数功能，我们可以在回调中同时更新队列
        self.head_sub_raw = self.create_subscription(
            Image, "/h10_w/head/color/image_raw", 
            lambda msg: self.camera_cb_raw(msg, 'head_rgb'), 10
        )
        self.left_sub_raw = self.create_subscription(
            Image, "/h10_w/left_wrist/color/image_rect_raw", 
            lambda msg: self.camera_cb_raw(msg, 'left_rgb'), 10
        )
        self.right_sub_raw = self.create_subscription(
            Image, "/h10_w/right_wrist/color/image_rect_raw", 
            lambda msg: self.camera_cb_raw(msg, 'right_rgb'), 10
        )
        
        # 定时打印统计
        self.create_timer(1.0, self.print_statistics)
        
        print("=" * 80)
        print("三相机同步测试器 (使用ApproximateTimeSynchronizer)")
        print(f"阈值: {self.config.msg_time_diff_threshold*1000:.1f}ms")
        print(f"同步容差: 50ms")
        print("=" * 80)

    def stamp_to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def update_frame_timing(self, cam_name, stamp):
    
        # interval
        last = self.last_stamp[cam_name]
        
        if last is not None:
            interval = stamp - last
            self.frame_intervals[cam_name].append(interval * 1000)

        self.last_stamp[cam_name] = stamp

        # phase
        phase = (stamp % self.frame_period) * 1000
        self.frame_phases[cam_name].append(phase)

    def camera_cb_raw(self, msg, topic_name):
        """原始相机回调函数（仅用于消息计数和队列维护）"""
        msg_data = {
            'message_time': self.stamp_to_sec(msg.header.stamp),
            'recv_time': time.time(),
            'msg': msg
        }
        self.inputs_dict[topic_name].append(msg_data)
        self.msg_count += 1

    def sync_callback(self, head_msg, left_msg, right_msg):
        """ApproximateTimeSynchronizer的回调函数"""
        self.sync_count += 1
        
        # 提取时间戳（转换为秒）
        head_time = self.stamp_to_sec(head_msg.header.stamp)
        left_time = self.stamp_to_sec(left_msg.header.stamp)
        right_time = self.stamp_to_sec(right_msg.header.stamp)


        self.update_frame_timing('head', head_time)
        self.update_frame_timing('left', left_time)
        self.update_frame_timing('right', right_time)
        
        # 计算三组时间差（毫秒）
        head_left_diff = abs(head_time - left_time) * 1000
        head_right_diff = abs(head_time - right_time) * 1000
        left_right_diff = abs(left_time - right_time) * 1000
        # 计算三相机最大时间差（新增）
        three_camera_diff = max(head_left_diff, head_right_diff, left_right_diff)
        
        # 记录统计
        self.stats['head_left_diff'].append(head_left_diff)
        self.stats['head_right_diff'].append(head_right_diff)
        self.stats['left_right_diff'].append(left_right_diff)
        self.stats['three_camera_diff'].append(three_camera_diff)  # 新增
        
        # 限制历史数据长度
        max_history = 1000
        for key in ['head_left_diff', 'head_right_diff', 'left_right_diff']:
            if len(self.stats[key]) > max_history:
                self.stats[key].pop(0)
        
        # 检查阈值
        threshold_ms = self.config.msg_time_diff_threshold * 1000
        left_ok = head_left_diff <= threshold_ms
        right_ok = head_right_diff <= threshold_ms
        lr_ok = left_right_diff <= threshold_ms
        three_ok = three_camera_diff <= threshold_ms  # 新增三相机检查
        
        if left_ok and right_ok and lr_ok and three_ok:
            self.stats['success_count'] += 1
        else:
            self.stats['fail_count'] += 1
            if not left_ok:
                self.stats['fail_reasons']['left_threshold'] += 1
            if not right_ok:
                self.stats['fail_reasons']['right_threshold'] += 1
            if not lr_ok:
                self.stats['fail_reasons']['all_threshold'] += 1
            if not three_ok:  # 新增三相机失败计数
                self.stats['fail_reasons']['three_camera_threshold'] += 1
        
        # 实时显示
        self.print_current_status(head_left_diff, head_right_diff, left_right_diff,three_camera_diff)

    # def print_current_status(self, hl, hr, lr, three_diff):
    #     """实时显示当前状态（增加三相机差异参数）"""
    #     threshold_ms = self.config.msg_time_diff_threshold * 1000
    #     status_line = (f"同步:{self.sync_count:4d} | "
    #                   f"HL:{hl:6.2f}ms {'✓' if hl<=threshold_ms else '✗'} | "
    #                   f"HR:{hr:6.2f}ms {'✓' if hr<=threshold_ms else '✗'} | "
    #                   f"LR:{lr:6.2f}ms {'✓' if lr<=threshold_ms else '✗'} | "
    #                   f"3Cam:{three_diff:6.2f}ms {'✓' if three_diff<=threshold_ms else '✗'} | "
    #                   f"成功/失败:{self.stats['success_count']}/{self.stats['fail_count']}")
    #     # 使用\r实现单行刷新
    #     print(f"\r{status_line}", end='', flush=True)
        
    def print_current_status(self, hl, hr, lr, three_diff):

        threshold_ms = self.config.msg_time_diff_threshold * 1000

        phase_head = self.frame_phases['head'][-1] if self.frame_phases['head'] else 0
        phase_left = self.frame_phases['left'][-1] if self.frame_phases['left'] else 0
        phase_right = self.frame_phases['right'][-1] if self.frame_phases['right'] else 0

        status_line = (
            f"sync:{self.sync_count:4d} | "
            f"HL:{hl:6.2f} | HR:{hr:6.2f} | LR:{lr:6.2f} | "
            f"3Cam:{three_diff:6.2f} | "
            f"phase H:{phase_head:5.1f} L:{phase_left:5.1f} R:{phase_right:5.1f}"
        )

        print(f"\r{status_line}", end='', flush=True)
        
        

    def print_statistics(self):
        """打印详细统计信息"""
        if len(self.stats['head_left_diff']) == 0:
            return
            
        print("\n" + "=" * 80)
        print(f"三相机同步统计 (基于最近 {len(self.stats['head_left_diff'])} 个同步样本)")
        print(f"总同步次数: {self.sync_count}")
        print(f"总消息数: {self.msg_count}")
        print("-" * 80)
        
        # 头-左相机
        hl_diffs = self.stats['head_left_diff']
        print("头-左相机同步延迟(ms):")
        print(f"  平均值: {np.mean(hl_diffs):8.2f}ms")
        print(f"  最小值: {np.min(hl_diffs):8.2f}ms")
        print(f"  最大值: {np.max(hl_diffs):8.2f}ms")
        print(f"  标准差: {np.std(hl_diffs):8.2f}ms")
        
        # 头-右相机
        hr_diffs = self.stats['head_right_diff']
        print("\n头-右相机同步延迟(ms):")
        print(f"  平均值: {np.mean(hr_diffs):8.2f}ms")
        print(f"  最小值: {np.min(hr_diffs):8.2f}ms")
        print(f"  最大值: {np.max(hr_diffs):8.2f}ms")
        print(f"  标准差: {np.std(hr_diffs):8.2f}ms")
        
        # 左-右相机
        lr_diffs = self.stats['left_right_diff']
        print("\n左-右相机同步延迟(ms):")
        print(f"  平均值: {np.mean(lr_diffs):8.2f}ms")
        print(f"  最小值: {np.min(lr_diffs):8.2f}ms")
        print(f"  最大值: {np.max(lr_diffs):8.2f}ms")
        print(f"  标准差: {np.std(lr_diffs):8.2f}ms")

                # 三相机最大时间差（新增详细统计）
        three_diffs = self.stats['three_camera_diff']
        print("\n三相机最大时间差(ms) [整体同步指标]:")
        print(f"  平均值: {np.mean(three_diffs):8.2f}ms")
        print(f"  最小值: {np.min(three_diffs):8.2f}ms")
        print(f"  最大值: {np.max(three_diffs):8.2f}ms")
        print(f"  标准差: {np.std(three_diffs):8.2f}ms")
        print(f"  达标率: {sum(1 for d in three_diffs if d<=self.config.msg_time_diff_threshold*1000)/len(three_diffs)*100:5.1f}%")
        
        # 阈值达标率
        threshold_ms = self.config.msg_time_diff_threshold * 1000
        hl_success = sum(1 for d in hl_diffs if d <= threshold_ms)
        hr_success = sum(1 for d in hr_diffs if d <= threshold_ms)
        lr_success = sum(1 for d in lr_diffs if d <= threshold_ms)
        three_success = sum(1 for d in three_diffs if d <= threshold_ms)  # 新增
        total_samples = len(hl_diffs)
        
        print("\n阈值达标率:")
        print(f"  头-左相机: {hl_success/total_samples*100:5.1f}% ({hl_success}/{total_samples})")
        print(f"  头-右相机: {hr_success/total_samples*100:5.1f}% ({hr_success}/{total_samples})")
        print(f"  左-右相机: {lr_success/total_samples*100:5.1f}% ({lr_success}/{total_samples})")
        print(f"  三相机整体: {three_success/total_samples*100:6.2f}% ({three_success:3d}/{total_samples})")  # 新增
        
        # 三相机完全同步率（三组都在阈值内）
        perfect_sync = sum(1 for i in range(total_samples) 
                          if hl_diffs[i] <= threshold_ms 
                          and hr_diffs[i] <= threshold_ms 
                          and lr_diffs[i] <= threshold_ms)
        print(f"\n三相机完全同步率: {perfect_sync/total_samples*100:5.1f}% ({perfect_sync}/{total_samples})")
        
        # 失败原因
        print("\n失败原因统计:")
        total_fails = sum(self.stats['fail_reasons'].values())
        if total_fails > 0:
            for reason, count in self.stats['fail_reasons'].items():
                if count > 0:
                    pct = count/total_fails*100
                    print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # 整体评估
        print("-" * 80)
        avg_sync = (np.mean(hl_diffs) + np.mean(hr_diffs) + np.mean(lr_diffs)) / 3
        perfect_rate = perfect_sync/total_samples*100
        
        if avg_sync < 10 and perfect_rate > 90:
            status = "✅ 优秀"
        elif avg_sync < 30 and perfect_rate > 70:
            status = "⚠️ 一般"
        else:
            status = "❌ 较差"
        
        print(f"系统状态: {status}")
        print(f"运行时间: {time.time() - self.start_time:.1f}秒")
        print("=" * 80)


        print("\nFrame interval (ms):")

        for cam in ['head','left','right']:

            data = list(self.frame_intervals[cam])

            if len(data) > 0:
                print(
                    f"{cam}: avg={np.mean(data):6.2f} "
                    f"min={np.min(data):6.2f} "
                    f"max={np.max(data):6.2f}"
                )


def main():
    rclpy.init()
    node = ThreeCameraSyncTester()
    
    print("\n三相机同步测试开始...")
    print("实时显示: HL=头-左, HR=头-右, LR=左-右")
    print("详细统计每秒更新一次")
    print("使用ApproximateTimeSynchronizer进行精确同步")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
        # 打印最终统计
        print("\n最终统计:")
        node.print_statistics()
        print("\n测试结束")


if __name__ == "__main__":
    main()