import numpy as np
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt

class Camera(object):

    def __init__(self,width=1280,height=720,fps=15,serial_number = None,name = 'd455',calibrate_done = False):
        self.im_height = height
        self.im_width = width
        self.fps = fps
        self.intrinsics = None
        self.pipeline = None
        self.pose = None
        self.serial_number = serial_number
        self.name = name
        self.calibrate_done = calibrate_done
        # self.serial_number1 = "311322300927"  # d455
        self.connect()

        self.dec_filter = rs.decimation_filter()     # 降采样
        self.spatial_filter = rs.spatial_filter()    # 空间滤波
        self.temporal_filter = rs.temporal_filter()  # 时间滤波
        self.hole_filter = rs.hole_filling_filter()  # 填补洞

    def connect(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)
        config.enable_device(self.serial_number)
          
        # Start streaming
        cfg = self.pipeline.start(config)

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        self.sensor = cfg.get_device().first_depth_sensor()

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = self.get_intrinsics(rgb_profile)

        if self.name == 'd455':
            if self.calibrate_done:
                self.intrinsics = np.loadtxt('pose_data_change_1/camera_mtx.txt', delimiter=' ')
                self.pose = np.loadtxt('pose_data_change_1/camera_pose.txt', delimiter=' ')
            print("D455 have connected ...")
        elif self.name == 'left':
            if self.calibrate_done:
                # 第一台的左臂
                # self.intrinsics = np.loadtxt('data_h10_w_45_left_0812_640_480/camera_mtx.txt', delimiter=' ')
                # self.pose = np.loadtxt('data_h10_w_45_left_0812_640_480/camera_pose.txt', delimiter=' ')
                # 最新一台的左臂
                self.intrinsics = np.loadtxt('data_h10_w_45_0907_left/camera_mtx.txt', delimiter=' ')
                self.pose = np.loadtxt('data_h10_w_45_0907_left/camera_pose.txt', delimiter=' ')

            print("D405 left have connected ...")
        elif self.name == 'right':
            if self.calibrate_done:
                self.intrinsics = np.loadtxt('data_h10_w_45_right_0812_640_480/camera_mtx.txt', delimiter=' ')
                self.pose = np.loadtxt('data_h10_w_45_right_0812_640_480/camera_pose.txt', delimiter=' ')
            print("D405 right have connected ...")
    def get_data(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames(timeout_ms=10000)

        # # align
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_frame = self.spatial_filter.process(aligned_depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.hole_filter.process(depth_frame)
        depth_image = np.asanyarray(depth_frame.get_data())

        # # 将无效值(0或NaN)设为最大深度
        # depth_valid = np.where(depth_image == 0, np.nan, depth_image)

        # # 归一化到 0-255
        # depth_norm = cv2.normalize(depth_valid, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # # 转换成伪彩色图
        # depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # cv2.imshow("Depth Colormap", depth_color)
        # cv2.waitKey(0)
        
        # no align
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.float32)
        # depth_image *= self.scale
        depth_image = np.expand_dims(depth_image, axis=2)
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image
    
    def get_pointcloud(self,color_image,depth_image):
        # color_image, depth_image = self.get_data()
        depth_img = np.squeeze(depth_image)*self.scale
        intrinsics = self.intrinsics
        fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
        # depth_masked = np.where(mask > 0, depth_img, 0) * scale
        x_map, y_map = np.meshgrid(np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0]))
        x_map = (x_map - cx)*depth_img / fx
        y_map = (y_map - cy)*depth_img / fy
        points = np.stack((x_map, y_map, depth_img), axis=-1)
        # points = np.stack((x_map, y_map, depth_img), axis=-1).reshape(-1, 3)
        # points = points[depth_masked.flatten() > 0]
        return points

    def plot_image(self):
        color_image,depth_image = self.get_data()
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # cv2.imwrite('color_image.png', color_image)
        cv2.waitKey(5000)

    def get_intrinsics(self,rgb_profile):
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        print("camera intrinsics:", raw_intrinsics)
        # camera intrinsics form is as follows.
        #[[fx,0,ppx],
        # [0,fy,ppy],
        # [0,0,1]]
        # intrinsics = np.array([615.284,0,309.623,0,614.557,247.967,0,0,1]).reshape(3,3) #640 480
        intrinsics = np.array([raw_intrinsics.fx, 0, raw_intrinsics.ppx, 0, raw_intrinsics.fy, raw_intrinsics.ppy, 0, 0, 1]).reshape(3, 3)

        return intrinsics
if __name__== '__main__':
    serial_number1_d455 = "333422304763"  # d455
    serial_number1_d405 = "218722271112" # d405
    # camera_d455 = Camera(width=1280,height=720,fps=15,serial_number = serial_number1_d455,camera_name = 'd455',calibrate_done = True)
    # camera_d405 = Camera(width=1280, height=720, fps=15, serial_number=serial_number1_d405, name='d405',
    #                      calibrate_done=True)
    # # color_455,depth_455 = camera_d455.get_data()
    # color_405,depth_405 = camera_d405.get_data()
    # # print(camera_d455.intrinsics)
    # # print(camera_d455.pose)
    # # print(color_455.shape)

    # print(camera_d405.intrinsics)
    # print(camera_d405.pose)
    # print(camera_d405.scale)
    camera_d455 = Camera(width=640, height=480, fps=30, serial_number=serial_number1_d405, name='left',
                         calibrate_done=False)
    print(camera_d455.scale)
    color_455,depth_455 = camera_d455.get_data()
    plt.imshow(color_455[..., ::-1])  # BGR → RGB
    plt.show()

