#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, sys
import cv2, pickle
import numpy as np
import message_filters, socket
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from copy import deepcopy as dp
# import yolo

# from detection_cls import Detector

class RGBDetection:
    def __init__(self):
        '''
        Initialization of synchronized subscription to topics.
        '''
        rospy.init_node('perception')
        self.TCP_IP = 'localhost'
        self.TCP_PORT = 5002
        self.encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

        # These 3 variables are necessary to compare bboxes and visualiza these results
        self.depth_bbox_updated = False
        self.rgb_bbox_updated = False
        self.bbox_image = None
        self.rgb_bbox = None
        self.depth_bbox = None

        self.header = None
        # rgb_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
        rospy.Subscriber('/usb_cam/image_raw', Image, self.callback_rgb, queue_size=1)
        rospy.Subscriber('/stair/bbox/depth', Float32MultiArray, self.update_depth_bbox, queue_size=1)
        # depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        # ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub], 10)
        # ts.registerCallback(self.callback)
        self.bridge = CvBridge()
        # self.detector = Detector()
        self.image_pub = rospy.Publisher('/bbox_rgb', Image)
        rospy.spin()

    def update_depth_bbox(self, bbox_msg):
        self.depth_bbox = bbox_msg.data
        self.bbox_image = self.draw_bbox(self.bbox_image, self.depth_bbox, 'depth')
        self.publish_image(self.bbox_image)
        self.depth_bbox_updated = True

    def draw_bbox(self, img, bbox, type_):
        color2 = (255, 0, 0)
        color1 = (0, 255, 0)
        color = (0, 0, 255)
        bbox = [int(b) for b in bbox]
        cv2.circle(img=img, center=(bbox[0], bbox[1]), radius=10, thickness=5, color=color1)
        cv2.circle(img=img, center=(bbox[2], bbox[3]), radius=10, thickness=5, color=color2)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color)
        return img

    def compare_bboxes(self):
        self.depth_bbox_updated = False
        self.rgb_bbox_updated = False
        overlapped = False
        # if overlapped:
        #     self.bbox_image = self.draw_bbox(img, self.depth_bbox, 'depth')
        #     self.bbox_image = self.draw_bbox(img, self.rgb_bbox, 'rgb')
        #     self.publish_image(img)

    def publish_image(self, img):
        try:
            ROS_img = self.bridge.cv2_to_imgmsg(img, "bgr8")
            ROS_img.header.frame_id = 'camera_rgb_optical_frame'
            # ROS_img.header = self.header
        except CvBridgeError as e:
            print(e)
            return None
        self.image_pub.publish(ROS_img)
            # except CvBridgeError as e:
            # print(e)

    def callback_rgb(self, rgb): #, depth):
        '''
        This callback transforms ROS image into OpenCV image, call a processing method and publishes
        a ROS image.
        working
        :param rgb: ROS rgb image
        :param depth: ROS depth image
        '''
        print('RGB callback')
        try:
            cv_rgb_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
            # cv_depth_image = self.bridge.imgmsg_to_cv2(depth, "32FC1")
        except CvBridgeError as e:
            print(e)
            return None
        self.bbox_image = dp(cv_rgb_image)
        self.header = rgb.header
        # Sending received image to the yolo network
        s = socket.socket()
        s.connect((self.TCP_IP, self.TCP_PORT))
        result, imgencode = cv2.imencode('.jpg', cv_rgb_image, self.encode_param)
        data = np.array(imgencode)
        stringData = data.tostring()
        s.send(str(len(stringData)).ljust(16))
        s.send(stringData)

        back_data = s.recv(4096)
        back_data = pickle.loads(back_data)
        print('back data', back_data)
        # bboxes = self.detector.detect(cv_rgb_image) # not working  because of python needed versions
        s.close()

        # Draw on image
        # drawn_img = cv_rgb_image

        if len(back_data) != 0:
            self.rgb_bbox_updated = True
            self.rgb_bbox = back_data

        if self.depth_bbox_updated and self.rgb_bbox_updated:
            self.compare_bboxes()




def test_rgb():
    TCP_IP = 'localhost'
    TCP_PORT = 5002
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    img = cv2.imread('test.png')
    s = socket.socket()
    s.connect((TCP_IP, TCP_PORT))
    result, imgencode = cv2.imencode('.jpg', img, encode_param)
    data = np.array(imgencode)
    stringData = data.tostring()
    s.send(str(len(stringData)).ljust(16));
    s.send(stringData);
    data = s.recv(4096)
    print('From server', pickle.loads(data))
    s.close()

if __name__ == '__main__':
    # test_rgb()
    RGBDetection()
