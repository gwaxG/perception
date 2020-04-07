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

    def get_overlapped(self, P1, P2):
        P11 = P1[0]
        P12 = P1[1]
        P21 = P2[0]
        P22 = P2[1]
        # Is the point 3 inside of the rectangle formed by p1 and p2 (p1 is the upper left corner) ?
        is_inside = lambda p1, p2, p3: True if p3[0] < p2[0] and p3[0] > p1[0] and p3[1] < p2[1] and p3[1] > p1[1] else False
        # Calculate surface of the rectangle formed by two points p1 and p2
        surf = lambda p1, p2:  (p2[1] - p1[1]) * (p2[0] - p1[0])

        if is_inside(P11, P12, P21) and not is_inside(P11, P12, P22):
            return surf(P21, P12) / surf(P11, P12), surf(P21, P12) / surf(P21, P22)

        elif is_inside(P11, P12, P21) and is_inside(P11, P12, P22):
            return surf(P21, P22) / surf(P11, P12), 1.0

        elif is_inside(P21, P22, P11) and is_inside(P21, P22, P12):
            return 1.0, surf(P11, P12) / surf(P21, P22)

        elif not is_inside(P11, P12, P21) and is_inside(P11, P12, P22):
            return surf(P11, P21) / surf(P11, P12), surf(P11, P21) / surf(P21, P22)

        else:
            return 0., 0.

    def compare_bboxes(self):
        self.depth_bbox_updated = False
        self.rgb_bbox_updated = False

        #  take two points that consist of 4 points: P11, P12, P21, P22; and get ration of overlapping for every surface

        s1, s2 = self.get_overlapped(self.rgb_bbox, self.depth_bbox)
        if s1 > 0.5 and s2 > 0.5:
            self.bbox_image = self.draw_bbox(self.bbox_image, self.depth_bbox, 'depth')
            self.bbox_image = self.draw_bbox(self.bbox_image, self.rgb_bbox, 'rgb')
            self.publish_image(self.bbox_image)

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
