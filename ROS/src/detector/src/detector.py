#!/usr/bin/env python2
import sys, time, os
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from PIL import Image as PILimage
from image_geometry import PinholeCameraModel
import message_filters

DEBUG = 1
CALIBRATION_MODE = 0
CAL_SCREEN_INIT = 0
CAL_COLOR = 'g'
CAL_PRINT_SINGLE = 0


"""
Main class for object detection
"""
class block_detector:
    def __init__(self):
        rospy.init_node('block_detector')
        self.image_subscriber = message_filters.Subscriber("/uav/rs_d435/color/image_raw", Image) #color image topic
        self.cam_info_subscriber = message_filters.Subscriber("/uav/rs_d435/color/camera_info", CameraInfo) #color camera info
        self.depth_subscriber = message_filters.Subscriber("/uav/rs_d435/depth/image_rect_raw", Image) #depth image topic
        self.depth_info_subscriber = message_filters.Subscriber("/uav/rs_d435/depth/camera_info", CameraInfo) #depth camera info
        self.plane_check_subscriber = rospy.Subscriber("/plane_check_result", String, self.plane_check_callback)
        self.pl_ch_msg_list = []

        self.bbox_img_data_pub = rospy.Publisher('bbox_image', Image, queue_size=10) #FINAL IMAGE TOPIC
        self.bbox_cand_img_data_pub = rospy.Publisher('bbox_image_candidate', Image, queue_size=10) #all candidates image
        self.test_img_data_pub = rospy.Publisher('image_rect', Image, queue_size=10) #output topic for depth crop
        self.test_cam_info_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=10) #output topic for camera info

        self.msg_img = Image()
        self.msg_ci = CameraInfo()
        self.coeff_ratio = []
        self.first_message = True
        self.P_img = np.zeros((3,4))
        self.P_dep = np.zeros((3,4))
        self.K_img = np.eye(3)
        self.K_img_inv = np.eye(3)
        self.K_depth = np.eye(3)
        self.K_conv_depth = np.eye(3)
        self.K_conv = np.eye(3)

        self.enlarge_coeff = {'b':1.2, 'g':1.35, 'r':1.5} #1.4

        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_subscriber,  self.depth_subscriber, self.cam_info_subscriber, self.depth_info_subscriber], 5, 0.1)
        self.ts.registerCallback(self.callback)
        self.area_thresh = 400
        self.max_lines_contour = 15
        self.approx_coeff = 0.01 #0.02
        self.hsv_filters = {}
        self.frame_ctr = 0
        self.time_calc = []
        """
        self.hsv_filters['b']  = [[95, 115, 45], [105, 255, 255], [105, 95, 45], [130, 255, 255]]
        self.hsv_filters['g']  = [[70 , 95, 15], [85, 255, 255], [85 , 60, 30], [100, 238, 238]]
        self.hsv_filters['r']  = [[0, 35, 0], [10, 255, 255], [170, 40, 0], [175, 255, 255]] #130 180
        """
        self.hsv_filters['b']  = [[97, 140, 75], [108, 255, 255]]
        self.hsv_filters['g']  = [[78 , 65, 35], [97, 255, 255]]
        self.hsv_filters['r']  = [[0, 100, 80], [6, 255, 255], [170, 100, 80], [180, 255, 255]] #130 180
        self.cnt_colours = {'b' : (255,0,0), 'g' : (0,255,0), 'r' : (0,0,255)}
        self.color_area_min_limit = {'b':0.45, 'g':0.15, 'r':0.10}

    #Function calculates bbox area
    def calc_region_area(self, x1, x2, y1, y2):
        x_dist = abs(x2 - x1)
        y_dist = abs(y2 - y1)
        return x_dist * y_dist



    #Function returns enlarged bbox coordinates based on region color
    def enlarge_crop(self, x1, x2, y1, y2, col):

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        x_dist = (abs(x2 - x1) / 2.0) * self.enlarge_coeff[col]
        y_dist = (abs(y2 - y1) / 2.0) * self.enlarge_coeff[col]

        x1, x2, y1, y2 = x_center - x_dist, x_center + x_dist, y_center - y_dist, y_center + y_dist
        x1, x2, y1, y2 = 0 if x1 < 0 else x1, 0 if x2 < 0 else x2, 0 if y1 < 0 else y1,  0 if y2 < 0 else y2

        return x1, x2, y1, y2


    #Function to store geometrical verification results
    def plane_check_callback(self, msg):
        self.pl_ch_msg_list.append(msg.data)



    #Main detector routine
    def callback(self, image_data, depth_data, cam_info, depth_info):
        start_time = time.time()
        np_data = np.fromstring(image_data.data, np.uint8)
        in_image = cv2.cvtColor(np_data.reshape(image_data.height, image_data.width,3), cv2.COLOR_RGB2BGR)

        np_data = np.fromstring(depth_data.data, np.uint16)
        in_depth = np_data.reshape(depth_data.height, depth_data.width)
        in_depth[np.isnan(in_depth)] = 0

        draw_circ = False

        if (self.first_message):

            self.K_img[0,0] =  cam_info.K[0]
            self.K_img[0,2] =  cam_info.K[2]
            self.K_img[1,1] =  cam_info.K[4]
            self.K_img[1,2] =  cam_info.K[5]


            self.K_img_inv = np.linalg.inv(self.K_img)

            self.K_depth[0,0] =  depth_info.K[0]
            self.K_depth[0,2] =  depth_info.K[2]
            self.K_depth[1,1] =  depth_info.K[4]
            self.K_depth[1,2] =  depth_info.K[5]

            self.P_img[0:3, 0:3] = self.K_img
            self.P_dep[0:3, 0:3] = self.K_depth

            self.K_conv = self.K_depth.dot(self.K_img_inv)
            self.K_conv_depth = self.K_img.dot(np.linalg.inv(self.K_depth))

            self.msg_img.header = depth_data.header
            self.msg_img.encoding = depth_data.encoding
            self.msg_img.is_bigendian = depth_data.is_bigendian
            self.coeff_ratio = [float(in_depth.shape[1])/float(in_image.shape[1]), float(in_depth.shape[0])/float(in_image.shape[0])]

            self.msg_ci.header = depth_data.header
            self.msg_ci.distortion_model = depth_info.distortion_model
            self.msg_ci.D = depth_info.D
            self.msg_ci.K = depth_info.K
            self.msg_ci.R = depth_info.R
            self.msg_ci.P = depth_info.P

            self.msg_ci.binning_x = depth_info.binning_x
            self.msg_ci.binning_y = depth_info.binning_y

            self.first_message = False

            self.msg_img.height = depth_data.height
            self.msg_img.width = depth_data.width

            self.msg_ci.height = depth_info.height
            self.msg_ci.width = depth_info.width

        else:
            self.msg_img.header = depth_data.header
            self.msg_ci.header = depth_data.header

        #color segmentation
        hsv = cv2.cvtColor(in_image, cv2.COLOR_BGR2HSV)
        image_with_candidate = in_image.copy()
        image_with_cnt = in_image.copy()
        countors = {}
        bboxes = []

        for col in self.hsv_filters.keys():
            countors[col], boundRect = self.get_countors(hsv, col)
            if (not len(countors[col])) or (not len(boundRect)): continue

            for i in range(len(countors[col])):
                y1,y2 = int(boundRect[i][1]), int((boundRect[i][1]+boundRect[i][3]))
                x1, x2 = int(boundRect[i][0]), int((boundRect[i][0]+boundRect[i][2]))
                bboxes.append([cv2.contourArea(countors[col][i]), x1, x2, y1, y2, countors[col][i], col])

        #region voting
        bbox_voter = np.zeros((in_image.shape[0], in_image.shape[1]), dtype=np.uint8)
        bboxes.sort(reverse=True)
        for bbox in bboxes:
            area, x1_i, x2_i, y1_i, y2_i, cnt, col = bbox
            dummy = np.zeros((in_image.shape[0], in_image.shape[1]), dtype=np.uint8)
            approx = cv2.approxPolyDP(cnt, self.approx_coeff*cv2.arcLength(cnt, True), True)
            cv2.fillPoly(dummy, [approx], 1)
            a =  (np.logical_and(bbox_voter != 0, dummy != 0)).astype(np.uint8)
            intersection = np.sum(a) / float(area)


            if DEBUG:
                print("COLOR_IMG: ", col, intersection)
                cv2.rectangle(image_with_candidate, (bbox[1], bbox[3]), (bbox[2], bbox[4]), self.cnt_colours[col], 2)

            if intersection < 0.1:
                #geometrical verification
                if DEBUG: print(x1_i, x2_i, y1_i, y2_i)
                bbox_voter[dummy != 0] = 1


                x1, x2, y1, y2 = self.enlarge_crop(x1_i, x2_i, y1_i, y2_i, col)
                x1, y1, _ = (self.K_conv.dot(np.array([x1, y1, 1]))).astype(int)
                x2, y2, _ = (self.K_conv.dot(np.array([x2, y2, 1]))).astype(int)

                mean_dist = in_depth[y1:y2, x1:x2].mean()
                if DEBUG: print(mean_dist, area)
                if (mean_dist < 700): continue

                w_tmp = x2 - x1
                h_tmp = y2 - y1

                """
                if (w_tmp*h_tmp > 25000):
                    print("RESIZED!")
                    w_tmp = int(w_tmp/2)
                    h_tmp = int(h_tmp/2)
                    tmp = in_depth[y1:y2, x1:x2]
                    cv2.resize(tmp, dsize=(w_tmp, h_tmp), interpolation=cv2.INTER_CUBIC)
                    self.msg_img.data = tmp.tostring()

                else:
                """

                self.msg_img.data = in_depth[y1:y2, x1:x2].tostring()

                self.msg_img.height = h_tmp
                self.msg_img.width = w_tmp
                self.msg_img.step = w_tmp * 2

                self.msg_ci.height = h_tmp
                self.msg_ci.width = w_tmp
                self.msg_ci.roi.height = h_tmp
                self.msg_ci.roi.width = w_tmp

                self.test_img_data_pub.publish(self.msg_img)
                self.test_cam_info_pub.publish(self.msg_ci)

                #waiting for the response
                while (len(self.pl_ch_msg_list) == 0):
                    pass

                #processing of response and object center projection
                answer = self.pl_ch_msg_list.pop(0)
                if DEBUG: print("MEAN DIST: ", mean_dist)

                if len(answer) > 2:
                    zones = answer.split(",")[0]
                    zones = zones[:-1].split("x")
                    tmp_buffer = []
                    if len(zones) and len(zones[0]):
                        tmp_buffer = []
                        for z in zones:

                            tmp = z.split("#")
                            if DEBUG: print(tmp)
                            area = float(tmp[0])
                            if area < self.color_area_min_limit[col]: continue
                            tmp_buffer.append([area, [float(tmp[1]), float(tmp[2]), float(tmp[3])]])


                    tmp_buffer.sort(reverse=True)
                    for i, z in enumerate(tmp_buffer):
                        xn, yn, zn = z[1]
                        zn += 0.03

                        u1 = self.msg_ci.P[2] + (self.msg_ci.P[0]*xn+self.msg_ci.P[3])/float(zn)
                        v1 = self.msg_ci.P[6] + (self.msg_ci.P[5]*yn+self.msg_ci.P[7])/float(zn)
                        u1 += x1
                        v1 += y1
                        d_cent = np.array([u1, v1, 1]).astype(np.uint16)

                        u2, v2, _ = (self.K_conv_depth.dot(d_cent)).astype(np.uint16)


                        if (v2 < dummy.shape[0] and u2 < dummy.shape[1] and dummy[v2, u2] != 0):
                            draw_circ = True
                        else:
                            dummy_cent = np.zeros((in_image.shape[0], in_image.shape[1]), dtype=np.uint8)
                            cv2.circle(dummy_cent, (int(u2), int(v2)) , 2, 1, -1)
                            if np.sum((np.logical_and(dummy_cent != 0, dummy != 0)).astype(np.uint8)) != 0 :
                                draw_circ = True
                        if draw_circ:
                            cv2.circle(image_with_cnt, (int(u2), int(v2)) , 10, self.cnt_colours[col], 7)

                    cv2.rectangle(image_with_cnt, (bbox[1], bbox[3]), (bbox[2], bbox[4]), self.cnt_colours[col], 2)


        if DEBUG:
            tmp_bbox = Image()
            image_with_cnt = cv2.cvtColor(image_with_cnt, cv2.COLOR_BGR2RGB)
            tmp_bbox.data = image_with_cnt.tostring()
            tmp_bbox.height = image_with_cnt.shape[0]
            tmp_bbox.width = image_with_cnt.shape[1]
            tmp_bbox.header = image_data.header
            tmp_bbox.encoding = image_data.encoding
            tmp_bbox.is_bigendian = image_data.is_bigendian
            self.bbox_img_data_pub.publish(tmp_bbox)
            image_with_candidate = cv2.cvtColor(image_with_candidate, cv2.COLOR_BGR2RGB)
            tmp_bbox.data = image_with_candidate.tostring()
            self.bbox_cand_img_data_pub.publish(tmp_bbox)


        if draw_circ:
            self.time_calc.append((time.time() - start_time))
            print("--- %s seconds ---" % self.time_calc[-1])
            print("--- avg %s seconds ---" % np.mean(np.array([self.time_calc])))
        print("--------------%s--------------" % self.frame_ctr)
        self.frame_ctr += 1

    #Function returns contours of color regions
    def get_countors(self, image, color):
        global CAL_PRINT_SINGLE
        if (CAL_PRINT_SINGLE): print(color)

        if len(self.hsv_filters[color]) == 2:
            mask = cv2.inRange(image, np.array(self.hsv_filters[color][0]), np.array(self.hsv_filters[color][1]))


        elif len(self.hsv_filters[color]) == 4:
            mask1 = cv2.inRange(image, np.array(self.hsv_filters[color][0]), np.array(self.hsv_filters[color][1]))
            mask2 = cv2.inRange(image, np.array(self.hsv_filters[color][2]), np.array(self.hsv_filters[color][3]))
            mask = mask1 + mask2


        countors_toRet = []

        if int(cv2.__version__[0]) > 3:
            contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        boundRect = []

        for i, cnt in enumerate(contours):
            if (cv2.contourArea(cnt) > self.area_thresh):
                approx = cv2.approxPolyDP(cnt, self.approx_coeff*cv2.arcLength(cnt, True), True)
                if len(approx) > self.max_lines_contour: continue
                countors_toRet.append(cnt)
                boundRect.append(cv2.boundingRect(approx))

        return countors_toRet, boundRect



if __name__ == '__main__':

    bd = block_detector()
    print("Block detector is running!")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down")
