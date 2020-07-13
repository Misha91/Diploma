#!/usr/bin/env python2
import sys, time, os
import numpy as np
import cv2
# Ros
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from PIL import Image as PILimage
from image_geometry import PinholeCameraModel
import message_filters
# Pytorch
#import torch



DEBUG = 1
CALIBRATION_MODE = 0
CAL_SCREEN_INIT = 0
CAL_COLOR = 'g'
CAL_PRINT_SINGLE = 0


class block_detector:
    def __init__(self):
        rospy.init_node('block_detector')
        self.image_subscriber = message_filters.Subscriber("/uav/rs_d435/color/image_raw", Image)
        self.cam_info_subscriber = message_filters.Subscriber("/uav/rs_d435/color/camera_info", CameraInfo)
        self.depth_subscriber = message_filters.Subscriber("/uav/rs_d435/depth/image_rect_raw", Image)
        self.depth_info_subscriber = message_filters.Subscriber("/uav/rs_d435/depth/camera_info", CameraInfo)
        self.plane_check_subscriber = rospy.Subscriber("/plane_check_result", String, self.plane_check_callback)
        self.pl_ch_msg_list = []

        self.bbox_img_data_pub = rospy.Publisher('bbox_image', Image, queue_size=10)
        self.test_img_data_pub = rospy.Publisher('image_rect', Image, queue_size=10)
        self.test_cam_info_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=10)
        self.msg_img = Image()
        self.msg_ci = CameraInfo()
        self.coeff_ratio = []
        self.first_message = True
        self.K_img_inv = np.eye(3)
        self.K_depth = np.eye(3)
        self.K_conv = np.eye(3)
        # time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_subscriber,  self.depth_subscriber, self.cam_info_subscriber, self.depth_info_subscriber], 5, 0.5)
        self.ts.registerCallback(self.callback)
        self.area_thresh = 400
        self.max_lines_contour = 15
        self.approx_coeff = 0.01 #0.02
        self.hsv_filters = {}
        self.hsv_filters['b']  = [[95, 115, 45], [105, 255, 255], [105, 95, 45], [130, 255, 255]]
        self.hsv_filters['g']  = [[70 , 95, 15], [85, 255, 255], [85 , 60, 30], [100, 238, 238]]
        self.hsv_filters['r']  = [[0, 35, 0], [10, 255, 255], [170, 40, 0], [175, 255, 255]] #130 180
        self.cnt_colours = {'b' : (255,0,0), 'g' : (0,255,0), 'r' : (0,0,255)}


    def plane_check_callback(self, msg):
        print(msg)
        self.pl_ch_msg_list.append(msg.data)

    def callback(self, image_data, depth_data, cam_info, depth_info):
        start_time = time.time()
        np_data = np.fromstring(image_data.data, np.uint8)

        in_image = cv2.cvtColor(np_data.reshape(image_data.height, image_data.width,3), cv2.COLOR_RGB2BGR)

        np_data = np.fromstring(depth_data.data, np.uint16)
        in_depth = np_data.reshape(depth_data.height, depth_data.width)
        in_depth[np.isnan(in_depth)] = 0
        in_depth = in_depth/10
        print(os.getcwd())
        cv2.imwrite("color.jpg", in_image)
        cv2.imwrite("depth.jpg", in_depth)
        # convert depth to PIL image and resize
        #depth_image = PILimage.fromarray(in_depth)
        #depth_image = depth_image.resize((in_image.shape[1], in_image.shape[0]))
        # convert depth to numpy back
        #in_depth = np.array(depth_image)

        if (self.first_message):

            self.K_img_inv[0,0] =  cam_info.K[0]
            self.K_img_inv[0,2] =  cam_info.K[2]
            self.K_img_inv[1,1] =  cam_info.K[4]
            self.K_img_inv[1,2] =  cam_info.K[5]

            self.K_img_inv = np.linalg.inv(self.K_img_inv)

            self.K_depth[0,0] =  depth_info.K[0]
            self.K_depth[0,2] =  depth_info.K[2]
            self.K_depth[1,1] =  depth_info.K[4]
            self.K_depth[1,2] =  depth_info.K[5]
            self.K_conv = self.K_depth.dot(self.K_img_inv)

            print(self.K_img_inv, self.K_depth, self.K_conv)



            self.msg_img.header = depth_data.header
            self.msg_img.encoding = depth_data.encoding
            self.msg_img.is_bigendian = depth_data.is_bigendian
            #self.coeff_ratio = [float(in_depth.shape[1])/float(in_image.shape[1]), float(in_depth.shape[0])/float(in_image.shape[0])]

            self.msg_ci.header = depth_info.header
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
            self.msg_ci.header = depth_info.header
        #print(np.max(depth))

        #print(np.linalg.inv(np.array([cam_info.P]).reshape(4,3)))
        hsv = cv2.cvtColor(in_image, cv2.COLOR_BGR2HSV)
        image_with_cnt = in_image.copy()
        countors = {}

        for col in self.hsv_filters.keys():
            countors[col], boundRect = self.get_countors(hsv, col)
            if (not len(countors[col])) or (not len(boundRect)): continue
            #print(countors[col], contours_poly, boundRect)
            for i in range(len(countors[col])):
                y1,y2 = int(boundRect[i][1]), int((boundRect[i][1]+boundRect[i][3]))
                x1, x2 = int(boundRect[i][0]), int((boundRect[i][0]+boundRect[i][2]))

                print("COLOR_IMG: ", x1, x2, y1, y2)
                x1, y1, _ = (self.K_conv.dot(np.array([x1, y1, 1]))).astype(int)
                x2, y2, _ = (self.K_conv.dot(np.array([x2, y2, 1]))).astype(int)
                #self.msg_img.data = np.zeros(in_depth.shape[0:2]).astype(np.uint16)
                #self.msg_img.data[y1:y2, x1:x2] = in_depth[y1:y2, x1:x2]
                self.msg_img.data = in_depth[y1:y2, x1:x2].tostring()

                w_tmp = x2 - x1
                h_tmp = y2 - y1
                self.msg_img.height = h_tmp
                self.msg_img.width = w_tmp
                self.msg_img.step = w_tmp * 2

                self.msg_ci.height = h_tmp
                self.msg_ci.width = w_tmp
                self.msg_ci.roi.height = h_tmp
                self.msg_ci.roi.width = w_tmp


                print("DEPTH: ", x1, x2, y1, y2)
                cv2.rectangle(image_with_cnt, (int(boundRect[i][0]), int(boundRect[i][1])), \
              (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), self.cnt_colours[col], 2)
                #self.msg_img.header.stamp = rospy.Time.now()
                #self.msg_ci.header.stamp = rospy.Time.now()
                print("pcd requested at " , rospy.get_rostime())
                self.test_img_data_pub.publish(self.msg_img)
                self.test_cam_info_pub.publish(self.msg_ci)
                tmp_cnt = 0
                while (len(self.pl_ch_msg_list) == 0):
                    tmp_cnt += 1
                    if tmp_cnt%100 == 0:
                        self.test_img_data_pub.publish(self.msg_img)
                        self.test_cam_info_pub.publish(self.msg_ci)

                answer = self.pl_ch_msg_list.pop(0)
                print(answer)
                """
                try:
                    answer = rospy.wait_for_message("/plane_check_result", String, timeout = 0.25)
                    print(answer)
                    answer = answer.data
                except:
                    answer = ""
                """
                print("answer at " , rospy.get_rostime())
                if len(answer.split(",")[0]):
                    cv2.rectangle(image_with_cnt, (int(boundRect[i][0]), int(boundRect[i][1])), \
                  (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (255,255,255), 3)

                self.pl_ch_msg_list = []
                #time.sleep(10)
                #while(True): pass

        if DEBUG:
            tmp_bbox = Image()
            cv2.cvtColor(image_with_cnt, cv2.COLOR_BGR2RGB)
            tmp_bbox.data = image_with_cnt.tostring()
            tmp_bbox.height = image_with_cnt.shape[0]
            tmp_bbox.width = image_with_cnt.shape[1]
            tmp_bbox.header = image_data.header
            tmp_bbox.encoding = image_data.encoding
            tmp_bbox.is_bigendian = image_data.is_bigendian
            self.bbox_img_data_pub.publish(tmp_bbox)

        #cv2.addWeighted(overlay, alpha, output, 1 - alpha,
		#0, output)

        #while (True): pass
        print("--- %s seconds ---" % (time.time() - start_time))
        #while(True): pass
        #cv2.imwrite("test.jpg", image_with_cnt)
        #cv2.imshow("test.jpg", image_with_cnt)
        #cv2.imshow("depth", depth)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

    def get_countors(self, image, color):
        global CAL_PRINT_SINGLE
        if (CAL_PRINT_SINGLE): print(color)

        if len(self.hsv_filters[color]) == 2:
            mask = cv2.inRange(image, np.array(self.hsv_filters[color][0]), np.array(self.hsv_filters[color][1]))
            if (color == CAL_COLOR):
                pass
                #cv2.imshow("mask", mask)



        elif len(self.hsv_filters[color]) == 4:
            mask1 = cv2.inRange(image, np.array(self.hsv_filters[color][0]), np.array(self.hsv_filters[color][1]))
            mask2 = cv2.inRange(image, np.array(self.hsv_filters[color][2]), np.array(self.hsv_filters[color][3]))
            mask = mask1 + mask2
            if (color == CAL_COLOR):
                pass
                #cv2.imshow("mask1", mask1)
                #cv2.imshow("mask2", mask2)
                #cv2.imshow("mask", mask)




        countors_toRet = []

        if int(cv2.__version__[0]) > 3:
            # Opencv 4.x.x
            contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Opencv 3.x.x
            _, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        boundRect = []

        for i, cnt in enumerate(contours):
            if (cv2.contourArea(cnt) > self.area_thresh):
                approx = cv2.approxPolyDP(cnt, self.approx_coeff*cv2.arcLength(cnt, True), True)
                #print(len(approx))
                if len(approx) > self.max_lines_contour: continue
                countors_toRet.append(cnt)
                #boundRect.append(cv2.boundingRect(cv2.approxPolyDP(cnt, 3, True)))
                boundRect.append(cv2.boundingRect(approx))

        return countors_toRet, boundRect

if __name__ == '__main__':

    bd = block_detector()
    print("Block detector is running!")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down")
