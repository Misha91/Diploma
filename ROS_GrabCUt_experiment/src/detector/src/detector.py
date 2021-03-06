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

DEBUG = 1
CALIBRATION_MODE = 0
CAL_SCREEN_INIT = 0
CAL_COLOR = 'g'
CAL_PRINT_SINGLE = 0

iter = 0

class block_detector:
    def __init__(self):
        rospy.init_node('block_detector')
        self.image_subscriber = message_filters.Subscriber("/uav/rs_d435/color/image_raw", Image) #/uav/rs_d435/color/image_raw
        self.cam_info_subscriber = message_filters.Subscriber("/uav/rs_d435/color/camera_info", CameraInfo)
        self.depth_subscriber = message_filters.Subscriber("/uav/rs_d435/depth/image_rect_raw", Image)
        self.depth_info_subscriber = message_filters.Subscriber("/uav/rs_d435/depth/camera_info", CameraInfo)
        self.plane_check_subscriber = rospy.Subscriber("/plane_check_result", String, self.plane_check_callback)
        self.pl_ch_msg_list = []
        self.abc_subscriber = rospy.Subscriber("/uav/rs_d435/color/image_raw", Image, self.callback)
        self.bbox_img_data_pub = rospy.Publisher('bbox_image', Image, queue_size=10)
        self.bbox_cand_img_data_pub = rospy.Publisher('bbox_image_candidate', Image, queue_size=10)
        self.test_img_data_pub = rospy.Publisher('image_rect', Image, queue_size=10)
        self.test_cam_info_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=10)
        self.msg_img = Image()
        self.msg_ci = CameraInfo()
        self.coeff_ratio = []
        self.first_message = True
        self.P_img = np.zeros((3,4))
        self.K_img = np.eye(3)
        self.K_img_inv = np.eye(3)
        self.K_depth = np.eye(3)
        self.K_conv_depth = np.eye(3)
        self.K_conv = np.eye(3)

        self.enlarge_coeff = 1.25 #1.4

        # time synchronizer
        #self.ts = message_filters.ApproximateTimeSynchronizer([self.image_subscriber,  self.depth_subscriber, self.cam_info_subscriber, self.depth_info_subscriber], 5, 0.1)
        #self.ts.registerCallback(self.callback)
        self.area_thresh = 400
        self.max_lines_contour = 15
        self.approx_coeff = 0.01 #0.02
        self.hsv_filters = {}
        """
        self.hsv_filters['b']  = [[97, 140, 75], [108, 255, 255]]
        self.hsv_filters['g']  = [[78 , 65, 35], [97, 255, 255]]
        self.hsv_filters['r']  = [[0, 100, 80], [6, 255, 255], [170, 100, 80], [180, 255, 255]] #130 180
        """
        self.hsv_filters['b']  = [[100, 140, 75], [108, 255, 255]]
        self.hsv_filters['g']  = [[78 , 65, 35], [93, 255, 255]]
        self.hsv_filters['r']  = [[0, 100, 80], [6, 255, 255], [170, 100, 80], [180, 255, 255]] #130 180
        self.cnt_colours = {'b' : (255,0,0), 'g' : (0,255,0), 'r' : (0,0,255)}


    def calc_region_area(self, x1, x2, y1, y2):
        x_dist = abs(x2 - x1)
        y_dist = abs(y2 - y1)
        return x_dist * y_dist

    def enlarge_crop(self, x1, x2, y1, y2, max_y, max_x):

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        x_dist = (abs(x2 - x1) / 2.0) * self.enlarge_coeff
        y_dist = (abs(y2 - y1) / 2.0) * self.enlarge_coeff

        x1, x2, y1, y2 = x_center - x_dist, x_center + x_dist, y_center - y_dist, y_center + y_dist
        x1, x2, y1, y2 = 0 if x1 < 0 else x1, 0 if x2 < 0 else x2, 0 if y1 < 0 else y1,  0 if y2 < 0 else y2
        x1, x2, y1, y2 = max_x if x1 > max_x else x1, max_x if x2 > max_x else x2, max_y if y1 > max_y else y1,  max_y if y2 > max_y else y2

        return x1, x2, y1, y2


    def plane_check_callback(self, msg):
        self.pl_ch_msg_list.append(msg.data)

    def callback(self, image_data):
        global iter
        np_data = np.fromstring(image_data.data, np.uint8)

        in_image = cv2.cvtColor(np_data.reshape(image_data.height, image_data.width,3), cv2.COLOR_RGB2BGR)
        cv2.imwrite("3/" + str(iter) + ".jpg", in_image)
        iter += 1
        """
        #in_image = np_data.reshape(image_data.height, image_data.width,3)
        np_data = np.fromstring(depth_data.data, np.uint16)
        in_depth = np_data.reshape(depth_data.height, depth_data.width)
        in_depth[np.isnan(in_depth)] = 0



        #in_depth = in_depth/10

        #cv2.imwrite("color.jpg", in_image)
        #cv2.imwrite("depth.jpg", in_depth)


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
            print(self.msg_ci.P)
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
                area = self.calc_region_area(x1, x2, y1, y2)
                bboxes.append([area, x1, x2, y1, y2, col])

        bbox_voter = np.zeros((in_image.shape[0], in_image.shape[1]), dtype=np.uint8)
        bboxes.sort(reverse=True)
        for bbox in bboxes:
            area, x1, x2, y1, y2, col = bbox
            dummy = np.zeros((in_image.shape[0], in_image.shape[1]), dtype=np.uint8)
            dummy[y1:y2, x1:x2] = 1
            a =  (np.logical_and(bbox_voter, dummy)).astype(np.uint8)
            intersection = np.sum(a) / float(area)
            print("COLOR_IMG: ", col, x1, x2, y1, y2, intersection)

            #print(bbox)
            if intersection < 0.8:
            #if True:
                bbox_voter[y1:y2, x1:x2] = 1

                x1, x2, y1, y2 = self.enlarge_crop(x1, x2, y1, y2, in_image.shape[0]-1, in_image.shape[1]-1)

                print(x1, x2, y1, y2)

                img = image_with_candidate.copy()

                mask = np.zeros(img.shape[:2],np.uint8)

                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)

                rect = (int(x1), int(y1), int(x2), int(y2))
                start_time = time.time()
                cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

                mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                img = img*mask2[:,:,np.newaxis]


                mask1 = cv2.inRange(hsv, np.array(self.hsv_filters['r'][0]), np.array(self.hsv_filters['r'][1]))
                mask2 = cv2.inRange(hsv, np.array(self.hsv_filters['r'][2]), np.array(self.hsv_filters['r'][3]))
                mask3 = mask1 + mask2
                mask[mask3 != 0] = 0
                mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

                mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                img2 = img*mask[:,:,np.newaxis]
                print("--- %s seconds ---" % (time.time() - start_time))
                cv2.rectangle(image_with_candidate, (int(x1), int(y1)), (int(x2), int(y2)), self.cnt_colours[col], 2)

                x1, y1, _ = (self.K_conv.dot(np.array([x1, y1, 1]))).astype(int)
                x2, y2, _ = (self.K_conv.dot(np.array([x2, y2, 1]))).astype(int)
                #print("DEPTH: ", x1, x2, y1, y2)
                mean_dist = in_depth[y1:y2, x1:x2].mean()
                print(mean_dist, area)
                if (mean_dist < 700): continue

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








                cv2.imshow("cand", image_with_candidate)

                cv2.imshow("GrabCut", img)
                cv2.imshow("GrabCut_update", img2)
                cv2.waitKey()
                cv2.destroyAllWindows()

                self.test_img_data_pub.publish(self.msg_img)
                self.test_cam_info_pub.publish(self.msg_ci)


                while (len(self.pl_ch_msg_list) == 0):
                    pass

                answer = self.pl_ch_msg_list.pop(0)
                print(answer)
                if len(answer) > 2:
                    zones = answer.split(",")[0]
                    zones = zones[:-1].split("x")
                    tmp_buffer = []
                    if len(zones) and len(zones[0]):
                        tmp_buffer = []
                        for z in zones:

                            tmp = z.split("#")
                            print(tmp)
                            area = float(tmp[0])
                            proj = self.P_img.dot(np.array([tmp[1], tmp[2], tmp[3], 1.], dtype=float))
                            proj /= proj[-1]

                            tmp_buffer.append([area, proj])
                            print(area, mean_dist)

                    tmp_buffer.sort(reverse=True)
                    for i, z in enumerate(tmp_buffer):
                        x1, y1, _ = z[1]
                        #cv2.circle(image_with_cnt, (int((bbox[1] + bbox[2]) / 2. + x1), int((bbox[3] + bbox[4])/2. + y1)) , 10, self.cnt_colours[col], 5)
                        if tmp_buffer[0][0]/float(z[0]) >= 7 or i >= 2: break
                    cv2.rectangle(image_with_cnt, (bbox[1], bbox[3]), (bbox[2], bbox[4]), self.cnt_colours[col], 2)

                #self.pl_ch_msg_list = []
                #time.sleep(10)
                #while(True): pass

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



        print("--- %s seconds ---" % (time.time() - start_time))
        #while(True): pass
        #cv2.imwrite("test.jpg", image_with_cnt)
        #cv2.imshow("test.jpg", image_with_cnt)
        #cv2.imshow("depth", depth)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        """

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
