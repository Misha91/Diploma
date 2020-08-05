#!/usr/bin/env python
import sys, time
import numpy as np


import cv2
import os

import random

hsv_filters = {}
hsv_filters['b']  = [[102, 135, 100], [108, 255, 255]]
hsv_filters['g']  = [[80 , 64, 44], [90, 255, 255]]
hsv_filters['r']  = [[0, 50, 85], [8, 255, 255], [155, 50, 85], [180, 255, 255]] #130 180
cnt_colours = {'b' : (255,0,0), 'g' : (0,255,0), 'r' : (0,0,255)}


def get_response():
    while(True):
        k = cv2.waitKey(1)
        if k == 27:
            print("rejected!")
            return False
        elif k == 32:
            print("accepted!")
            return True

def enlarge_crop(x1, x2, y1, y2, max_y, max_x):
    enlarge_coeff = 1.7
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    x_dist = (abs(x2 - x1) / 2.0) * enlarge_coeff
    y_dist = (abs(y2 - y1) / 2.0) * enlarge_coeff

    x1, x2, y1, y2 = x_center - x_dist, x_center + x_dist, y_center - y_dist, y_center + y_dist
    x1, x2, y1, y2 = 0 if x1 < 0 else x1, 0 if x2 < 0 else x2, 0 if y1 < 0 else y1,  0 if y2 < 0 else y2
    x1, x2, y1, y2 = max_x if x1 > max_x else x1, max_x if x2 > max_x else x2, max_y if y1 > max_y else y1,  max_y if y2 > max_y else y2

    return x1, x2, y1, y2

def get_countors(image, color):
    global hsv_filters
    area_thresh = 300
    max_lines_contour = 18
    approx_coeff = 0.01

    if len(hsv_filters[color]) == 2:
        mask = cv2.inRange(image, np.array(hsv_filters[color][0]), np.array(hsv_filters[color][1]))

    elif len(hsv_filters[color]) == 4:
        mask1 = cv2.inRange(image, np.array(hsv_filters[color][0]), np.array(hsv_filters[color][1]))
        mask2 = cv2.inRange(image, np.array(hsv_filters[color][2]), np.array(hsv_filters[color][3]))
        mask = mask1 + mask2

    countors_toRet = []

    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Opencv 3.x.x
        _, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    boundRect = []

    for i, cnt in enumerate(contours):
        if (cv2.contourArea(cnt) > 250):
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

            if len(approx) > 99: continue


            boundRect.append(cv2.boundingRect(approx))

    return boundRect




out_heat = {}
det_treshhold = 4.0
scales = [1]#, 1.5, 2, 3]
s_i = 0
threshold_NN = 5



all_pics = []

for root, dirs, files in os.walk("val_set", topdown=False):
    for name in files:
       if name.endswith("jpg"):
           all_pics.append(os.path.join(root, name))


all_pics.sort()

fp = 0
fn = 0
tp = 0
tn = 0
time_arr = []
countors = {}

for x in range(25):
    pic = all_pics[random.randint(0, len(all_pics) - 1)]
    print(x, pic)
    image = cv2.imread(pic)
    image_to_show = np.copy(image)
    start = time.time()
    output_mask = np.zeros_like(image)
    hsv = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2HSV)
    for col in hsv_filters.keys():
        color_mask = np.zeros_like(image)
        countors[col] = get_countors(hsv, col)
        if len(countors[col]):
            for cnt in countors[col]:
                y1,y2 = int(cnt[1]), int((cnt[1]+cnt[3]))
                x1, x2 = int(cnt[0]), int((cnt[0]+cnt[2]))
                cv2.rectangle(image_to_show, (x1, y1), (x2, y2), (255,255,255), 2)




    time_arr.append(time.time() - start)
    cv2.imshow("res", image_to_show)
    cv2.imshow("init", image)
    try:
        if not (get_response()):
            fp += int(input("fp? +"))
            fn += int(input("fn? +"))

        tp += int(input("tp? +"))

    except:
        print("wrong input!")

print("fp, fn, tp")
print(fp, fn, tp)
print("avg time per pic:")
print(np.mean(np.array([time_arr])))
