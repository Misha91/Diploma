#!/usr/bin/env python
import sys, time
import numpy as np

import cv2
import os

import random

hsv_filters = {}
hsv_filters['b']  = [[93, 87, 75], [108, 255, 255]]
hsv_filters['g']  = [[70 , 64, 44], [102, 255, 255]]
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
        if (cv2.contourArea(cnt) > area_thresh):
            approx = cv2.approxPolyDP(cnt, approx_coeff*cv2.arcLength(cnt, True), True)
            if len(approx) > max_lines_contour: continue
            countors_toRet.append(approx)


    return countors_toRet




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

for x in range(100):
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

                cv2.drawContours(image_to_show, [cnt], 0, (220,220,220), -1)



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
