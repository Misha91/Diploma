#!/usr/bin/env python
import sys, time
import numpy as np

# Ros

from PIL import Image as PILimage


# Pytorch
import torch
# Our library
import network
import utils
import cv2
import os
import time
import random



def to_tensor(image):
    # normalize
    mean = [88, 113, 113]
    std = [50, 50, 59]
    image = image.astype(np.float32)
    image = (image - mean) / std
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    return image

def get_response():
    while(True):
        k = cv2.waitKey(1)
        if k == 27:
            print("rejected!")
            return False
        elif k == 32:
            print("accepted!")
            return True


weights_path = "net_epoch99"
model = network.Net()
model.load_state_dict(torch.load(weights_path, map_location='cpu'))


out_heat = {}
det_treshhold = 4.0
scales = [1]#, 1.5, 2, 3]
s_i = 0
threshold_NN = 4



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

for x in range(25):
    pic = all_pics[random.randint(0, len(all_pics) - 1)]
    print(x, pic)
    im = PILimage.open("val_set/3608.jpg")
    in_image = np.array(im)
    out_mask = np.zeros((in_image.shape[0], in_image.shape[1]))
    start = time.time()



    image_r = to_tensor(in_image)
    # evaluate model
    print(image_r.shape)
    output = model(image_r)
    print(output.shape)
    out_heat[s_i] = output[0, 0, :, :].detach().cpu().numpy()
    out_heat[s_i] -= threshold_NN
    out_heat[s_i][out_heat[s_i] < 0] = 0
    out_heat[s_i][out_heat[s_i] > 0] = 255
    tmp = PILimage.fromarray(out_heat[s_i])
    tmp = tmp.resize((in_image.shape[1], in_image.shape[0]))


    out_mask += np.array(tmp)
    out_mask[out_mask < 0] = 0
    out_mask[out_mask > 255] = 255

    s_i += 1

    out_mask = np.uint8(out_mask)
    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, hier = cv2.findContours(out_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Opencv 3.x.x
        _, contours, hier = cv2.findContours(out_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    countors_toRet = []
    for i, cnt in enumerate(contours):
        if (cv2.contourArea(cnt) > 300):
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

            countors_toRet.append(approx)

    for cnt in countors_toRet:


        cv2.drawContours(in_image, [cnt], 0, (255,255,255), 3)


    time_arr.append(time.time() - start)
    in_image = cv2.cvtColor(in_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("res", in_image)
    get_response()
    """
    try:
        if not (get_response()):
            fp += int(input("fp? +"))
            fn += int(input("fn? +"))

        tp += int(input("tp? +"))

    except:
        print("wrong input!")
    """
print("fp, fn, tp")
print(fp, fn, tp)
print("avg time per pic:")
print(np.mean(np.array([time_arr])))
