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




def to_tensor(image):
    # normalize
    mean = [118, 117, 117]
    std = [57, 58, 60]
    image = image.astype(np.float32)
    image = (image - mean) / std
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    return image


weights_path = "net_epoch99"
model = network.Net()
model.load_state_dict(torch.load(weights_path, map_location='cpu'))


out_heat = {}
det_treshhold = 4.0
scales = [1]#, 1.5, 2, 3]
s_i = 0
threshold_NN = 10



all_pics = []
fldr = {}
for root, dirs, files in os.walk("../hsv_tool", topdown=False):
    if "venv" in root: continue
    if "images" in root: continue
    for name in files:
       if name.endswith("jpg"):
           all_pics.append(os.path.join(root, name))
           if root not in fldr:
               fldr[root] = 1

all_pics.sort()

for pic in all_pics[::10]:
    im = PILimage.open(pic)
    in_image = np.array(im)
    out_mask = np.zeros((in_image.shape[0], in_image.shape[1]))
    start = time.time()
    for scale in scales:
        # resize image
        #im = PILimage.fromarray(in_image)
        image_r = im.resize((int(im.size[0] / scale), int(im.size[1] / scale)))
        image_r = np.array(image_r)
        # transform numpy to tensor
        image_r = to_tensor(image_r)
        # evaluate model
        output = model(image_r)
        print(output.shape, type(output))
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


    print(time.time() - start)
    tmp = PILimage.fromarray(out_mask)
    tmp.show()
    im = PILimage.fromarray(in_image)
    im.show()
    a = input("go?")

    """
    # maximum output of all scales
    max_idx = np.argmax([out_heat[0].max(), out_heat[1].max(), out_heat[2].max(), out_heat[3].max()])
    max_val = np.max([out_heat[0].max(), out_heat[1].max(), out_heat[2].max(), out_heat[3].max()])
    print("MAX_VAL", max_val)
    det_treshhold = 5.
    if max_val> det_treshhold:
    #   count +=1
    #if count > 2:
        det = True
        out_max = utils.max_filter(out_heat[max_idx], size=500)
        # get bbox of detection
        bbox = utils.bbox_in_image(
            np.zeros([int(in_image.shape[0] / scales[max_idx]), int(in_image.shape[1] / scales[max_idx])]), out_max,
            [32, 24], det_treshhold)
        width, height = im.size
        in_image[int(bbox[0, 1] * height):int(bbox[0, 3] * height),
        int(bbox[0, 0] * width):int(bbox[0, 0] * width) + 2, 1] = 255
        in_image[int(bbox[0, 1] * height):int(bbox[0, 3] * height),
        int(bbox[0, 2] * width) - 3:int(bbox[0, 2] * width) - 1, 1] = 255
        in_image[int(bbox[0, 1] * height):int(bbox[0, 1] * height) + 2,
        int(bbox[0, 0] * width):int(bbox[0, 2] * width), 1] = 255
        in_image[int(bbox[0, 3] * height) - 3:int(bbox[0, 3] * height) - 1,
        int(bbox[0, 0] * width):int(bbox[0, 2] * width), 1] = 255
    """
