import cv2, numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



mouse_pressed = False
s_x = s_y = e_x = e_y = -1

def mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed
    #print(x,y)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            image_to_show = np.copy(image)
            cv2.rectangle(image_to_show, (s_x, s_y),
                          (x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x, y

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

import os

for root, dirs, files in os.walk("2", topdown=False):
   for name in files:
        print(os.path.join(root, name))
        image = cv2.imread(os.path.join(root, name))
        image_to_show = np.copy(image)

        while True:
            cv2.imshow('image', image_to_show)
            k = cv2.waitKey(1)

            if k == ord('c'):

                x1 = s_x if s_x < e_x else e_x
                x2 = s_x if s_x > e_x else e_x
                y1 = s_y if s_y < e_y else e_y
                y2 = s_y if s_y > e_y else e_y

                x1 = 0 if x1 < 0 else x1
                x2 = 0 if x2 < 0 else x2
                y1 = 0 if y1 < 0 else y1
                y2 = 0 if y2 < 0 else y2

                x_cent = (x1 + x2 ) / 2
                y_cent = (y1 + y2 ) / 2
                print(x1, y1, x2, y2, x_cent, y_cent)
                crop = image_to_show[y1:y2, x1:x2]
                crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                #print(np.argmin(crop_hsv[0]), )
                vals, bins = np.histogram(crop_hsv[:,:,0], bins=60, range = (0, 180))
                #imgplot = plt.imshow(crop_hsv)
                #plt.show()
                print("H")
                #print(vals, bins)
                print(bins[:-1][vals!=0])
                """
                arg_max = np.argmax(vals)
                biggest = vals[arg_max]

                range = [bins[arg_max]]
                print(range)
                vals[arg_max] = 0
                while(biggest / float(np.max(vals)) < 10):
                    arg_max = np.argmax(vals)
                    biggest = vals[arg_max]
                    range.append(bins[arg_max])
                    vals[arg_max] = 0

                range = np.array([range])
                print(range)
                tmp_s = crop_hsv[:,:,1] #[crop_hsv[:,:,0] >= np.min(range)]
                print(np.max(tmp_s), np.min(tmp_s))
                """
                print("S")
                vals, bins = np.histogram(crop_hsv[:,:,1], bins=64, range = (0, 255))
                #imgplot = plt.imshow(crop_hsv)
                #plt.show()
                #print(vals, bins)
                print(bins[:-1][vals!=0])
                print("V")
                vals, bins = np.histogram(crop_hsv[:,:,2], bins=64, range = (0, 255))
                #imgplot = plt.imshow(crop_hsv)
                #plt.show()
                #print(vals, bins)
                print(bins[:-1][vals!=0])



            if k == ord('d'):

                a = input("low limit: ")
                b = input("upper limit: ")



            elif k == 27:
                break

cv2.destroyAllWindows()
