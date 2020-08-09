import cv2, numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

mouse_pressed = False
s_x = s_y = e_x = e_y = -1

image = np.empty((0,0))
image_to_show = np.empty((0,0))

hsv_filters = {}
hsv_filters['b']  = [[93, 87, 75], [108, 255, 255]]
hsv_filters['g']  = [[70 , 64, 44], [102, 255, 255]]
hsv_filters['r']  = [[0, 50, 85], [8, 255, 255], [155, 50, 85], [180, 255, 255]] #130 180
cnt_colours = {'b' : (255,0,0), 'g' : (0,255,0), 'r' : (0,0,255)}

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
                          (x, y), (255, 255, 255), 3)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x, y

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

def get_response():
    while(True):
        k = cv2.waitKey(1)
        if k == 27:
            print("rejected!")
            return False
        elif k == 32:
            print("accepted!")
            return True

def create_dataset():
    global hsv_filters, cnt_colours
    all_pics = []
    countors = {}
    h_hist_storage = {}
    s_hist_storage = {}
    v_hist_storage = {}

    fldr = {}
    offset = 0
    answr = input("del? 1/0 ")

    if int(answr) == 1:
        answr = input("sure? 1/0 ")
        if int(answr) == 1:
            for root, dirs, files in os.walk("images/", topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

            for root, dirs, files in os.walk("labels/", topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

        assert not os.system("mkdir -p labels;mkdir -p images"), "CANT MAKE FOLDERS"
    for root, dirs, files in os.walk(".", topdown=False):
        if "venv" in root: continue
        if "images" in root: continue
        for name in files:
           if name.endswith("jpg"):
               all_pics.append(os.path.join(root, name))
               if root not in fldr:
                   fldr[root] = 1

    all_pics.sort()
    print(all_pics[1252])
    if int(answr) != 1:
        print("Select folder to start with:")
        for i, f in enumerate(fldr.keys()):
            print(str(i + 1) + ". " + str(f))


        while(True):
            try:
                answr = input("Your answer: ")
                answr = int(answr) - 1
                if answr < len(fldr.keys()):
                    fn = fldr.keys()[answr] + "/"
                    answr = input("file id: ")
                    fn += str(answr) + ".jpg"
                    offset = all_pics.index(fn)
                    break


            except Exception as e:
                print(e)
                print("Try again!")

    #print(all_pics)
    for i in range(len(all_pics)):
        pic = all_pics[random.randint(0, len(all_pics) - 1)]
        #pic = all_pics[i]
        image = cv2.imread(pic)
        image_to_show = np.copy(image)

        output_mask = np.zeros_like(image)
        hsv = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2HSV)
        for col in hsv_filters.keys():
            color_mask = np.zeros_like(image)
            countors[col] = get_countors(hsv, col)
            if len(countors[col]):
                for cnt in countors[col]:
                    image_to_show_tmp = image_to_show.copy()
                    cv2.drawContours(image_to_show_tmp, [cnt], 0, cnt_colours[col], 3)
                    cv2.imshow('image', image_to_show_tmp)
                    if (get_response()):
                        cv2.drawContours(image_to_show, [cnt], 0, (220,220,220), -1)
                        cv2.drawContours(output_mask, [cnt], 0, (255,255,255), -1)
                        cv2.drawContours(color_mask, [cnt], 0, (255,255,255), -1)


            crop_hsv = np.zeros_like(hsv)
            #crop_hsv[crop_hsv == 0] = -1
            crop_hsv[color_mask == 255] = hsv[color_mask == 255]
            #crop_hsv[crop_hsv >= 252] = -1
            print(np.sum(crop_hsv >= 252))
            vals, bins = np.histogram(crop_hsv[:,:,0], bins=180, range = (1, 179))
            if col not in h_hist_storage:
                h_hist_storage[col] = np.zeros_like(vals)
            h_hist_storage[col] += vals


            vals, bins = np.histogram(crop_hsv[:,:,1], bins=255, range = (1, 254))
            if col not in s_hist_storage:
                s_hist_storage[col] = np.zeros_like(vals)
            s_hist_storage[col] += vals


            vals, bins = np.histogram(crop_hsv[:,:,2], bins=255, range = (1, 254))
            if col not in v_hist_storage:
                v_hist_storage[col] = np.zeros_like(vals)
            v_hist_storage[col] += vals

        ind = (offset + i)
        print(ind)
        f = open("hsv/" + str(ind) + "_hsv.txt", "w")
        for col in hsv_filters.keys():
            f.write("\n" + str(col) + "\n")
            f.write("H\n")
            np.savetxt(f, h_hist_storage[col], delimiter=',')
            f.write("\nS\n")
            np.savetxt(f, s_hist_storage[col], delimiter=',')
            f.write("\nV\n")
            np.savetxt(f, v_hist_storage[col], delimiter=',')
        f.close()

        cv2.imwrite("images/" + (8 - len(str(ind*4)))*"0" + str(ind*4) + ".jpg", image)
        cv2.imwrite("labels/" + (8 - len(str(ind*4)))*"0" + str(ind*4) + ".png", cv2.resize(output_mask, (80,40)))

        cv2.imwrite("images/" + (8 - len(str(ind*4 + 1)))*"0" + str(ind*4 + 1) + ".jpg", cv2.resize(image, (426, 213)))
        cv2.imwrite("labels/" + (8 - len(str(ind*4 + 1)))*"0" + str(ind*4 + 1) + ".png", cv2.resize(output_mask, (53,26)))

        cv2.imwrite("images/" + (8 - len(str(ind*4 + 2)))*"0" + str(ind*4 + 2) + ".jpg", cv2.resize(image, (320, 160)))
        cv2.imwrite("labels/" + (8 - len(str(ind*4 + 2)))*"0" + str(ind*4 + 2) + ".png", cv2.resize(output_mask, (40,20)))

        cv2.imwrite("images/" + (8 - len(str(ind*4 + 3)))*"0" + str(ind*4 + 3) + ".jpg", cv2.resize(image, (213, 106)))
        cv2.imwrite("labels/" + (8 - len(str(ind*4 + 3)))*"0" + str(ind*4 + 3) + ".png", cv2.resize(output_mask, (26,16)))

    plt.figure(100)
    plt.bar(np.arange(len(h_hist_storage['g'])), h_hist_storage['g'], width=3,bottom=0, color = (0.0,1.0,0.0,0.7))
    plt.bar(np.arange(len(h_hist_storage['b'])), h_hist_storage['b'], width=3,bottom=0, color = (0.0,0.0,1.0,0.7))
    plt.bar(np.arange(len(h_hist_storage['r'])), h_hist_storage['r'], width=3,bottom=0, color = (1.0,0.0,0.0,0.7))
    # Add title and axis names
    plt.title('Hue Distribution Histogram')
    plt.xlabel('Hue')
    plt.ylabel('Number of pixels')


    plt.figure(200)
    plt.bar(np.arange(len(s_hist_storage['g'])), s_hist_storage['g'], width=3,bottom=0, color = (0.0,1.0,0.0,0.7))
    plt.bar(np.arange(len(s_hist_storage['b'])), s_hist_storage['b'], width=3,bottom=0, color = (0.0,0.0,1.0,0.7))
    plt.bar(np.arange(len(s_hist_storage['r'])), s_hist_storage['r'], width=3,bottom=0, color = (1.0,0.0,0.0,0.7))
    # Add title and axis names
    plt.title('Saturation Distribution Histogram')
    plt.xlabel('Saturation')
    plt.ylabel('Number of pixels')

    plt.figure(300)
    plt.bar(np.arange(len(v_hist_storage['g'])), v_hist_storage['g'], width=3,bottom=0, color = (0.0,1.0,0.0,0.7))
    plt.bar(np.arange(len(v_hist_storage['b'])), v_hist_storage['b'], width=3,bottom=0, color = (0.0,0.0,1.0,0.7))
    plt.bar(np.arange(len(v_hist_storage['r'])), v_hist_storage['r'], width=3,bottom=0, color = (1.0,0.0,0.0,0.7))
    # Add title and axis names
    plt.title('Value Distribution Histogram')
    plt.xlabel('Value')
    plt.ylabel('Number of pixels')
    plt.show()




def hsv_hist():
    global image, image_to_show
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)
    print("Image will be runned one by one. Select zone and press \"c\" to get histogram or Esc for next image")
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

                    print("H")
                    vals, bins = np.histogram(crop_hsv[:,:,0], bins=180, range = (1, 179))
                    print(vals)
                    #print(bins[:-1][vals!=0])

                    plt.figure(100)
                    plt.bar(np.arange(len(vals)), vals, width=3,bottom=0, color = (1.0,0.0,0.0,0.7))

                    # Add title and axis names
                    plt.title('Hue Distribution Histogram')
                    plt.xlabel('Hue')
                    plt.ylabel('Number of pixels')

                    print("S")
                    vals, bins = np.histogram(crop_hsv[:,:,1], bins=255, range = (1, 254))
                    print(vals)
                    #print(bins[:-1][vals!=0])
                    plt.figure(200)

                    plt.bar(np.arange(len(vals)), vals, width=3,bottom=0, color = (1.0,0.0,0.0,0.7))
                    # Add title and axis names
                    plt.title('Saturation Distribution Histogram')
                    plt.xlabel('Saturation')
                    plt.ylabel('Number of pixels')

                    print("V")
                    vals, bins = np.histogram(crop_hsv[:,:,2], bins=255, range = (1, 254))
                    print(vals)
                    #print(bins[:-1][vals!=0])






                    plt.figure(300)

                    plt.bar(np.arange(len(vals)), vals, width=3,bottom=0, color = (1.0,0.0,0.0,0.7))
                    # Add title and axis names
                    plt.title('Value Distribution Histogram')
                    plt.xlabel('Value')
                    plt.ylabel('Number of pixels')
                    plt.show()

                elif k == 27:
                    break

    cv2.destroyAllWindows()


a = input("Hello and wellcome to the HSV tool! Please select mode:\n1. Get HSV histogram for choosen zone\n2. Create dataset based on given range\nYour answer: ")
while(True):

    try:
        mode = int(a)

        if mode == 1:
            hsv_hist()
            break

        elif mode == 2:
            create_dataset()
            break

        else: raise Exception()
    except Exception as e:
        print(e)
        print("Wrong input! Please try again!")
        a = input("Your answer: ")
