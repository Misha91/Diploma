import cv2
import numpy as np
import os
import Queue
import time
#folder = "img_color/"
folder = "2/"

CALIBRATION_MODE = 0
CAL_SCREEN_INIT = 0
CAL_COLOR = 'g'
CAL_PRINT_SINGLE = 0
area_thresh = 400
approx_coeff = 0.01 #0.02
hsv_filters = {}
hsv_filters['b']  = [[95, 115, 45], [105, 255, 255], [105, 95, 45], [130, 255, 255]]
hsv_filters['g']  = [[70 , 95, 15], [85, 255, 255], [85 , 60, 30], [100, 238, 238]]
hsv_filters['r']  = [[0, 55, 0], [10, 255, 255], [170, 40, 0], [175, 255, 255]] #130 180
cnt_colours = {'b' : (255,0,0), 'g' : (0,255,0), 'r' : (0,0,255)}

def skel(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True

    return skel

def retErode(init):
    img = np.float32(init) / 255.0
    img = cv2.medianBlur(img, 5)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    #print(mag.dtype)
    coeff = float(255)/float(np.max(mag))
    mag = np.uint8(mag*coeff)

    gray = cv2.cvtColor(mag, cv2.COLOR_BGR2GRAY)
    gray = np.uint8(gray*(float(255)/float(np.max(gray))))
    cv2.imshow("gray", gray)
    #print(np.amax(gray))
    #gray = cv2.medianBlur(gray, 5)

    _, bin = cv2.threshold(gray,10,255,cv2.THRESH_BINARY) #+cv2.THRESH_OTSU
    erode = bin.copy()
    #erode = cv2.medianBlur(erode, 5)
    kernel = np.ones((2,2), np.uint8)
    #erode = cv2.erode(bin, kernel)
    erode = cv2.erode(erode, kernel, iterations=1)
    #erode = cv2.medianBlur(erode, 5)
    erode = skel(erode)
    cv2.imshow("bin", bin)
    return erode

def calibration(col):
    global CALIBRATION_MODE, CAL_SCREEN_INIT

    if CALIBRATION_MODE == 1:
        if CAL_SCREEN_INIT == 0:
            if len(hsv_filters[col]) == 2:
                cv2.namedWindow("Filter 1")
                cv2.createTrackbar("L-H", "Filter 1", hsv_filters[col][0][0], 180, nothing)
                cv2.createTrackbar("L-S", "Filter 1", hsv_filters[col][0][1], 255, nothing)
                cv2.createTrackbar("L-V", "Filter 1", hsv_filters[col][0][2], 255, nothing)
                cv2.createTrackbar("U-H", "Filter 1", hsv_filters[col][1][0], 180, nothing)
                cv2.createTrackbar("U-S", "Filter 1", hsv_filters[col][1][1], 255, nothing)
                cv2.createTrackbar("U-V", "Filter 1", hsv_filters[col][1][2], 255, nothing)
            if len(hsv_filters[col]) == 4:
                cv2.namedWindow("Filter 1")
                cv2.createTrackbar("L-H", "Filter 1", hsv_filters[col][0][0], 180, nothing)
                cv2.createTrackbar("L-S", "Filter 1", hsv_filters[col][0][1], 255, nothing)
                cv2.createTrackbar("L-V", "Filter 1", hsv_filters[col][0][2], 255, nothing)
                cv2.createTrackbar("U-H", "Filter 1", hsv_filters[col][1][0], 180, nothing)
                cv2.createTrackbar("U-S", "Filter 1", hsv_filters[col][1][1], 255, nothing)
                cv2.createTrackbar("U-V", "Filter 1", hsv_filters[col][1][2], 255, nothing)
                cv2.namedWindow("Filter 2")
                cv2.createTrackbar("L-H", "Filter 2", hsv_filters[col][2][0], 180, nothing)
                cv2.createTrackbar("L-S", "Filter 2", hsv_filters[col][2][1], 255, nothing)
                cv2.createTrackbar("L-V", "Filter 2", hsv_filters[col][2][2], 255, nothing)
                cv2.createTrackbar("U-H", "Filter 2", hsv_filters[col][3][0], 180, nothing)
                cv2.createTrackbar("U-S", "Filter 2", hsv_filters[col][3][1], 255, nothing)
                cv2.createTrackbar("U-V", "Filter 2", hsv_filters[col][3][2], 255, nothing)
            CAL_SCREEN_INIT = 1

        if len(hsv_filters[col]) == 2:
            hsv_filters[col][0][0] = cv2.getTrackbarPos("L-H", "Filter 1")
            hsv_filters[col][0][1] = cv2.getTrackbarPos("L-S", "Filter 1")
            hsv_filters[col][0][2] = cv2.getTrackbarPos("L-V", "Filter 1")
            hsv_filters[col][1][0] = cv2.getTrackbarPos("U-H", "Filter 1")
            hsv_filters[col][1][1] = cv2.getTrackbarPos("U-S", "Filter 1")
            hsv_filters[col][1][2] = cv2.getTrackbarPos("U-V", "Filter 1")
        if len(hsv_filters[col]) == 4:
            hsv_filters[col][0][0] = cv2.getTrackbarPos("L-H", "Filter 1")
            hsv_filters[col][0][1] = cv2.getTrackbarPos("L-S", "Filter 1")
            hsv_filters[col][0][2] = cv2.getTrackbarPos("L-V", "Filter 1")
            hsv_filters[col][1][0] = cv2.getTrackbarPos("U-H", "Filter 1")
            hsv_filters[col][1][1] = cv2.getTrackbarPos("U-S", "Filter 1")
            hsv_filters[col][1][2] = cv2.getTrackbarPos("U-V", "Filter 1")

            hsv_filters[col][2][0] = cv2.getTrackbarPos("L-H", "Filter 2")
            hsv_filters[col][2][1] = cv2.getTrackbarPos("L-S", "Filter 2")
            hsv_filters[col][2][2] = cv2.getTrackbarPos("L-V", "Filter 2")
            hsv_filters[col][3][0] = cv2.getTrackbarPos("U-H", "Filter 2")
            hsv_filters[col][3][1] = cv2.getTrackbarPos("U-S", "Filter 2")
            hsv_filters[col][3][2] = cv2.getTrackbarPos("U-V", "Filter 2")

def pressButton():
    print("\nPress Esc to continue...")
    key = cv2.waitKey()
    while(True):
        if key == 27:
            break


def manhDist(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def blackNeighb(img, point):
    pos = [ [0,1], [0,-1], [1,0], [-1,0], [1,1], [-1,1], [1,-1], [-1,-1] ]
    maxX = img.shape[1] - 1
    maxY = img.shape[0] - 1

    for p in pos:
        newX = point[0] + p[0]
        newY = point[1] + p[1]
        if (newX <= maxX) and (newX >= 0) and \
            (newY <= maxY) and (newY >= 0) and \
            img[newY][newX] == 0:
                return True
    return False

def checkAreaCorrect(orig, points):
    dummy = np.zeros((orig.shape[0],orig.shape[1],3) )

    #dummy[orig > 0] = np.array([255.,  0.,  0.])
    #dummy = orig.copy()
    pts_new = (np.array(np.mat(points)))[:, np.newaxis, :]
    #pts_new = pts_new


    cv2.fillPoly(dummy, pts =[pts_new], color=(255,0,0))
    #print(np.sum(orig ^ dummy), np.sum(orig))
    #i_and = cv.bitwise_and()
    print(dummy.dtype, orig.dtype)
    dummy = dummy.astype('uint8')
    dummy = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY)
    _, dummy = cv2.threshold(dummy, 10, 255, cv2.THRESH_BINARY)
    orig_copy = orig.copy()
    orig_copy = orig_copy.astype('uint8')
    orig_copy = cv2.cvtColor(orig_copy, cv2.COLOR_BGR2GRAY)
    _, orig_copy = cv2.threshold(orig_copy, 10, 255, cv2.THRESH_BINARY)
    xorM = cv2.bitwise_xor(dummy, orig_copy)
    andM = cv2.bitwise_and(dummy, orig_copy)
    print(dummy.dtype, orig_copy.dtype)
    print(dummy.shape, orig_copy.shape)
    xorVal = np.sum(xorM == 255)
    andVal = np.sum(andM == 255)
    origVal = np.sum(orig_copy == 255)
    print(xorVal, andVal, origVal)
    cv2.imshow("abc", dummy)
    return [xorVal, andVal, origVal]

def distantTake(shape, visited, skip, point):

    if skip == 1: return True

    pos = [ [0,1], [0,-1], [1,0], [-1,0], [1,1], [-1,1], [1,-1], [-1,-1] ]
    maxX = shape[1] - 1
    maxY = shape[0] - 1

    for p in pos:
        newX = point[0] + p[0]
        newY = point[1] + p[1]
        if (newX <= maxX) and (newX >= 0) and \
            (newY <= maxY) and (newY >= 0) and \
            [newX, newY] in visited:
                return False
    return True

def bfsContour(img, orig, x, y):
    pos = [ [0,1], [0,-1], [1,0], [-1,0], [1,1], [-1,1], [1,-1], [-1,-1] ]
    img_new = np.zeros((img.shape[0],img.shape[1],3) )
    img_new[img>0] = np.array([255., 255., 255.])
    start = [x,y]
    qStart = Queue.Queue()
    qStart.put([x,y])
    stVisited = [[x,y]]
    bfsVisited = []
    maxX = img.shape[1] - 1
    maxY = img.shape[0] - 1
    iterStart = 0
    maxSkip = 5

    while(not qStart.empty()):
        potStart = qStart.get()
        print(iterStart, potStart)
        img_new[potStart[1]][potStart[0]] = np.array([ 0.,  0.,  255.])

        if np.min(img[potStart[1]][potStart[0]] == np.array([ 255.,  255.,  255.])):
            print("NOT EMPTY!")
            qbfs = Queue.Queue()
            bfsVisited.append(potStart)
            qbfs.put([potStart])
            top = {str(potStart):[potStart]}
            merged = []
            while(not qbfs.empty()):
                buffer = []
                allBlack = 0
                bfsStart = qbfs.get()
                if bfsStart[-1] in merged:
                    merged.pop(merged.index(bfsStart[-1]))
                    continue

                del top[str(bfsStart[-1])]
                for point in bfsStart:
                    img_new[point[1]][point[0]] = np.array([255.,  0.,  255.])
                print(qbfs.qsize())
                #print(top.keys())
                #if len(bfsStart)>1 and bfsStart[-1] == potStart:
                #    print("WE ARE LOOPED!")
                #    break


                for p in pos:
                    for skip in range(1, maxSkip):
                        if skip != 1 and len(buffer): break

                        newXbfs = bfsStart[-1][0] + p[0]*skip
                        newYbfs = bfsStart[-1][1] + p[1]*skip
                        if len(bfsStart)>10 and str([newXbfs, newYbfs]) in top.keys() and manhDist(bfsStart[-10], top[str([newXbfs, newYbfs])][-10])>16:
                            if bfsStart[1] != top[str([newXbfs, newYbfs])][1]:
                                print("WE ARE LOOPED!")
                                print(bfsStart[-10], top[str([newXbfs, newYbfs])][-10], manhDist(bfsStart[-10], top[str([newXbfs, newYbfs])][-10]))
                                print(top[str([newXbfs, newYbfs])])
                                bfsStart += top[str([newXbfs, newYbfs])]

                                for point in bfsStart:
                                    img_new[point[1]][point[0]] = np.array([ 0.,  255.,  0.])
                                print(str([newXbfs, newYbfs]))
                                res = checkAreaCorrect(orig, bfsStart) #xor, and, orig
                                quality =float(float(res[1])/float(res[2]))
                                print(res, quality)
                                if quality < 0.1:
                                    pass

                                else:
                                    for b in buffer:
                                        b = b[:-1] + bfsStart + [b[-1]]
                                    qbfs.put(bfsStart)
                                    top[str([newXbfs, newYbfs])] = bfsStart
                                    merged.append([newXbfs, newYbfs])
                                    #print(bfsStart)


                                cv2.imshow("bfs", img_new)
                                pressButton()

                                if quality >= 0.95: return (np.array(np.mat(bfsStart)))[:, np.newaxis, :]

                                #return
                        if (newXbfs <= maxX) and (newXbfs >= 0) and \
                            (newYbfs <= maxY) and (newYbfs >= 0) and \
                            ([newXbfs, newYbfs] not in bfsVisited) and \
                            img[newYbfs][newXbfs] != 0 and distantTake(img.shape, bfsVisited, skip, [newXbfs, newYbfs]): #and blackNeighb(img, [newXbfs, newYbfs])

                            tmpPath = bfsStart[:]
                            tmpPath.append([newXbfs, newYbfs])
                            buffer.append(tmpPath)
                            img_new[tmpPath[-1][1]][tmpPath[-1][0]] = np.array([ 255.,  0.,  0.])
                            print("adding", [newXbfs, newYbfs])
                for b in buffer:
                    qbfs.put(b)
                    bfsVisited.append(b[-1])
                    top[str(b[-1])] = b


                #time.sleep(0.5)
                cv2.imshow("bfs", img_new)
                #pressButton()

                for point in bfsStart:
                    img_new[point[1]][point[0]] = np.array([255.,  0.,  0.])

        for p in pos:
            newX = potStart[0] + p[0]
            newY = potStart[1] + p[1]
            if (newX <= maxX) and (newX >= 0) and \
                (newY <= maxY) and (newY >= 0) and ([newX, newY] not in stVisited):

                qStart.put([newX, newY])
                stVisited.append([newX, newY])
                print("adding", [newX, newY])

        iterStart += 1
        if (iterStart >= 100): break
        cv2.imshow("bfs", img_new)

        #pressButton()


def magicSearch(image, cnt, corner):
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print(cX,cY)
    erode = retErode(image)

    img = np.zeros((image.shape[0],image.shape[1],3) ) # create a single channel 200x200 pixel black image
    print(cnt.shape)
    cv2.fillPoly(img, pts =[cnt], color=(255,255,255))
    if (np.min(img[cY][cX] == np.array([ 255.,  255.,  255.]))):
        img[cY][cX] = np.array([ 0.,  0.,  255.])
        cv2.imshow("mS", img)
        corr_cnt = bfsContour(erode, img, corner[0], corner[1])
        return corr_cnt


    pressButton()

def get_countors(image, color):
    global area_thresh, approx_coeff, CAL_PRINT_SINGLE
    if (CAL_PRINT_SINGLE): print(color)

    erode = retErode(image)

    if len(hsv_filters[color]) == 2:
        mask = cv2.inRange(image, np.array(hsv_filters[color][0]), np.array(hsv_filters[color][1]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)
        if (color == CAL_COLOR):
            cv2.imshow("mask", mask)
            cv2.imshow("erode", erode)
        #print(tuple(hsv_filters[color][0]))
        #print(tuple(hsv_filters[color][1]))

    elif len(hsv_filters[color]) == 4:
        mask1 = cv2.inRange(image, np.array(hsv_filters[color][0]), np.array(hsv_filters[color][1]))
        mask2 = cv2.inRange(image, np.array(hsv_filters[color][2]), np.array(hsv_filters[color][3]))
        mask = mask1 + mask2
        if (color == CAL_COLOR):
            #cv2.imshow("mask1", mask1)
            #cv2.imshow("mask2", mask2)
            cv2.imshow("mask", mask)
            cv2.imshow("erode", erode)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)

    else:
        return []

    countors_toRet = []

    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Opencv 3.x.x
        _, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    if color == "r": print(len(contours))
    #print(color, hier)
    for i, cnt in enumerate(contours):
        if color == "r": print(cv2.contourArea(cnt))
        if (cv2.contourArea(cnt) > area_thresh):
            approx = cv2.approxPolyDP(cnt, approx_coeff*cv2.arcLength(cnt, True), True)
            if (CAL_PRINT_SINGLE): print(len(approx))
            #if len(approx) <= 10:
            #print(i)

            corner = np.where(approx == np.amax(approx))
            corner = (approx[corner[0]][corner[1]])[0][0]
            #print(approx)
            #print(corner)
            bfs_cnt = magicSearch(image, cnt, corner)
            countors_toRet.append(approx)
            if len(bfs_cnt): countors_toRet.append(bfs_cnt)
            hull.append(cv2.convexHull(cnt, False))

    return countors_toRet
    #return hull

def nothing(x):
    # any operation
    pass

def get_blocks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_with_cnt = image.copy()
    countors = {}
    for col in hsv_filters.keys():
        countors[col] = get_countors(hsv, col)
        for cnt in countors[col]:
            cv2.drawContours(image_with_cnt, [cnt], 0, cnt_colours[col], 5)
    return image_with_cnt

def test_function():
    global CAL_PRINT_SINGLE, CALIBRATION_MODE
    #for x in range(0,367):
    if CALIBRATION_MODE == 1:
        print("*************")
        CAL_PRINT_SINGLE = 1
    #print(str(x) + ".jpg")
    #while (True):
        #init = cv2.imread(folder + str(x) + ".jpg")
    #init = cv2.imread(folder + str(0) + ".jpg")
    init = cv2.imread("orig.jpg")
    print(init)
    image_with_cnt = get_blocks(init)


    cv2.imshow("abc", image_with_cnt)
    calibration(CAL_COLOR)
    key = cv2.waitKey()
    CAL_PRINT_SINGLE = 0
    while(True):
        pass
        if key == 27:
            break

    cv2.destroyAllWindows()

test_function()
