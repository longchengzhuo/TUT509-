import cv2
import time
import math
import numpy as np


def distance_erweima(a,frame):
    copy_src = frame.copy()
    L = np.zeros(a.shape)

    contours, hierarchy = cv2.findContours(
            a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dis = []
    J = []

    print("-------------------------")
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        rectpoint = cv2.boxPoints(rect)
        widthC, heigthC = rect[1]
        if ((widthC == 0) | (heigthC == 0)) == True:
            continue
        angleCache = rect[2]
        rectpoint = np.int0(rectpoint)
        rectpoint[0][1] = rectpoint[0][1] + 240
        rectpoint[1][1] = rectpoint[1][1] + 240
        rectpoint[2][1] = rectpoint[2][1] + 240
        rectpoint[3][1] = rectpoint[3][1] + 240
        rectbox1x = []
        rectbox2x = []
        rectbox3x = []
        rectbox4x = []
        rectbox1y = []
        rectbox2y = []
        rectbox3y = []
        rectbox4y = []
        rectbox11 = []
        rectbox22 = []
        rectbox33 = []
        rectbox44 = []
        if angleCache < -45:
            v = heigthC
            heigthC = widthC
            widthC = v


        if 1.1>= (heigthC / widthC) >= 0.85:  # 灯条是竖直放置，长宽比满足条件

            G = int((heigthC + widthC) / 2)

            cv2.drawContours(frame, [rectpoint], -1, (0, 255, 0,), 2)
            rectbox1x.append(int(rectpoint[0][0]))
            rectbox2x.append(int(rectpoint[1][0]))
            rectbox3x.append(int(rectpoint[2][0]))
            rectbox4x.append(int(rectpoint[3][0]))
            rectbox1y.append(int(rectpoint[0][1]))
            rectbox2y.append(int(rectpoint[1][1]))
            rectbox3y.append(int(rectpoint[2][1]))
            rectbox4y.append(int(rectpoint[3][1]))

            rectbox11.append([rectbox1x[0],
                              rectbox1y[0]])
            rectbox22.append([rectbox2x[0],
                              rectbox2y[0]])
            rectbox33.append([rectbox3x[0],
                              rectbox3y[0]])
            rectbox44.append([rectbox4x[0],
                              rectbox4y[0]])
            rectbox = rectbox11 + rectbox22 + rectbox33 + rectbox44

            zsy = rectbox1y[0] - G * 4
            yxy = rectbox1y[0]
            zsx = rectbox1x[0] - G
            yxx = rectbox1x[0] + G * 2

            Q = copy_src[zsy:yxy, zsx:yxx]

            img_points = np.array(rectbox, dtype=np.double)


            obj_points = [[-55., -55., 0.],
                          [55., -55., 0.],
                          [55., 55., 0.],
                          [-55., 55., 0.]]
            obj_points = np.reshape(obj_points, (4, 3))

            success, rvecs, tvecs = cv2.solvePnP(obj_points, img_points,
                                                 np.array([[690.89279836, 0., 419.30439067],
                                                           [0., 678.23485916, 324.65639788],
                                                           [0., 0., 1.]], dtype=np.double),
                                                 np.array([[-0.38759803, -0.24131306, -0.01149803,  0.008781,   1.92499479]],
                                                          dtype=np.double))  # 这些都是智能车的参数,摄像头是真垃圾
            print("距离", tvecs[2][0])
            dis.append(tvecs[2][0])
            cv2.putText(frame, 'The dis is %d mm' % np.mean(dis), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)

    if dis != []:
        print("距离均值", np.mean(dis))
        if 1500 <= np.mean(dis) <=2200:
            if (L != Q) and (Q != []):
                J = Q
    print("----------------------------------------------")
    return J





def honglvdeng(img, frame):

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    # HSV 红色
    lower_red = np.array([150, 108, 75])
    upper_red = np.array([179, 209, 255])
    red = cv2.inRange(imgHSV, lower_red, upper_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    r = cv2.dilate(red, kernel)
    r = cv2.erode(r, kernel)


    # HSV 绿色
    lower_green = np.array([19, 41, 54])
    upper_green = np.array([91, 153, 255])
    green = cv2.inRange(imgHSV, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    g = cv2.dilate(green, kernel)
    g = cv2.erode(g, kernel)


    g_contours, hierarchy = cv2.findContours(
        g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r_contours, hierarchy1 = cv2.findContours(
        r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if (g_contours != []) or (r_contours != []):
        if g_contours == []:
            cv2.putText(frame, "RED", (300, 300), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        elif r_contours == []:
            cv2.putText(frame, "GREEN", (300, 300), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        else:
            RR = 0
            GG = 0
            for r_contour in r_contours:
                rect_r = cv2.minAreaRect(r_contour)
                xrCache, yrCache = rect_r[1]
                RR = RR + xrCache + yrCache
            for g_contour in g_contours:
                rect_g = cv2.minAreaRect(g_contour)
                xgCache, ygCache = rect_g[1]
                GG = GG + xgCache + ygCache

            if RR > GG:
                cv2.putText(frame, "RED", (300, 300), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            else:
                cv2.putText(frame, "GRENN", (300, 300), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)







cap = cv2.VideoCapture('ldd.webm')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))


def tra():


    while(cap.isOpened()):
        print("1")
        ret, frame = cap.read()
        if dengdeng != []:
            continue
        t1 = cv2.getTickCount()
        img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proimage0 = img1.copy()
        ROI = np.zeros(img1.shape, np.uint8)  # 感兴趣区域ROI
        ROI[240:430, 0:801] = 1
        masked_img = cv2.bitwise_and(proimage0, proimage0, mask=ROI)  # 图像交运算 ，获取的是原图处理——提取轮廓后的ROI
        masked_img = cv2.bitwise_not(masked_img)
        a = masked_img[240:430, 0:801]
        ret2, k = cv2.threshold(a, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        k = cv2.erode(k, kernel)
        a = cv2.dilate(k, kernel)
        srcc = cv2.erode(a, kernel)

        dengdeng = distance_erweima(srcc, frame)
        print(dengdeng)
        if dengdeng != []:
            print("2",dengdeng)
            honglvdeng(np.array(dengdeng), frame)
        t2 = cv2.getTickCount()
        spendTime = (t2 - t1) * 1 / (cv2.getTickFrequency())
        FPS = 1 / spendTime
        FPS = 'The fps is %d' %(FPS)

        print("# 嘉然带我走吧，一天极速爆肝肝要没了")
        cv2.putText(frame, FPS, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        cv2.imshow('frame', frame)
        out.write(frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()



    cv2.destroyAllWindows()

tra()
