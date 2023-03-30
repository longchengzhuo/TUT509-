import cv2
import time
import math
import numpy as np

src = cv2.imread("D:/biancheng\opencv python\zhinengche/hongdeng.jpg")
copy_src = src.copy()

def distance_erweima(src):
    img1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    proimage0 = img1.copy()
    ROI = np.zeros(img1.shape, np.uint8) #感兴趣区域ROI
    ROI[280:430, 0:801 ] = 1
    masked_img = cv2.bitwise_and(proimage0, proimage0, mask=ROI)                   #图像交运算 ，获取的是原图处理——提取轮廓后的ROI
    masked_img = cv2.bitwise_not(masked_img)
    a = masked_img[280:430, 0:801 ]
    ret2, k = cv2.threshold(a, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    k = cv2.erode(k, kernel)
    a = cv2.dilate(k, kernel)
    a = cv2.erode(a, kernel)
    contours, hierarchy = cv2.findContours(
            a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓
    cv2.drawContours(src, contours, -1, (0, 255, 0,), 2)
    cv2.imshow("debug", a)
    dis = []
    J = 0
    for contour in contours:
        print("1")
        rect = cv2.minAreaRect(contour)  # 获取最小包围矩形
        xCache, yCache = rect[0]  # 获取矩形的中心坐标
        rectpoint = cv2.boxPoints(rect)  # 获取矩形四个角点
        widthC, heigthC = rect[1]  # 获取矩形的长宽
        if ((widthC == 0) | (heigthC == 0)) == True:
            continue
        angleCache = rect[2]  # 角度为x轴逆时针旋转，第一次接触到矩形边界时的值，范围：0~-90°，第一次接触的边界是宽！！！！！
        rectpoint = np.int0(rectpoint)
        # cv2.drawContours(a, [rectpoint], -1, (0, 255, 255,), 2)
        rectpoint[0][1] = rectpoint[0][1] + 280
        rectpoint[1][1] = rectpoint[1][1] + 280
        rectpoint[2][1] = rectpoint[2][1] + 280
        rectpoint[3][1] = rectpoint[3][1] + 280
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
            # for i in rectpoint:
            #     print(i)
            # print("rr", rectbox1y)
            G = int((heigthC + widthC) / 2)

            cv2.drawContours(src, [rectpoint], -1, (0, 255, 0,), 2)
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
                                                          dtype=np.double))  # 这些都是智能车的参数
            # cv2.imshow('imi', Q)
            print("距离", tvecs[2][0])
            dis.append(tvecs[2][0])
    print("距离均值", np.mean(dis))
    if np.mean(dis) <= 3000:
        J = Q
        cv2.imshow('deng', J)
        cv2.imwrite("deng.jpg", J)
    return J




dengdeng =  distance_erweima(src)



cv2.imshow('imgroi',dengdeng)
cv2.waitKey(0)
cv2.destroyAllWindows()


















