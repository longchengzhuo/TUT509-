import cv2
import time
import math
import numpy as np
import cProfile
import pstats
import matplotlib.pyplot as plt


X = []
AREA = []
KK = []
CALC = []
XX = 0
#嘉然带我走吧我不想写了！！！
def L(img):
    global XX
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    t3 = cv2.getTickCount()#---------------------------------------------------
    mode = "BLU"       # 填写敌方阵营的颜色，可以是 RED 和 BLUE
    shuzi_img = img.copy()
    img00 = img.copy()

    t20 = cv2.getTickCount()#---------------------------------------------------
    spendTime100 = (t20 - t3) * 1 / (cv2.getTickFrequency())
    print("100 time:", spendTime100)  # 0.011

    # 定义准星位置
    sightX = int((img.shape[1]) / 2)
    sightY = int((img.shape[0]) / 2)

    img00 = cv2.erode(img00, kernel)

    if mode == "BLUE":
        t12 = cv2.getTickCount()
        # 根据颜色筛选
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # HSV BLUE
        lower = np.array([100, 43, 46], dtype="uint8")  # [70, 0, 250]
        upper = np.array([124, 255, 255], dtype="uint8")  # [150, 80, 255]
        mask = cv2.inRange(hsv_image, lower, upper)

    else:
        # 根据颜色筛选
        img00 = cv2.resize(img00, (int(img00.shape[1] / 2), int(img00.shape[0] / 2)))
        t12 = cv2.getTickCount()#---------------------------------------------------

        spendTime1000 = (t12 - t20) * 1 / (cv2.getTickFrequency())
        print("111 time:", spendTime1000)  # 0.011

        hsv_image = cv2.cvtColor(img00, cv2.COLOR_BGR2HSV)
        # HSV 红色
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(hsv_image, lower_red, upper_red)
        # 区间2
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
        # 拼接两个区间
        mask = mask0 + mask1
        mask = cv2.resize(mask, (int(mask.shape[1] * 2), int(mask.shape[0] * 2)))
    t9 = cv2.getTickCount()#---------------------------------------------------
    spendTime5 = (t9 - t12) * 1 / (cv2.getTickFrequency())
    print("5 time:", spendTime5)#0.011

    img2 = mask
    # cv2.imwrite("mask0.jpg", img2)

    # img2 = cv2.dilate(img2, kernel)
    # cv2.imwrite("mask1.jpg", th2)
    img2 = cv2.erode(img2, kernel)
    # cv2.imwrite("mask2.jpg", img2)
    img2 = cv2.erode(img2, kernel)
    # # cv2.imwrite("mask3.jpg", img2)
    img2 = cv2.dilate(img2, kernel)
    # cv2.imwrite("mask4.jpg", img2)#可以再膨胀一次
    img2 = cv2.erode(img2, kernel)
    # img2 = cv2.erode(img2, kernel)
    # cv2.imwrite("mask5.jpg", img2)

    # img2 = cv2.resize(img2, (int(img2.shape[1] * 3), int(img2.shape[0] * 3)))
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # img2 = cv2.erode(img2, kernel1)
    binnary , contours, hierarchy = cv2.findContours(
        img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓


    t10 = cv2.getTickCount()#---------------------------------------------------
    spendTime5 = (t10 - t9) * 1 / (cv2.getTickFrequency())
    print("6 time:", spendTime5)#0.004

    n = 0
    x = []
    y = []
    rectbox1x = []
    rectbox2x = []
    rectbox3x = []
    rectbox4x = []
    rectbox1y = []
    rectbox2y = []
    rectbox3y = []
    rectbox4y = []
    longSide = []
    shortSide = []
    angle = []  # 初始化变量
    print("--------------------------")
    for contour in contours:
        rect = cv2.minAreaRect(contour)  # 获取最小包围矩形
        xCache, yCache = rect[0]  # 获取矩形的中心坐标
        rectpoint = cv2.boxPoints(rect)  #获取矩形四个角点
        widthC, heigthC = rect[1]  # 获取矩形的长宽
        if ((widthC==0) | (heigthC==0))==True:
            break
        angleCache = rect[2]  #  角度为x轴逆时针旋转，第一次接触到矩形边界时的值，范围：0~-90°，第一次接触的边界是宽！！！！！
        print("nmd",(int(xCache), int(yCache)))
        rectpoint = np.int0(rectpoint)
        cv2.drawContours(img, [rectpoint], -1, (0, 255, 0,), 2)


        if angleCache < -45:
            v = heigthC
            heigthC = widthC
            widthC = v



        print("·widthC:", widthC)
        print(" heigthC:", heigthC)
        print("angleCache", angleCache)
        print("heigthC / widthC", (heigthC / widthC))

        if 25 >= (heigthC / widthC) >= 2:  # 灯条是竖直放置，长宽比满足条件
            x.append(int(xCache))
            y.append(int(yCache))
            rectbox1x.append(int(rectpoint[0][0]))
            rectbox2x.append(int(rectpoint[1][0]))
            rectbox3x.append(int(rectpoint[2][0]))
            rectbox4x.append(int(rectpoint[3][0]))
            rectbox1y.append(int(rectpoint[0][1]))
            rectbox2y.append(int(rectpoint[1][1]))
            rectbox3y.append(int(rectpoint[2][1]))
            rectbox4y.append(int(rectpoint[3][1]))
            longSide.append(int(heigthC))
            shortSide.append(int(widthC))
            # shortSide.append(widthC)
            angle.append(angleCache)
            #print("widthC < heigthC")
            n = n + 1  # 有效矩形计数

    t4 = cv2.getTickCount()#---------------------------------------------------
    spendTime1 = (t4 - t10) * 1 / (cv2.getTickFrequency())
    spendTime11 = (t4 - t3) * 1 / (cv2.getTickFrequency())
    print("1 time:", spendTime1)
    print("1.1 time:", spendTime11)

    target = []  # 存储配对的两个灯条的编号 (L1, L2)

    rectbox = []

    LOCX = []
    LOCY = []
    locX = []
    locY = []  # 存储计算得到的中心点坐标
    dis = []  # 存储中心点与准星的距离
    pairNum = 0  # 初始化计数变量
    real_rectbox = []
    if n >= 2:  # 图像中找到两个以上的灯条
        for count in range(0, n):
            findCache = count + 1  # 初始化计数变量
            while findCache < n:  # 未超界，进行匹配运算

                calcCache = math.sqrt(
                    (x[findCache] - x[count]) ** 2 + (y[findCache] - y[count]) ** 2)  # 求中心点连线长

                calcCache = (2 * calcCache) / (longSide[count] + longSide[findCache])  # 求快捷计算单位
                area1 = longSide[count] * shortSide[count]
                area2 = longSide[findCache] * shortSide[findCache]

                if area1 != 0 and area2 != 0:
                    calc_area = area1 / area2
                    if calc_area < 1:
                        calc_area = 1 / calc_area
                    print("两个灯条角度差值：", abs(angle[count] - angle[findCache]), "大矩形长宽比值：", calcCache)
                    print("面积比：", calc_area)
                    if (1 <= calc_area < 5) and (2 < calcCache < 5) and (x[findCache] - x[count]) ** 2 > (
                            y[findCache] - y[count]) ** 2:  # 满足匹配条件
                        target.append((count, findCache))
                        locX.append(int((x[count] + x[findCache]) / 2))
                        locY.append(int((y[count] + y[findCache]) / 2))

                        # cv2.putText(img, cac, (locX[pairNum], locY[pairNum]), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)

                        AREA.append(calc_area)
                        CALC.append(calcCache)

                        # XX = XX + 1
                        # X.append(XX)
                        cv2.circle(
                            img, (locX[pairNum], locY[pairNum]), 3, (0, 0, 255), -1)
                        # 画两个圆来显示中心点的位置
                        cv2.circle(
                            img, (locX[pairNum], locY[pairNum]), 8, (0, 0, 255), 2)

                        pairNum = pairNum + 1  # 计数变量自增
                        break
                    findCache = findCache + 1
                else:
                    break
        print("pairNum", pairNum)
        if pairNum != 0:
            realnum = 0

            for count in range(0, pairNum):
                rectbox11 = []
                rectbox21 = []
                rectbox12 = []
                rectbox22 = []
                targetNum = count
                K = abs(y[target[targetNum][0]] - y[target[targetNum][1]]) / abs(
                    x[target[targetNum][0]] - x[target[targetNum][1]])
                print('K', K)
                XX = XX + 1
                X.append(XX)
                KK.append(K)
                if K < 0.4:

                    # 这里主要得到左边灯条矩形上边的一组x和y， 1为左上，2为右上，3为左下，4为右下
                    y1 = np.array(
                        [rectbox1y[target[targetNum][0]], rectbox2y[target[targetNum][0]],
                         rectbox3y[target[targetNum][0]],
                         rectbox4y[target[targetNum][0]]])
                    x1 = np.array(
                        [rectbox1x[target[targetNum][0]], rectbox2x[target[targetNum][0]],
                         rectbox3x[target[targetNum][0]],
                         rectbox4x[target[targetNum][0]]])
                    top11 = y1.argsort()[-1]
                    top12 = y1.argsort()[-2]
                    top13 = y1.argsort()[-3]
                    top14 = y1.argsort()[-4]

                    y2 = np.array(
                        [rectbox1y[target[targetNum][1]], rectbox2y[target[targetNum][1]],
                         rectbox3y[target[targetNum][1]],
                         rectbox4y[target[targetNum][1]]])
                    x2 = np.array(
                        [rectbox1x[target[targetNum][1]], rectbox2x[target[targetNum][1]],
                         rectbox3x[target[targetNum][1]],
                         rectbox4x[target[targetNum][1]]])
                    top21 = y2.argsort()[-1]
                    top22 = y2.argsort()[-2]
                    top23 = y2.argsort()[-3]
                    top24 = y2.argsort()[-4]

                    yy4 = np.array(
                        [y2[top21], y2[top22]])
                    yy2 = np.array(
                        [y2[top23], y2[top24]])
                    yy1 = np.array(
                        [y1[top13], y1[top14]])
                    yy3 = np.array(
                        [y1[top11], y1[top12]])

                    xx4 = np.array(
                        [x2[top21], x2[top22]])
                    xx2 = np.array(
                        [x2[top23], x2[top24]])
                    xx1 = np.array(
                        [x1[top13], x1[top14]])
                    xx3 = np.array(
                        [x1[top11], x1[top12]])

                    r111 = xx1.argsort()[-1]  # -1代表返回最大值索引
                    r112 = xx1.argsort()[-2]
                    r311 = xx3.argsort()[-1]
                    r312 = xx3.argsort()[-2]
                    r221 = xx2.argsort()[-1]
                    r222 = xx2.argsort()[-2]
                    r421 = xx4.argsort()[-1]
                    r422 = xx4.argsort()[-2]

                    if xx1[r111] > xx2[r222]:
                        r111 = r112
                        r222 = r221
                        r311 = r312
                        r422 = r421

                    # 小矩形
                    zs = (xx1[r111], yy1[r111])
                    ys = (xx2[r222], yy2[r222])
                    zx = (xx3[r311], yy3[r311])
                    yx = (xx4[r422], yy4[r422])

                    ts_zs = [xx1[r111], yy1[r111]]
                    ts_ys = [xx2[r222], yy2[r222]]
                    ts_zx = [xx3[r311], yy3[r311]]
                    ts_yx = [xx4[r422], yy4[r422]]

                    if ts_zs[0] > ts_ys[0]:  # 始终保持z是左，y是右
                        h_zs = ts_zs
                        h_zx = ts_zx
                        ts_zs = ts_ys
                        ts_zx = ts_yx
                        ts_ys = h_zs
                        ts_yx = h_zx

                    ts_box = [ts_zs, ts_ys, ts_yx, ts_zx]  # 透视变换的坐标box

                    AA = int((xx3[r311] + xx2[r222]) / 2)
                    BB = int((yy1[r111] + yy4[r422]) / 2)  # AA,BB为矩形中间点坐标

                    tf_img = toushibianhuan(shuzi_img, ts_box)
                    if Pipei(tf_img, jiugongge, AA, BB):

                        # 画蓝色小矩形
                        cv2.line(img, zs, ys, (255, 0, 0), 3)
                        cv2.line(img, ys, yx, (255, 0, 0), 3)
                        cv2.line(img, yx, zx, (255, 0, 0), 3)
                        cv2.line(img, zx, zs, (255, 0, 0), 3)

                        rectbox11.append([int((x1[top11] + x1[top12]) / 2),
                                          int((y1[top11] + y1[top12]) / 2)])
                        rectbox12.append([int((x1[top13] + x1[top14]) / 2),
                                          int((y1[top13] + y1[top14]) / 2)])
                        rectbox21.append([int((x2[top21] + x2[top22]) / 2),
                                          int((y2[top21] + y2[top22]) / 2)])
                        rectbox22.append([int((x2[top23] + x2[top24]) / 2),
                                          int((y2[top23] + y2[top24]) / 2)])

                        if int((rectbox21[0][0] + rectbox22[0][0]) / 2) >= int(
                                (rectbox11[0][0] + rectbox12[0][0]) / 2):  # 始终保持2在左
                            a = rectbox11
                            b = rectbox12
                            rectbox11 = rectbox21
                            rectbox12 = rectbox22
                            rectbox21 = a
                            rectbox22 = b

                        box = rectbox22 + rectbox12 + rectbox21 + rectbox11
                        rectbox.append(box)
                        LOCX.append(locX[count])
                        LOCY.append(locY[count])
                        dis.append(
                            math.sqrt((locX[count] - sightX) ** 2 + (locY[count] - sightY) ** 2))

                        realnum = realnum + 1

                else:
                    pass

            if realnum != 0:
                disCalcCache = dis[0]
                real_targetNum = 0  # 存储距离准星最进的装甲板编号
                for count in range(0, realnum):
                    if dis[count] < disCalcCache:
                        real_targetNum = count
                        disCalcCache = dis[count]

                real_rectbox = rectbox[real_targetNum]
                print("real_targetNum", real_targetNum)
                cv2.line(img, (LOCX[real_targetNum], LOCY[real_targetNum]),
                         (sightX, sightY), (255, 0, 255), 2)  # 画指向线
                print("(LOCX, LOCY)", (LOCX, LOCY))
                print("rectbox", rectbox)


    t5 = cv2.getTickCount()
    spendTime2 = (t5 - t4) * 1 / (cv2.getTickFrequency())
    print("2 time:", spendTime2)
    global  lost_count
    global  aim_count
    if real_rectbox != []:

        print("rectbox", real_rectbox)


        if aim_count == 0:
            kf.statePost = np.array([0, 0, 0, 0, 0, 0], np.float32)
        if True:
            aim_count = aim_count + 1
            lost_count = 0
            img_points = np.array(real_rectbox, dtype=np.double)
            obj_points = [[-67., -25., 0.],
                          [67., -25., 0.],
                          [-67., 25., 0.],
                          [67., 25., 0.]]
            obj_points = np.reshape(obj_points, (4, 3))

            success, rvecs, tvecs = cv2.solvePnP(obj_points, img_points,
                                                 np.array([[1.34386883e+03, 0., 1.02702867e+03],
                                                           [0., 1.38920787e+03, 5.62395968e+02],
                                                           [0., 0., 1.]], dtype=np.double),
                                                 np.array([[-0.13939675, 0.42409417, -0.00454986, 0.01027033, -0.61637364]],
                                                          dtype=np.double))#这些都是大恒长炮的参数


            tvecs = np.array(tvecs)
            tvecs1 = [(tvecs[0][0] , tvecs[1][0] ,tvecs[2][0])]

            kal(tvecs1)
            print("tvecs", tvecs1)
            # theta_x, theta_y = rotateMatrixToEulerAngles2(tvecs)
            # ser.write(("$%4.3f, %4.3f@" %(theta_x, theta_y)).encode())


    elif lost_count <= 15:
        aim_count = 0
        lost_count = lost_count + 1
        x = kf.statePost[0] + kf.statePost[3]
        y = kf.statePost[1] + kf.statePost[4]
        z = kf.statePost[2] + kf.statePost[5]
        coor = [(x, y, z)]
        kal(coor)


    else:
        kf.statePost = np.array([0, 0, 0, 0, 0, 0], np.float32)
        lost_count = 0


    t6 = cv2.getTickCount()
    spendTime3 = (t6 - t5) * 1 / (cv2.getTickFrequency())
    print("3 time:", spendTime3)

    # 画准星
    cv2.line(img, (sightX, sightY - 4),
             (sightX, sightY - 9), (50, 250, 100), 2)  # 上
    cv2.line(img, (sightX, sightY + 4),
             (sightX, sightY + 9), (50, 250, 100), 2)  # 下
    cv2.line(img, (sightX - 4, sightY),
             (sightX - 9, sightY), (50, 250, 100), 2)  # 左
    cv2.line(img, (sightX + 4, sightY),
             (sightX + 9, sightY), (50, 250, 100), 2)  # 右

    print("--------------------------")
    return img

def rotateMatrixToEulerAngles2(RM):
    theta_x = np.arctan2(RM[0, 0], RM[2, 0]) / np.pi * 180
    theta_y = np.arctan2(RM[1, 0], np.sqrt(RM[0, 0] * RM[0, 0] + RM[2, 0] * RM[2, 0])) / np.pi * 180
    print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}")
    return theta_x, theta_y

def Pipei(img11, jiugongge, AA, BB):
    img11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img1111 = cv2.erode(img11, kernel)
    img11 = cv2.dilate(img1111, kernel)
    ret2, temple = cv2.threshold(img11, 0, 255, cv2.THRESH_OTSU)
    temple = cv2.cvtColor(temple, cv2.COLOR_GRAY2BGR)
    R = 0
    n = 0
    y_sum = 0
    x_sum = 0

    temple = cv2.resize(temple, (int(temple.shape[1] / 4), int(temple.shape[0] / 4)))

    jiugongge = cv2.resize(jiugongge, (int(jiugongge.shape[1] / 4), int(jiugongge.shape[0] / 4)))
    h, w, c = temple.shape

    results = cv2.matchTemplate(jiugongge, temple, cv2.TM_CCOEFF_NORMED)  # 按照标准相关系数匹配
    for y in range(len(results)):  # 遍历结果数组的行
        for x in range(len(results[y])):  # 遍历结果数组的列
            # print("老子把你吊起打", results[y][x])
            if results[y][x] > 0.85:  # 如果相关系数大于0.99则认为匹配成功
                print("老子把你吊起打", results[y][x])
                R = R + results[y][x]
                n = n + 1
                y_sum = y_sum + y
                x_sum = x_sum + x
    if n != 0:
        if R / n >= 0.85:
            # cv2.putText(img, "Number Recognized:", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            print("R", R / n)
            print("我真的服", int(y_sum / n + h / 2), int(x_sum / n + w / 2))
            if 0 < int(y_sum / n + h / 2) < 50/4 and 0 < int(x_sum / n + w / 2) < 60/4:
                cv2.putText(img, "4", (AA-20, BB+20), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)
                # cv2.imwrite("temmm4.jpg", temple)
                print("4")
            elif 0 < int(y_sum / n + h / 2) < 50/4 and 60/4 < int(x_sum / n + w / 2) < 140/4:
                cv2.putText(img, "1", (AA-20, BB+20), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)
                # cv2.imwrite("temmm1.jpg", temple)
                print("1")
            elif 0 < int(y_sum / n + h / 2) < 50/4 and 140/4 < int(x_sum / n + w / 2) < 200/4:
                cv2.putText(img, "5", (AA-20, BB+20), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)
                print("5")
            elif 0 < int(y_sum / n + h / 2) < 50/4 and 200/4 < int(x_sum / n + w / 2) < 255/4:
                cv2.putText(img, "3", (AA-20, BB+20), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)
                print("3")

            return True

def toushibianhuan(img, ts_box):

    wid, hei = 50, 25
    pts1 = np.float32(ts_box)
    pts2 = np.float32([[0, 0], [wid, 0], [wid, hei], [0, hei]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (wid, hei))

    return warped



def kal(positions):

    pt = positions[0]


    print("pt", pt)
    predicted = predict(pt[0], pt[1], pt[2])

    coordinate = [[predicted[0]],[predicted[1]],[predicted[2]]]

    camera_coordinate = coordinate

    Zc = coordinate[2][0]
    if Zc != 0:
        # 相机坐标系转图像坐标系 (Xc,Yc,Zc) --> (x, y)  下边的f改为焦距
        focal_length = np.mat([
            [1.34386883e+03, 0., 1.02702867e+03],
            [0., 1.38920787e+03, 5.62395968e+02],
            [0., 0., 1.]
        ])
        image_coordinate = (focal_length * camera_coordinate) / Zc
        data = np.array(image_coordinate)
        print("image_coordinate", image_coordinate)
        if np.isnan(data).sum() == 3 :
            # image_coordinate = [[960][600][1]]
            # print("0", image_coordinate.any())
            pass
        else:
            image_coordinat = np.array(image_coordinate).reshape(-1)
            print(f'图像坐标为：\n{(int(image_coordinat[0]), int(image_coordinat[1]))}')
            # print(np.array(image_coordinate[0][0]))
            print(image_coordinat[1])
            cv2.circle(
                img, (int(image_coordinat[0]), int(image_coordinat[1])), 20, (0, 0, 255), 5)

lost_count = 0
aim_count = 0
kf = cv2.KalmanFilter(6, 3)
kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],  [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]], np.float32)
kf.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)*10
# kf.statePost = np.array([pt[0], pt[1], pt[2], 0, 0, 0], np.float32)
def predict(coordX, coordY, coordZ):
    ''' This function estimates the position of the object'''
    measured = np.array([[np.float32(coordX)], [np.float32(coordY)], [np.float32(coordZ)]])
    predicted = kf.predict()
    kf.correct(measured)
    T = 1 + coordZ/16.5/21.5

    print("T", T)
    x, y, z = int(kf.statePost[0] + T*kf.statePost[3]), int(kf.statePost[1] + T*kf.statePost[4]), int(kf.statePost[2] + T*kf.statePost[5])
    return x, y, z





cap = cv2.VideoCapture('D:/biancheng\opencv python/robotmaster/result.avi')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("width:",width, "height:", height)

out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920, 1200))
jiugongge = cv2.imread("D:/biancheng\opencv python\camera jiaozheng\moban.jpg")

while(cap.isOpened()):
    # kf = KalmanFilter()
    ret, frame = cap.read()
    print(frame.shape)
    img = frame

    t1 = cv2.getTickCount()
    img = L(img)
    t2 = cv2.getTickCount()
    spendTime = (t2 - t1) * 1 / (cv2.getTickFrequency())
    print("total time:", spendTime)
    print("---------------------------------------------------------------------")

    FPS = 1 / spendTime
    FPS = 'The fps is %d' %(FPS)
    img = cv2.resize(img, (960, 600), interpolation=cv2.INTER_AREA)

    cv2.putText(img, FPS, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)

    cv2.imshow('frame', img)
    img = cv2.resize(img, (1920, 1200), interpolation=cv2.INTER_AREA)
    out.write(img)
    print(aim_count)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# y1 = AREA
y1 = KK
plt.plot(X,y1,'o')
plt.show()
plt.savefig('sin_cos.png')


cap.release()



cv2.destroyAllWindows()

