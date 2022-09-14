import cv2
import time
import math
import numpy as np
import cProfile
import pstats


# from config import *
#嘉然带我走吧我不想写了！！！
def L(img):
    t3 = cv2.getTickCount()
    mode = "BLU"       # 填写敌方阵营的颜色，可以是 RED 和 BLUE
    debug = False       # 一键开启调试模式
    shuzi_img = img.copy()

    # 定义准星位置
    sightX = int((img.shape[1]) / 2)
    sightY = int((img.shape[0]) / 2)

    if mode == "BLUE":

        # 根据颜色筛选
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # HSV BLUE
        lower = np.array([100, 43, 46], dtype="uint8")  # [70, 0, 250]
        upper = np.array([124, 255, 255], dtype="uint8")  # [150, 80, 255]
        mask = cv2.inRange(hsv_image, lower, upper)

    else:
        # 根据颜色筛选

        t12 = cv2.getTickCount()

        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
    t9 = cv2.getTickCount()
    spendTime5 = (t9 - t12) * 1 / (cv2.getTickFrequency())
    print("5 time:", spendTime5)#0.011

    img2 = mask
    # cv2.imwrite("mask0.jpg", img2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th2 = cv2.dilate(img2, kernel)
    # cv2.imwrite("mask1.jpg", th2)
    img2 = cv2.erode(th2, kernel)
    # cv2.imwrite("mask2.jpg", img2)
    img2 = cv2.erode(img2, kernel)
    # cv2.imwrite("mask3.jpg", img2)
    img2 = cv2.dilate(img2, kernel)
    # cv2.imwrite("mask4.jpg", img2)#可以再膨胀一次
    img2 = cv2.erode(img2, kernel)
    # cv2.imwrite("mask5.jpg", img2)
    binnary , contours, hierarchy = cv2.findContours(
        img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓

    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    t10 = cv2.getTickCount()
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
        # cv2.putText(img, heigthC,(int(xCache), int(yCache)), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        # cv2.putText(img, "1,13", (x1[top13], y1[top13]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),


        if angleCache < -45:
            v = heigthC
            heigthC = widthC
            widthC = v



        print("·widthC:", widthC)
        print(" heigthC:", heigthC)
        print("angleCache", angleCache)
        print("heigthC / widthC", (heigthC / widthC))

        if 15 >= (heigthC / widthC) >= 2.3:  # 灯条是竖直放置，长宽比满足条件
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
            # shortSide.append(widthC)
            angle.append(angleCache)
            #print("widthC < heigthC")
            n = n + 1  # 有效矩形计数
        # else:
        #     #print("widthC / heigthC ", (widthC / heigthC))
        #     if 4 >= (widthC / heigthC) >= 2.3:  # 长宽比满足条件
        #         x.append(int(xCache))
        #         y.append(int(yCache))
        #         rectbox1x.append(int(rectpoint[0][0]))
        #         rectbox2x.append(int(rectpoint[1][0]))
        #         rectbox3x.append(int(rectpoint[2][0]))
        #         rectbox4x.append(int(rectpoint[3][0]))
        #         rectbox1y.append(int(rectpoint[0][1]))
        #         rectbox2y.append(int(rectpoint[1][1]))
        #         rectbox3y.append(int(rectpoint[2][1]))
        #         rectbox4y.append(int(rectpoint[3][1]))
        #         longSide.append(int(widthC))
        #         # shortSide.append(heigthC)
        #         angle.append(angleCache + 90)
        #         #print("widthC > heigthC")
        #         n = n + 1  # 有效矩形计数
    #print("有效矩形个数为", n)
    t4 = cv2.getTickCount()
    spendTime1 = (t4 - t3) * 1 / (cv2.getTickFrequency())
    print("1 time:", spendTime1)
    target = []  # 存储配对的两个灯条的编号 (L1, L2)
    rectbox11 = []
    rectbox21 = []
    rectbox12 = []
    rectbox22 = []
    locX = []
    locY = []  # 存储计算得到的中心点坐标
    dis = []  # 存储中心点与准星的距离
    pairNum = 0  # 初始化计数变量
    if n >= 2:  # 图像中找到两个以上的灯条
        for count in range(0, n):
            findCache = count + 1  # 初始化计数变量
            while findCache < n:  # 未超界，进行匹配运算
                calcCache = math.sqrt(
                    (x[findCache] - x[count]) ** 2 + (y[findCache] - y[count]) ** 2)  # 求中心点连线长

                calcCache = (2*calcCache) / (longSide[count] + longSide[findCache])  # 求快捷计算单位

                print("两个灯条角度差值：", abs(angle[count] - angle[findCache]), "大矩形长宽比值：", calcCache)

                if (2 < calcCache < 3.5) and (x[findCache] - x[count]) ** 2 > (y[findCache] - y[count]) ** 2:  # 满足匹配条件
                    target.append((count, findCache))
                    #loc.append((int((x[count] + x[findCache]) / 2), int((y[count] + y[findCache]) / 2)))
                    locX.append(int((x[count] + x[findCache]) / 2))
                    locY.append(int((y[count] + y[findCache]) / 2))
                    dis.append(
                        math.sqrt((locX[pairNum] - sightX) ** 2 + (locY[pairNum] - sightY) ** 2))


                    cv2.circle(
                        img, (locX[pairNum], locY[pairNum]), 3, (0, 0, 255), -1)
                    # 画两个圆来显示中心点的位置
                    cv2.circle(
                        img, (locX[pairNum], locY[pairNum]), 8, (0, 0, 255), 2)
                    #print("··Group", pairNum, ":")


                    pairNum = pairNum + 1  # 计数变量自增
                    break
                findCache = findCache + 1
        #print("中心点坐标", locX,locY)

        # ···在此处添加数字验证真假装甲板代码 添加你妈

        if pairNum != 0:
            # 寻找离准星最近的装甲板
            disCalcCache = dis[0]
            targetNum = 0  # 存储距离准星做进的装甲板编号
            for count in range(0, pairNum):
                if dis[count] < disCalcCache:
                    targetNum = count
                    disCalcCache = dis[count]


            cv2.line(img, (locX[targetNum], 0),
                     (locX[targetNum], 480), (0, 255, 0), 1)  # 画竖线
            cv2.line(img, (0, locY[targetNum]),
                     (720, locY[targetNum]), (0, 255, 0), 1)  # 画横线
            cv2.line(img, (x[target[targetNum][0]], y[target[targetNum][0]]), (
                x[target[targetNum][1]], y[target[targetNum][1]]), (255, 255, 255), 2)  # 画连接线
            cv2.line(img, (locX[targetNum], locY[targetNum]),
                     (sightX, sightY), (255, 0, 255), 2)  # 画指向线

            y1 = np.array(
                [rectbox1y[target[targetNum][0]], rectbox2y[target[targetNum][0]], rectbox3y[target[targetNum][0]],
                 rectbox4y[target[targetNum][0]]])
            x1 = np.array(
                [rectbox1x[target[targetNum][0]], rectbox2x[target[targetNum][0]], rectbox3x[target[targetNum][0]],
                 rectbox4x[target[targetNum][0]]])
            top11 = y1.argsort()[-1]
            top12 = y1.argsort()[-2]
            top13 = y1.argsort()[-3]
            top14 = y1.argsort()[-4]

            y2 = np.array(
                [rectbox1y[target[targetNum][1]], rectbox2y[target[targetNum][1]], rectbox3y[target[targetNum][1]],
                 rectbox4y[target[targetNum][1]]])
            x2 = np.array(
                [rectbox1x[target[targetNum][1]], rectbox2x[target[targetNum][1]], rectbox3x[target[targetNum][1]],
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

            r111 = xx1.argsort()[-1]
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

            ts_box = [ts_zs,ts_ys,ts_yx, ts_zx]
            # pts = np.array([zs,ys,zx,yx], np.int32)
            # pts = pts.reshape((-1, 1, 2))
            # cv2.polylines(img, pts=pts, isClosed=True, color=(255, 255, 255), thickness=20)
            cv2.line(img, zs, ys, (255, 0, 0), 3)
            cv2.line(img, ys, yx, (255, 0, 0), 3)
            cv2.line(img, yx, zx, (255, 0, 0), 3)
            cv2.line(img, zx, zs, (255, 0, 0), 3)
            print("1234",zs,ys,zx,yx)
            AA = int((xx3[r311] + xx2[r222])/2)
            BB = int((yy1[r111] + yy4[r422])/2)
            cv2.circle(
                img, (int((x1[top11] + x1[top12]) / 2),
                      int((y1[top11] + y1[top12]) / 2)), 5, (255, 255, 100), 1)
            cv2.circle(
                img, (int((x1[top13] + x1[top14]) / 2),
                      int((y1[top13] + y1[top14]) / 2)), 5, (255, 100, 255), 1)

            rectbox11.append([int((x1[top11] + x1[top12]) / 2),
                              int((y1[top11] + y1[top12]) / 2)])
            rectbox12.append([int((x1[top13] + x1[top14]) / 2),
                              int((y1[top13] + y1[top14]) / 2)])

            cv2.circle(
                img, (int((x2[top21] + x2[top22]) / 2),
                      int((y2[top21] + y2[top22]) / 2)), 5, (255, 255, 100), 1)
            cv2.circle(
                img, (int((x2[top23] + x2[top24]) / 2),
                      int((y2[top23] + y2[top24]) / 2)), 5, (255, 100, 255), 1)

            rectbox21.append([int((x2[top21] + x2[top22]) / 2),
                              int((y2[top21] + y2[top22]) / 2)])
            rectbox22.append([int((x2[top23] + x2[top24]) / 2),
                              int((y2[top23] + y2[top24]) / 2)])

            cv2.putText(img, "11", (x1[top11], y1[top11]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                        1)
            cv2.putText(img, "12", (x1[top12], y1[top12]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                        1)
            cv2.putText(img, "13", (x1[top13], y1[top13]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                        1)
            cv2.putText(img, "14", (x1[top14], y1[top14]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                        1)

            cv2.putText(img, "21", (x2[top21], y2[top21]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                        1)
            cv2.putText(img, "22", (x2[top22], y2[top22]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                        1)
            cv2.putText(img, "23", (x2[top23], y2[top23]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                        1)
            cv2.putText(img, "24", (x2[top24], y2[top24]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                        1)
            print("23",(x2[top23], y2[top23]))
            print("14",(x1[top14], y1[top14]))
            print("21", (x2[top21], y2[top21]))
            print("12",(x1[top12], y1[top12]))

            print("第一个矩阵：","x1[top11]", x1[top11], "x1[top12]", x1[top12], "\ny1[top11]", y1[top11], "y1[top12]", y1[top12], "\nx1[top13]", x1[top13], "x1[top14]", x1[top14], "\ny1[top13]", y1[top13], "y1[top14]", y1[top14])
            print("第一个矩阵：", "x2[top11]", x2[top11], "x2[top12]", x2[top12], "\ny2[top11]", y2[top11], "y2[top12]",
                  y2[top12], "\nx2[top13]", x2[top13], "x2[top14]", x2[top14], "\ny2[top13]", y2[top13], "y2[top14]",
                  y2[top14])
            if int((rectbox21[0][0] + rectbox22[0][0]) / 2) >= int((rectbox11[0][0] + rectbox12[0][0]) / 2):
                a = rectbox11
                b = rectbox12
                rectbox11 = rectbox21
                rectbox12 = rectbox22
                rectbox21 = a
                rectbox22 = b

    rectbox = rectbox22 + rectbox12 + rectbox21 + rectbox11
    t5 = cv2.getTickCount()
    spendTime2 = (t5 - t4) * 1 / (cv2.getTickFrequency())
    print("2 time:", spendTime2)
    if rectbox != []:
        print("rectbox", rectbox)
        tf_img = toushibianhuan(shuzi_img, ts_box)
        # print(Pipei(tf_img))
        if Pipei(tf_img, jiugongge):
        # if True:
            # cv2.putText(img, "yesyesyesyes", (500, 500), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200),5)
            # cv2.putText(img, "11", (rectbox11[0][0], rectbox11[0][1]), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            # cv2.putText(img, "12", (rectbox12[0][0], rectbox12[0][1]), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            # cv2.putText(img, "21", (rectbox21[0][0], rectbox21[0][1]), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            # cv2.putText(img, "22", (rectbox22[0][0], rectbox22[0][1]), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            img_points = np.array(rectbox, dtype=np.double)
            obj_points = [[-62., -30., 0.],
                          [62., -30., 0.],
                          [-62.,30., 0.],
                          [62., 30., 0.]]
            obj_points = np.reshape(obj_points, (4, 3))

            success, rvecs, tvecs = cv2.solvePnP(obj_points, img_points,
                                                 np.array([[1.60954972e+03, 0., 6.36716902e+02],
                                                           [0., 1.61293941e+03, 5.27210575e+02],
                                                           [0., 0., 1.]], dtype=np.double),
                                                 np.array([[-0.08045919, 0.28519144, 0.00188052, -0.00196669, -0.07837469]],
                                                          dtype=np.double))#这些都是大恒长炮的参数

            print("tvecs", tvecs)
            tvecs = np.array(tvecs)
            theta_x, theta_y = rotateMatrixToEulerAngles2(tvecs)
            #ser.write(("$%4.3f, %4.3f@" %(theta_x, theta_y)).encode())
    else:
        cv2.imwrite("maskmm.jpg", img)
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


def Pipei(img11, jiugongge,):
    # cv2.imwrite("templepp.png", img11)
    img11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img11 = cv2.erode(img11, kernel)
    img11 = cv2.dilate(img11, kernel)
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
            if results[y][x] > 0.8:  # 如果相关系数大于0.99则认为匹配成功
                print("老子把你吊起打", results[y][x])
                R = R + results[y][x]
                n = n + 1
                y_sum = y_sum + y
                x_sum = x_sum + x
                # cv2.rectangle(jiugongge, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # print("老子把你吊起打")
    # print("R", R / n)
    # if R > n * 0.75:
    if n != 0:
        print("R", R / n)
        print("我真的服", int(y_sum / n + h / 2), int(x_sum / n + w / 2))
        if 0 < int(y_sum / n + h / 2) < 50/4 and 0 < int(x_sum / n + w / 2) < 60/4:
            cv2.putText(img, "Number Recognized:", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            cv2.putText(img, "4", (1000, 450), cv2.FONT_HERSHEY_COMPLEX, 5.0, (0, 0, 255), 7)
            print("4")
        elif 0 < int(y_sum / n + h / 2) < 50/4 and 60/4 < int(x_sum / n + w / 2) < 140/4:
            cv2.putText(img, "1", (250, 250), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            print("1")
            # cv2.imwrite("why1.png", jiugongge)
            # cv2.imwrite("whyy1.png", temple)
        elif 0 < int(y_sum / n + h / 2) < 50/4 and 140/4 < int(x_sum / n + w / 2) < 200/4:
            cv2.putText(img, "5", (250, 250), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
            print("5")
        elif 0 < int(y_sum / n + h / 2) < 50/4 and 200/4 < int(x_sum / n + w / 2) < 255/4:
            cv2.putText(img, "3", (250, 250), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
            print("3")

        return True


def toushibianhuan(img, ts_box):
    # pts1 = np.float32(ts_box)
    # pts2 = np.float32([[0, 0], [100, 0], [100, 100], [0, 100] ])
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # warped = cv2.warpPerspective(img, M, (100, 100))
    wid, hei = 50, 25
    pts1 = np.float32(ts_box)
    pts2 = np.float32([[0, 0], [wid, 0], [wid, hei], [0, hei]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (wid, hei))
    # cv2.imwrite('D:/biancheng\opencv python/robotmaster\PythonApplication1/nice/bad.jpg', warped)

    return warped



cap = cv2.VideoCapture('D:/biancheng\opencv python/robotmaster/1.avi')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
#
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print("width:",width, "height:", height)

# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))
jiugongge = cv2.imread("D:/biancheng\opencv python\camera jiaozheng\moban.jpg")
# jiugongge = cv2.resize(jiugongge,(int(jiugongge.shape[1]/2),int(jiugongge.shape[0]/2)))# 读取原始图像
# jiugongge = cv2.resize(jiugongge,(int(jiugongge.shape[1]/4),int(jiugongge.shape[0]/4)))
# def mainmain():
#     t1 = cv2.getTickCount()
#     frame = cv2.imread("D:/biancheng\opencv python/robotmaster\PythonApplication1\maskmm.jpg")
#     print(frame.shape)
#     img = frame
#     # img = frame
#     img = L(img)
#     t2 = cv2.getTickCount()
#     spendTime = (t2 - t1) * 1 / (cv2.getTickFrequency())
#     FPS = 1 / spendTime
#     FPS = 'The fps is %d' % (FPS)
#     img = cv2.resize(img, (960, 600), interpolation=cv2.INTER_AREA)
#
#     cv2.putText(img, FPS, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
#     # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imshow('frame', img)
#     img = cv2.resize(img, (1920, 1200), interpolation=cv2.INTER_AREA)
#     # out.write(img)
#     cv2.imwrite("heihie.jpg", jiugongge)
# mainmain()
while(cap.isOpened()):

    ret, frame = cap.read()
    print(frame.shape)
    img = frame
    #img = frame
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
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', img)
    img = cv2.resize(img, (1920, 1200), interpolation=cv2.INTER_AREA)
    # out.write(img)
    cv2.imwrite("heihie.jpg", jiugongge)
    # mainmain()
    # cProfile.run("mainmain()", filename="result.out", sort="cumulative")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# cProfile.run("mainmain()", filename="result.out", sort="cumulative")

cv2.destroyAllWindows()

