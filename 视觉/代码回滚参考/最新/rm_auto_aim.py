from config import *


# 嘉然带我走吧我不想写了！！！
def L(img):
    global lost_count
    global aim_count
    global last_locx
    global last_locy
    global last_width
    global last_height
    global XX
    real_rectbox = []
    n         = 0
    x         = []
    y         = []
    dis       = []  # 存储中心点与准星的距离
    LOCX      = []
    LOCY      = []
    locX      = []
    locY      = []  # 存储计算得到的中心点坐标
    angle     = []  # 初始化变量
    WIDTH     = []
    HEIGHT    = []
    target    = []  # 存储配对的两个灯条的编号 (L1, L2)
    rectbox   = []
    pairNum   = 0  # 初始化计数变量
    longSide  = []
    rectbox1x = []
    rectbox2x = []
    rectbox3x = []
    rectbox4x = []
    rectbox1y = []
    rectbox2y = []
    rectbox3y = []
    rectbox4y = []
    shortSide = []


    WarpPerspect_img = img.copy()
    preprocess_Img = img.copy()



    # 定义准星位置
    sightX = int((img.shape[1]) / 2)
    sightY = int((img.shape[0]) / 2)

    preprocess_Img = cv2.erode(preprocess_Img, kernel)

    if mode == "BLUE":
        # 根据颜色筛选
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # HSV BLUE
        lower = np.array([100, 43, 46], dtype="uint8")  # [70, 0, 250]
        upper = np.array([124, 255, 255], dtype="uint8")  # [150, 80, 255]
        mask = cv2.inRange(hsv_image, lower, upper)

    else:
        # 根据颜色筛选
        preprocess_Img = cv2.resize(preprocess_Img, (int(preprocess_Img.shape[1] / 2), int(preprocess_Img.shape[0] / 2)))
        hsv_image = cv2.cvtColor(preprocess_Img, cv2.COLOR_BGR2HSV)
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


    img2 = mask

    img2 = cv2.erode(img2, kernel)
    img2 = cv2.erode(img2, kernel)
    img2 = cv2.dilate(img2, kernel)
    img2 = cv2.erode(img2, kernel)

    binnary, contours, hierarchy = cv2.findContours(
        img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓



    print("--------------------------")
    for contour in contours:
        rect = cv2.minAreaRect(contour)  # 获取最小包围矩形
        xCache, yCache = rect[0]  # 获取矩形的中心坐标
        rectpoint = cv2.boxPoints(rect)  # 获取矩形四个角点
        widthC, heigthC = rect[1]  # 获取矩形的长宽
        if ((widthC == 0) | (heigthC == 0)) == True:
            break
        angleCache = rect[2]  # 角度为x轴逆时针旋转，第一次接触到矩形边界时的值，范围：0~-90°，第一次接触的边界是宽！！！！！
        print("nmd", (int(xCache), int(yCache)))
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
            # print("widthC < heigthC")
            n = n + 1  # 有效矩形计数



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
                    if (1 <= calc_area < 10) and (0.6 < calcCache < 5) and (x[findCache] - x[count]) ** 2 > (
                            y[findCache] - y[count]) ** 2:  # 满足匹配条件
                        target.append((count, findCache))
                        locX.append(int((x[count] + x[findCache]) / 2))
                        locY.append(int((y[count] + y[findCache]) / 2))

                        # cv2.putText(img, cac, (locX[pairNum], locY[pairNum]), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)
                        if debug and mathod == 1:
                            CALC.append(calcCache)
                            XX = XX + 1
                            X.append(XX)

                        elif debug and mathod == 2:
                            CALC.append(calc_area)
                            XX = XX + 1
                            X.append(XX)
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
                    x[target[targetNum][0]] - x[target[targetNum][1]])  # 两灯条中心连线斜率
                print('K', K)
                if debug and mathod == 3:
                    CALC.append(K)
                    XX = XX + 1
                    X.append(XX)

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

                    rect_Center_X = int((xx3[r311] + xx2[r222]) / 2)
                    rect_Center_Y = int((yy1[r111] + yy4[r422]) / 2)  # rect_Center_X,rect_Center_Y为矩形中间点坐标

                    tf_img = WarpPerspect(WarpPerspect_img, ts_box)
                    if Match(tf_img, number_img, rect_Center_X, rect_Center_Y, R, N, x_sum, y_sum, img):

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
                        WIDTH.append(np.sqrt((ts_zx[0] - ts_ys[0]) ** 2 + (ts_zx[1] - ts_ys[1]) ** 2))
                        HEIGHT.append(np.sqrt((ts_ys[0] - ts_yx[0]) ** 2 + (ts_ys[1] - ts_yx[1]) ** 2))
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
                cv2.line(img, (LOCX[real_targetNum], LOCY[real_targetNum]),
                         (sightX, sightY), (255, 0, 255), 2)  # 画指向线
                current_width = WIDTH[real_targetNum]
                current_height = HEIGHT[real_targetNum]


                print("real_targetNum", real_targetNum)
                print("(LOCX, LOCY)", (LOCX, LOCY))
                print("rectbox", rectbox)

    if real_rectbox != []:

        if ((last_locx - LOCX[real_targetNum]) ** 2 + (last_locy - LOCY[real_targetNum]) ** 2) >= (
                (last_width + last_height) ** 2) / 4:
            kf.statePost = np.array([0, 0, 0, 0, 0, 0], np.float32)
        last_locx = LOCX[real_targetNum]
        last_locy = LOCY[real_targetNum]
        last_width = current_width
        last_height = current_height

        if aim_count == 0:
            kf.statePost = np.array([0, 0, 0, 0, 0, 0], np.float32)

        aim_count = aim_count + 1
        lost_count = 0
        img_points = np.array(real_rectbox, dtype=np.double)
        obj_points = [[-67., -25., 0.],
                      [67., -25., 0.],
                      [-67., 25., 0.],
                      [67., 25., 0.]]
        obj_points = np.reshape(obj_points, (4, 3))

        success, rvecs, tvecs = cv2.solvePnP(obj_points, img_points,
                                             MDV_OldCamera_intrinsic["mtx"],
                                             MDV_OldCamera_intrinsic["dist"])

        tvecs = np.array(tvecs)
        positions = [(tvecs[0][0], tvecs[1][0], tvecs[2][0])]
        predicted_X, predicted_Y, predicted_Z = Kal_predict(positions[0][0], positions[0][1], positions[0][2])
        Kal_Visualization(predicted_X, predicted_Y, predicted_Z, img)
        print("tvecs", positions)
        # theta_x, theta_y = rotateMatrixToEulerAngles2(tvecs)
        # ser.write(("$%4.3f, %4.3f@" %(theta_x, theta_y)).encode())


    elif lost_count <= 5:
        aim_count = 0
        lost_count = lost_count + 1
        x = kf.statePost[0] + kf.statePost[3]
        y = kf.statePost[1] + kf.statePost[4]
        z = kf.statePost[2] + kf.statePost[5]
        predicted_X, predicted_Y, predicted_Z = Kal_predict(x, y, z)
        Kal_Visualization(predicted_X, predicted_Y, predicted_Z, img)


    else:
        kf.statePost = np.array([0, 0, 0, 0, 0, 0], np.float32)
        # lost_count = 0

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


def main_loop(out):
    while cap.isOpened():
        ret, img = cap.read()
        img = SHOW(L, img)
        if out != 0:
            out.write(cv2.resize(img, (1920, 1200)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    if debug:
        out = WriteVideo()
    else:
        out = 0
    main_loop(out)
    if debug:
        Para_visualization(CALC, X)
    cap.release()
    cv2.destroyAllWindows()


main()
