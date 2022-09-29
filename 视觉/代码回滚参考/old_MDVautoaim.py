import cv2
import time
import math
import numpy as np
import serial
import mvsdk
import platform
from CRC import *



X = []
AREA = []
KK = []
CALC = []
XX = 0

#嘉然带我走吧我不想写了！！！
#print(ser.readall().decode())

serialPort = "COM3"  # 串口
baudRate = 115200  # 波特率
ser = serial.Serial(serialPort, baudRate, timeout=0.5)
print("参数设置：串口=%s ，波特率=%d" % (serialPort, baudRate))

fps = 1
def main_loop():
    global fps
    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]
    print(DevInfo)

    # 打开相机
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))
        return

    # 获取相机特性描述
    cap = mvsdk.CameraGetCapability(hCamera)

    # 判断是黑白相机还是彩色相机
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # 相机模式切换成连续采集
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    # 手动曝光，曝光时间30ms
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 0.7 * 1000)

    # 让SDK内部取图线程开始工作
    mvsdk.CameraPlay(hCamera)

    # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

    # 分配RGB buffer，用来存放ISP输出的图像
    # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        # 从相机取一帧图片
        try:
            t1 = cv2.getTickCount()
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
            # linux下直接输出正的，不需要上下翻转
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            img = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                   1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            img = cv2.flip(img, -1, dst=None)


            L(img, fps)



            t2 = cv2.getTickCount()
            spendTime = (t2 - t1) * 1 / (cv2.getTickFrequency())


            FPS = 1 / spendTime
            fps = int(FPS)
            # string = str("$%4.3f@" % (fps))
            # ser.write(string.encode("utf-8"))
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Press q to end", img)

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

    # 关闭相机
    mvsdk.CameraUnInit(hCamera)

    # 释放帧缓存
    mvsdk.CameraAlignFree(pFrameBuffer)

def L(img, fps):
    global XX
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    t3 = cv2.getTickCount()#---------------------------------------------------
    mode = "RED"       # 填写敌方阵营的颜色，可以是 RED 和 BLUE
    shuzi_img = img.copy()
    img00 = img.copy()

    t20 = cv2.getTickCount()#---------------------------------------------------
    spendTime100 = (t20 - t3) * 1 / (cv2.getTickFrequency())
    # print("100 time:", spendTime100)  # 0.011

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
        # print("111 time:", spendTime1000)  # 0.011

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
    # print("5 time:", spendTime5)#0.011

    img2 = mask
    # cv2.imwrite("mask0.jpg", img2)

    img2 = cv2.dilate(img2, kernel)
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
    # print("6 time:", spendTime5)#0.004

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
        # print("nmd",(int(xCache), int(yCache)))
        rectpoint = np.int0(rectpoint)
        cv2.drawContours(img, [rectpoint], -1, (0, 255, 0,), 2)


        if angleCache < -45:
            v = heigthC
            heigthC = widthC
            widthC = v



        # print("·widthC:", widthC)
        # print(" heigthC:", heigthC)
        # print("angleCache", angleCache)
        # print("heigthC / widthC", (heigthC / widthC))

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
    # print("1 time:", spendTime1)
    # print("1.1 time:", spendTime11)
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
                area1 = longSide[count] * shortSide[count]
                area2 = longSide[findCache] * shortSide[findCache]

                if area1 != 0 and area2 != 0:
                    calc_area = area1 / area2
                    if calc_area < 1:
                        calc_area = 1 / calc_area
                    # print("两个灯条角度差值：", abs(angle[count] - angle[findCache]), "大矩形长宽比值：", calcCache)
                    # print("面积比：", calc_area)
                    if (1 <= calc_area < 5) and (2 < calcCache < 5) and (x[findCache] - x[count]) ** 2 > (y[findCache] - y[count]) ** 2:  # 满足匹配条件
                        target.append((count, findCache))
                        locX.append(int((x[count] + x[findCache]) / 2))
                        locY.append(int((y[count] + y[findCache]) / 2))
                        dis.append(
                            math.sqrt((locX[pairNum] - sightX) ** 2 + (locY[pairNum] - sightY) ** 2))
                        cac = 'The %f' %(calc_area)
                        # cv2.putText(img, cac, (locX[pairNum], locY[pairNum]), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)

                        # AREA.append(calc_area)
                        # CALC.append(calcCache)

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
        # print("pairNum", pairNum)
        if pairNum != 0:
            # 寻找离准星最近的装甲板
            disCalcCache = dis[0]
            targetNum = 0  # 存储距离准星做进的装甲板编号
            for count in range(0, pairNum):
                if dis[count] < disCalcCache:
                    targetNum = count
                    disCalcCache = dis[count]

            K = abs(y[target[targetNum][0]] - y[target[targetNum][1]]) / abs(x[target[targetNum][0]] - x[target[targetNum][1]])
            # print('K', K)
            # XX = XX + 1
            # X.append(XX)
            KK.append(K)
            if K<0.4 :

                # cv2.line(img, (locX[targetNum], 0),
                #          (locX[targetNum], 480), (0, 255, 0), 1)  # 画竖线
                # cv2.line(img, (0, locY[targetNum]),
                #          (720, locY[targetNum]), (0, 255, 0), 1)  # 画横线
                # cv2.line(img, (x[target[targetNum][0]], y[target[targetNum][0]]), (
                #     x[target[targetNum][1]], y[target[targetNum][1]]), (255, 255, 255), 2)  # 画连接线
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

                if ts_zs[0] > ts_ys[0]:
                    h_zs = ts_zs
                    h_zx = ts_zx
                    ts_zs = ts_ys
                    ts_zx = ts_yx
                    ts_ys = h_zs
                    ts_yx = h_zx


                ts_box = [ts_zs,ts_ys,ts_yx, ts_zx]

                cv2.line(img, zs, ys, (255, 0, 0), 3)
                cv2.line(img, ys, yx, (255, 0, 0), 3)
                cv2.line(img, yx, zx, (255, 0, 0), 3)
                cv2.line(img, zx, zs, (255, 0, 0), 3)

                # cv2.putText(img, "1", zs, cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200),
                #             5)
                # cv2.putText(img, "2", ys, cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200),
                #             5)
                # cv2.putText(img, "3", yx, cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200),
                #             5)
                # cv2.putText(img, "4", zx, cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200),
                #             5)
                # print("1234",zs,ys,zx,yx)
                AA = int((xx3[r311] + xx2[r222])/2)
                BB = int((yy1[r111] + yy4[r422])/2)
                # cv2.circle(
                #     img, (int((x1[top11] + x1[top12]) / 2),
                #           int((y1[top11] + y1[top12]) / 2)), 5, (255, 255, 100), 1)
                # cv2.circle(
                #     img, (int((x1[top13] + x1[top14]) / 2),
                #           int((y1[top13] + y1[top14]) / 2)), 5, (255, 100, 255), 1)

                rectbox11.append([int((x1[top11] + x1[top12]) / 2),
                                  int((y1[top11] + y1[top12]) / 2)])
                rectbox12.append([int((x1[top13] + x1[top14]) / 2),
                                  int((y1[top13] + y1[top14]) / 2)])

                # cv2.circle(
                #     img, (int((x2[top21] + x2[top22]) / 2),
                #           int((y2[top21] + y2[top22]) / 2)), 5, (255, 255, 100), 1)
                # cv2.circle(
                #     img, (int((x2[top23] + x2[top24]) / 2),
                #           int((y2[top23] + y2[top24]) / 2)), 5, (255, 100, 255), 1)

                rectbox21.append([int((x2[top21] + x2[top22]) / 2),
                                  int((y2[top21] + y2[top22]) / 2)])
                rectbox22.append([int((x2[top23] + x2[top24]) / 2),
                                  int((y2[top23] + y2[top24]) / 2)])

                # cv2.putText(img, "11", (x1[top11], y1[top11]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                #             1)
                # cv2.putText(img, "12", (x1[top12], y1[top12]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                #             1)
                # cv2.putText(img, "13", (x1[top13], y1[top13]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                #             1)
                # cv2.putText(img, "14", (x1[top14], y1[top14]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                #             1)
                #
                # cv2.putText(img, "21", (x2[top21], y2[top21]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                #             1)
                # cv2.putText(img, "22", (x2[top22], y2[top22]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                #             1)
                # cv2.putText(img, "23", (x2[top23], y2[top23]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                #             1)
                # cv2.putText(img, "24", (x2[top24], y2[top24]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200),
                #             1)

                # print("23",(x2[top23], y2[top23]))
                # print("14",(x1[top14], y1[top14]))
                # print("21", (x2[top21], y2[top21]))
                # print("12",(x1[top12], y1[top12]))
                #
                # print("第一个矩阵：","x1[top11]", x1[top11], "x1[top12]", x1[top12], "\ny1[top11]", y1[top11], "y1[top12]", y1[top12], "\nx1[top13]", x1[top13], "x1[top14]", x1[top14], "\ny1[top13]", y1[top13], "y1[top14]", y1[top14])
                # print("第一个矩阵：", "x2[top11]", x2[top11], "x2[top12]", x2[top12], "\ny2[top11]", y2[top11], "y2[top12]",
                #       y2[top12], "\nx2[top13]", x2[top13], "x2[top14]", x2[top14], "\ny2[top13]", y2[top13], "y2[top14]",
                #       y2[top14])
                if int((rectbox21[0][0] + rectbox22[0][0]) / 2) >= int((rectbox11[0][0] + rectbox12[0][0]) / 2):
                    a = rectbox11
                    b = rectbox12
                    rectbox11 = rectbox21
                    rectbox12 = rectbox22
                    rectbox21 = a
                    rectbox22 = b
            else:
                pass
    rectbox = rectbox22 + rectbox12 + rectbox21 + rectbox11
    t5 = cv2.getTickCount()
    spendTime2 = (t5 - t4) * 1 / (cv2.getTickFrequency())
    # print("2 time:", spendTime2)
    global  lost_count
    global  aim_count
    if rectbox != []:
        # print("rectbox", rectbox)
        # tf_img = toushibianhuan(shuzi_img, ts_box)
        # # print(Pipei(tf_img))
        # # if Pipei(tf_img, jiugongge, AA, BB):
        # if aim_count == 0:
        #     kf.statePost = np.array([0, 0, 0, 0, 0, 0], np.float32)
        if True:
            # aim_count = aim_count + 1
            lost_count = 0
            img_points = np.array(rectbox, dtype=np.double)
            obj_points = [[-67., -25., 0.],
                          [67., -25., 0.],
                          [-67.,25., 0.],
                          [67., 25., 0.]]
            obj_points = np.reshape(obj_points, (4, 3))

            success, rvecs, tvecs = cv2.solvePnP(obj_points, img_points,
                                                 np.array([[1.60954972e+03, 0., 6.36716902e+02], [0., 1.61293941e+03, 5.27210575e+02], [0., 0., 1.]], dtype=np.double),
                                                 np.array([[-0.08045919, 0.28519144, 0.00188052, -0.00196669, -0.07837469]],
                                                          dtype=np.double))#这些都是迈德威的参数
            tvecs = np.array(tvecs)

            theta_x, theta_y = rotateMatrixToEulerAngles2(tvecs)

            f1 = bytes("$", encoding='utf8')
            f2 = 10
            f3 = float(theta_x)
            f4 = float(theta_y)
            f5 = fps

            # print('fps_______-----------------------------------------------------__', f2, f3, fps)
            pch_Message1 = get_Bytes(f1, is_datalen_or_fps=0)
            pch_Message2 = get_Bytes(f2, is_datalen_or_fps=1)
            pch_Message3 = get_Bytes(f3, is_datalen_or_fps=0)
            pch_Message4 = get_Bytes(f4, is_datalen_or_fps=0)
            pch_Message5 = get_Bytes(f5, is_datalen_or_fps=2)


            pch_Message = pch_Message1 + pch_Message2 + pch_Message3 + pch_Message4 + pch_Message5
            # # print(pch_Message)
            # print("------------------------------------------------", pch_Message1 , pch_Message2 , pch_Message3 , pch_Message4 )




            wCRC = get_CRC16_check_sum(pch_Message, CRC16_INIT)
            ser.write(struct.pack("=cBffHi", f1, f2, f3, f4, f5, wCRC))



    else:
        f1 = bytes("$", encoding='utf8')
        f2 = 10
        f3 = 0
        f4 = 0
        f5 = 1

        pch_Message1 = get_Bytes(f1, is_datalen_or_fps=0)
        pch_Message2 = get_Bytes(f2, is_datalen_or_fps=1)
        pch_Message3 = get_Bytes(f3, is_datalen_or_fps=0)
        pch_Message4 = get_Bytes(f4, is_datalen_or_fps=0)
        pch_Message5 = get_Bytes(f5, is_datalen_or_fps=2)

        pch_Message = pch_Message1 + pch_Message2 + pch_Message3 + pch_Message4 + pch_Message5
        # print(pch_Message)

        wCRC = get_CRC16_check_sum(pch_Message, CRC16_INIT)

        # string = pch_Message + str(format(wCRC, '#04x')[2:])
        #
        # print(string.encode("utf-8"))
        ser.write(struct.pack("=cBffHi", f1, f2, f3, f4, f5, wCRC))

    # elif lost_count <= 15:
    #     aim_count = 0
    #     lost_count = lost_count + 1
    #     x = kf.statePost[0] + kf.statePost[3]
    #     y = kf.statePost[1] + kf.statePost[4]
    #     z = kf.statePost[2] + kf.statePost[5]
    #     coor = [(x, y, z)]
    #     kal(coor)
    #
    #
    # else:
    #     kf.statePost = np.array([0, 0, 0, 0, 0, 0], np.float32)
    #     lost_count = 0


    t6 = cv2.getTickCount()
    spendTime3 = (t6 - t5) * 1 / (cv2.getTickFrequency())
    # print("3 time:", spendTime3)

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
    theta_x = np.arctan2(RM[0, 0], RM[2, 0])
    theta_y = np.arctan2(RM[1, 0], np.sqrt(RM[0, 0] * RM[0, 0] + RM[2, 0] * RM[2, 0]))
    #print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}")
    return theta_x, theta_y
# for_test


def main():
	try:
		main_loop()
	finally:
		cv2.destroyAllWindows()

main()
