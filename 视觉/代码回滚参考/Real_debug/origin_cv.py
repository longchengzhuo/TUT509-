import cv2
import time
import math
import numpy as np
import serial
import mvsdk
import platform



#嘉然带我走吧我不想写了！！！
#print(ser.readall().decode())

serialPort = "COM3"  # 串口
baudRate = 115200  # 波特率
ser = serial.Serial(serialPort, baudRate, timeout=0.5)
print("参数设置：串口=%s ，波特率=%d" % (serialPort, baudRate))


def main_loop():
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
    mvsdk.CameraSetExposureTime(hCamera, 0.07 * 1000)

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
            L(img)
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Press q to end", img)

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

    # 关闭相机
    mvsdk.CameraUnInit(hCamera)

    # 释放帧缓存
    mvsdk.CameraAlignFree(pFrameBuffer)


def L(img):
    mode = "BLUE"       # 填写敌方阵营的颜色，可以是 RED 和 BLUE
    debug = False       # 一键开启调试模式


    # 定义准星位置
    sightX = int((img.shape[1]) / 2)
    sightY = int((img.shape[0]) / 2)

    if mode == "BLUE":

        # 根据颜色筛选
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # HSV BLUE
        lower = np.array([100, 43, 46], dtype="uint8")  # [70, 0, 250]
        upper = np.array([124, 255, 255], dtype="uint8")  # [150, 80, 255]
        hsv_mask = cv2.inRange(hsv_image, lower, upper)

    else:
        # 根据颜色筛选
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # HSV 红色
        lower = np.array([0, 43, 46], dtype="uint8")  # 175, 205, 170
        upper = np.array([10, 255, 255], dtype="uint8")  # 40 ,155, 255
        hsv_mask = cv2.inRange(hsv_image, lower, upper)

    img2 = hsv_mask
    binnary , contours, hierarchy = cv2.findContours(
        img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓

    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

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
    # for_test
    #print("--------------------------")
    for contour in contours:
        rect = cv2.minAreaRect(contours[n])  # 获取最小包围矩形
        xCache, yCache = rect[0]  # 获取矩形的中心坐标
        rectpoint = cv2.boxPoints(rect)  #获取矩形四个角点
        widthC, heigthC = rect[1]  # 获取矩形的长宽
        if ((widthC==0) | (heigthC==0))==True:
            break
        angleCache = rect[2]  # 获取矩形的角度 (-90, 0]
        # print("·widthC:", widthC)
        # print(" heigthC:", heigthC)
        if heigthC > widthC:
            # if heigthC / widthC >= 2.1:  # 长宽比满足条件
            #print("heigthC / widthC", (heigthC / widthC))
            if widthC < heigthC and heigthC / widthC >= 1:  # 灯条是竖直放置，长宽比满足条件
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
        else:
            #print("widthC / heigthC ", (widthC / heigthC))
            if widthC / heigthC >= 1:  # 长宽比满足条件
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
                longSide.append(int(widthC))
                # shortSide.append(heigthC)
                angle.append(angleCache + 90)
                #print("widthC > heigthC")
                n = n + 1  # 有效矩形计数
    #print("有效矩形个数为", n)
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

                #print("两个灯条角度差值：", abs(angle[count] - angle[findCache]), "大矩形长宽比值：", calcCache)
                if abs(angle[count] - angle[findCache]) < 90 and (2 < calcCache) and (x[findCache] - x[count]) ** 2 > (y[findCache] - y[count]) ** 2:  # 满足匹配条件


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


    rectbox = rectbox11 + rectbox12 + rectbox21 + rectbox22
    if rectbox != []:
        img_points = np.array(rectbox, dtype=np.double)
        obj_points = [[-62., -30., 0.],
                       [-62., 30., 0.],
                       [62., -30., 0.],
                       [ 62., 30., 0.]]
        obj_points = np.reshape(obj_points, (4, 3))

        success, rvecs, tvecs = cv2.solvePnP(obj_points, img_points,
                                             np.array([[1.60954972e+03, 0., 6.36716902e+02],
                                                       [0., 1.61293941e+03, 5.27210575e+02],
                                                       [0., 0., 1.]], dtype=np.double),
                                             np.array([[-0.08045919, 0.28519144, 0.00188052, -0.00196669, -0.07837469]], dtype=np.double))
        #for_test
        #print(tvecs)
        tvecs = np.array(tvecs)
        theta_x, theta_y = rotateMatrixToEulerAngles2(tvecs)
        print("1")
        string = str("$%4.3f,%4.3f@" %(theta_x, theta_y))
        # print(string)
        ser.write(string.encode("utf-8"))
        # ser.write(("$%4.3f,%4.3f@" %(theta_x, theta_y)).encode("utf-8","strict"))
        # print(ser.readall().decode("utf-8","strict"))
    else:
        string = str("$0.0,0.0@")
        ser.write(string.encode("utf-8"))


    # 画准星
    cv2.line(img, (sightX, sightY - 4),
             (sightX, sightY - 9), (50, 250, 100), 2)  # 上
    cv2.line(img, (sightX, sightY + 4),
             (sightX, sightY + 9), (50, 250, 100), 2)  # 下
    cv2.line(img, (sightX - 4, sightY),
             (sightX - 9, sightY), (50, 250, 100), 2)  # 左
    cv2.line(img, (sightX + 4, sightY),
             (sightX + 9, sightY), (50, 250, 100), 2)  # 右
    # for_test
    #print("--------------------------")

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