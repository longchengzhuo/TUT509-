import cv2  # 导入opencv包
import gxipy as gx  # 导入大恒相机Python包
import time
import cv2
import time
import math
import numpy as np
import cProfile
import pstats

def READ():
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()

    # 获取设备基本信息列表
    str_sn = dev_info_list[0].get("sn")
    # 通过序列号打开设备
    cam = device_manager.open_device_by_sn(str_sn)
    cam.ExposureTime.set(1000.0)

    cam.BalanceWhiteAuto.set(1)
    if cam.GammaParam.is_readable():
     gamma_value = cam.GammaParam.get()
     print(gamma_value)
     gamma_lut = gx.Utility.get_gamma_lut(3)

    else:
     gamma_lut = None
    if cam.ContrastParam.is_readable():
     contrast_value = cam.ContrastParam.get()
     contrast_lut = gx.Utility.get_contrast_lut(contrast_value)

    else:
     contrast_lut = None
    color_correction_param = cam.ColorCorrectionParam.get()


    # 开始采集
    cam.stream_on()



    while 1:
        t1 = cv2.getTickCount()

        raw_image = cam.data_stream[0].get_image()  # 使用相机采集一张图片





        rgb_image = raw_image.convert("RGB")  # 从彩色原始图像获取 RGB 图像
        rgb_image.image_improvement(color_correction_param, contrast_lut,gamma_lut)
        numpy_image = rgb_image.get_numpy_array()  # 从 RGB 图像数据创建 numpy 数组
        if rgb_image is None:
            continue
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)  # opencv采用的是BGR图像， 讲RGB转为BGR


        cv2.namedWindow('video', cv2.WINDOW_NORMAL)  # 创建一个名为video的窗口
        cv2.imshow('video', numpy_image)  # 将捕捉到的图像在video窗口显示

        c = cv2.waitKey(1);
        if (c == 27):
            break

    # 停止录制,关闭设备
    # cam.stream_off()
    # cam.close_device()






# READ()


img = cv2.imread("D:/biancheng\opencv python/robotmaster\ss.jpg")

#嘉然带我走吧我不想写了！！！
def L(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
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
    t99 = cv2.getTickCount()
    # img00 = cv2.erode(img00, kernel)

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

        img00 = cv2.resize(img00, (int(img00.shape[1] / 3), int(img00.shape[0] / 3)))
        t12 = cv2.getTickCount()#---------------------------------------------------

        spendTime1000 = (t12 - t99) * 1 / (cv2.getTickFrequency())
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
        # mask = cv2.resize(mask, (int(mask.shape[1] * 2), int(mask.shape[0] * 2)))
    t9 = cv2.getTickCount()#---------------------------------------------------
    spendTime5 = (t9 - t12) * 1 / (cv2.getTickFrequency())
    print("5 time:", spendTime5)#0.011

    img2 = mask
    # cv2.imwrite("mask0.jpg", img2)

    # th2 = cv2.dilate(img2, kernel)
    # # cv2.imwrite("mask1.jpg", th2)
    # cv2.imshow("2", th2)
    img2 = cv2.erode(img2, kernel)
    # cv2.imwrite("mask2.jpg", img2)
    cv2.imshow("3", img2)
    # img2 = cv2.erode(img2, kernel)
    # cv2.imshow("4", img2)

    # cv2.imwrite("mask3.jpg", img2)
    # img2 = cv2.dilate(img2, kernel)
    # img2 = cv2.dilate(img2, kernel)
    # cv2.imwrite("mask4.jpg", img2)#可以再膨胀一次
    # img2 = cv2.erode(img2, kernel)
    # img2 = cv2.dilate(img2, kernel)
    # cv2.imwrite("mask5.jpg", img2)
    img2 = cv2.resize(img2, (int(img2.shape[1] * 3), int(img2.shape[0] * 3)))

    binnary , contours, hierarchy = cv2.findContours(
        img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓

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


    t10 = cv2.getTickCount()#---------------------------------------------------
    spendTime5 = (t10 - t9) * 1 / (cv2.getTickFrequency())
    print("6 time:", spendTime5)#0.004
    print(img2.shape)
    cv2.imshow("1", img2)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img2 = cv2.erode(img2, kernel1)
    cv2.imshow("5", img2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
L(img)