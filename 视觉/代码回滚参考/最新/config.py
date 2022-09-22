import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

#调试用视频地址
cap = cv2.VideoCapture('D:/biancheng\opencv python/robotmaster\DEBUG/result.avi')
#模板匹配所用模板
number_img = cv2.imread("D:/biancheng\opencv python/robotmaster\DEBUG\moban.jpg")
# 填写敌方阵营的颜色，可以是 RED 和 BLUE
mode = "BLU"
# 1为开启调试参数反馈以及参数可视化，0反之
debug = 1
#腐蚀膨胀核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 内参数矩阵  （？）---------------------------------------------------------------------------------------
DAHENG_OldCamera_intrinsic = {

    "mtx": np.array([[1.34386883e+03, 0., 1.02702867e+03],
                     [0., 1.38920787e+03, 5.62395968e+02],
                     [0., 0., 1.]],
                    dtype=np.double),
    "dist": np.array([[-0.13939675, 0.42409417, -0.00454986, 0.01027033, -0.61637364]], dtype=np.double)

}

MDV_OldCamera_intrinsic = {

    "mtx": np.array([[1.60954972e+03, 0., 6.36716902e+02],
                     [0., 1.61293941e+03, 5.27210575e+02],
                     [0., 0., 1.]],
                    dtype=np.double),
    "dist": np.array([[-0.08045919, 0.28519144, 0.00188052, -0.00196669, -0.07837469]], dtype=np.double)

}

# 模板匹配相关系数
correlation = 0.8

# 以下为用于参数可视化的参数初始化（勿动）-------------------------------------------------------------------
mathod = 2  # 1为灯条面积比可视化，2为大矩形长宽比，3为两灯条中心连线斜率，4为相关系数结果可视化
X      = []
XX     = 0
CALC   = []
Match_counter = 0

# 以下为用于 上一帧 和 当前帧 对比检测的 循环 参数初始化（勿动）----------------------------------------------
last_locx   = 0
last_locy   = 0
# 上一帧中装甲板中心坐标
last_width  = 0
last_height = 0
# 上一帧中装甲板长宽

# Pipei 中的参数初始化------------------------------------------------------------------------------------
R     = 0  # 相关系数的累计，方便可视化均值
N     = 0  # 达到几次所要求相似度
x_sum = 0
y_sum = 0

# 卡尔曼-------------------------------------------------------------------------------------------------


# 卡尔曼重置标志（计数用）
lost_count = 0
aim_count  = 0

kf = cv2.KalmanFilter(6, 3)
kf.measurementMatrix = np.array(
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0]], np.float32)

kf.transitionMatrix = np.array(
    [[1, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 1],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1]], np.float32)

kf.processNoiseCov = np.array(
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1]], np.float32) * 0.1
# 此系数越小表明,越相信"过程",越不相信"观测",就会离pnp解算值越远


# 函数-----------------------------------------------------------------------------------------------------
def rotateMatrixToEulerAngles2(coor):
    theta_x = np.arctan2(coor[0, 0], coor[2, 0]) / np.pi * 180
    theta_y = np.arctan2(coor[1, 0], np.sqrt(coor[0, 0] * coor[0, 0] + coor[2, 0] * coor[2, 0])) / np.pi * 180
    print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}")
    return theta_x, theta_y


def Match(match_img, number_img, rect_Center_X, rect_Center_Y, R, N, x_sum, y_sum, img):
    global Match_counter

    # 图像去噪、大津法二值化、缩小为原来的1/16
    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2GRAY)
    match_img = cv2.erode(match_img, kernel)
    match_img = cv2.dilate(match_img, kernel)
    ret2, temple = cv2.threshold(match_img, 0, 255, cv2.THRESH_OTSU)
    temple = cv2.cvtColor(temple, cv2.COLOR_GRAY2BGR)
    temple = cv2.resize(temple, (int(temple.shape[1] / 4), int(temple.shape[0] / 4)))
    number_img = cv2.resize(number_img, (int(number_img.shape[1] / 4), int(number_img.shape[0] / 4)))
    h, w, c = temple.shape

    results = cv2.matchTemplate(number_img, temple, cv2.TM_CCOEFF_NORMED)  # 按照标准相关系数匹配

    for y in range(len(results)):  # 遍历结果数组的行
        for x in range(len(results[y])):  # 遍历结果数组的列
            if results[y][x] > correlation:  # 如果相关系数大于0.99则认为匹配成功

                R = R + results[y][x]
                N = N + 1
                y_sum = y_sum + y
                x_sum = x_sum + x
    if N != 0:
        Match_counter = Match_counter + 1

        if debug and mathod == 4:
            CALC.append(R / N)
            X.append(Match_counter)

        if 0 < int(y_sum / N + h / 2) < 50 / 4 and 0 < int(x_sum / N + w / 2) < 60 / 4:
            cv2.putText(img, "4", (rect_Center_X - 20, rect_Center_Y + 20), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                        (255, 255, 255), 3)
        elif 0 < int(y_sum / N + h / 2) < 50 / 4 and 60 / 4 < int(x_sum / N + w / 2) < 140 / 4:
            cv2.putText(img, "1", (rect_Center_X - 20, rect_Center_Y + 20), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                        (255, 255, 255), 3)
        elif 0 < int(y_sum / N + h / 2) < 50 / 4 and 140 / 4 < int(x_sum / N + w / 2) < 200 / 4:
            cv2.putText(img, "5", (rect_Center_X - 20, rect_Center_Y + 20), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                        (255, 255, 255), 3)
        elif 0 < int(y_sum / N + h / 2) < 50 / 4 and 200 / 4 < int(x_sum / N + w / 2) < 255 / 4:
            cv2.putText(img, "3", (rect_Center_X - 20, rect_Center_Y + 20), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                        (255, 255, 255), 3)
        if debug:
            print("我真的服", int(y_sum / N + h / 2), int(x_sum / N + w / 2))
            print("R--------------------------------------------------------------------------------------------------",
                  R / N)

        return True


def WarpPerspect(img, ts_box):
    wid, hei = 50, 25
    pts1 = np.float32(ts_box)
    pts2 = np.float32([[0, 0], [wid, 0], [wid, hei], [0, hei]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (wid, hei))

    return warped


def Kal_Visualization(predicted_X, predicted_Y, predicted_Z, img):
    coordinate = [[predicted_X], [predicted_Y], [predicted_Z]]
    Zc = coordinate[2][0]
    if Zc != 0:
        focal_length = np.mat(MDV_OldCamera_intrinsic["mtx"])
        image_coordinate = (focal_length * coordinate) / Zc
        data = np.array(image_coordinate)
        if np.isnan(data).sum() == 3:
            pass
        else:
            image_coordinat = np.array(image_coordinate).reshape(-1)
            cv2.circle(img, (int(image_coordinat[0]), int(image_coordinat[1])), 20, (0, 0, 255), 5)


def Kal_predict(coordX, coordY, coordZ):
    measured = np.array([[np.float32(coordX)], [np.float32(coordY)], [np.float32(coordZ)]])
    predicted = kf.predict()
    kf.correct(measured)
    T = 1 + coordZ / 16.5 / 21.5

    print("T", T)
    x, y, z = int(kf.statePost[0] + T * kf.statePost[3]), \
              int(kf.statePost[1] + T * kf.statePost[4]), \
              int(kf.statePost[2] + T * kf.statePost[5])
    return x, y, z


def WriteVideo():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("width:", width, "height:", height)

    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920, 1200))
    return out


def Para_visualization(CALC, X):
    Y = CALC
    plt.plot(X, Y, 'o')
    plt.show()


def SHOW(L, img):
    t1 = cv2.getTickCount()
    img = L(img)
    t2 = cv2.getTickCount()
    spendTime = (t2 - t1) * 1 / (cv2.getTickFrequency())
    print("spendTime---------------------------------------------------------------", spendTime)
    FPS = int(1 / spendTime)
    FPS = 'The fps is %d' % (FPS)
    img = cv2.resize(img, (960, 600))
    cv2.putText(img, FPS, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    cv2.imshow('frame', img)
    return img

