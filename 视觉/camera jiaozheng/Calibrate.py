import cv2
import numpy as np
import glob
#from config import *

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[-96:96:8j, -67:67:6j].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y


obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob('D:/biancheng\opencv python\camera jiaozheng\G\*.jpg')  #   拍摄的十几张棋盘图片所在目录

i=0
for fname in images:

    img = cv2.imread(fname)
    # 获取画面中心点
    #获取图像的长宽
    h1, w1 = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    u, v = img.shape[:2]
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        print("i:", i)
        i = i+1
        # 在原角点的基础上寻找亚像素角点
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #追加进入世界三维点和平面二维点中
        obj_points.append(objp)
        if [corners2]:
            print("2")
            img_points.append(corners2)
        else:
            img_points.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 640, 480)
        cv2.imshow('findCorners',img)
        i += 1
        cv2.imwrite('D:/biancheng\opencv python\camera jiaozheng\A/01red_conimg' + str(i) + '.jpg', img)
        cv2.waitKey(100)
cv2.destroyAllWindows()
# #%% 标定
# print('正在计算')
# #标定
ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
#
#
# print("ret:",ret  )
# print("mtx:\n",mtx)      # 内参数矩阵
# print("dist畸变值:\n",dist   )   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# print("rvecs旋转（向量）外参:\n",rvecs)   # 旋转向量  # 外参数
# print("tvecs平移（向量）外参:\n",tvecs  )  # 平移向量  # 外参数
# print(img_points)


MDV_WhiteCamera_intrinsic = {

    "mtx": np.array([[1.38012881e+03, 0.00000000e+00, 6.32563131e+02],
                     [0.00000000e+00, 1.41745621e+03, 3.95972569e+02],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.double),
    "dist": np.array([[ 0.00417728, -0.30223262, -0.01761736, -0.00159902, 1.24798931]], dtype=np.double)

}

MDV_OldCamera_intrinsic = {

    "mtx": np.array([[1.28643352e+03, 0.00000000e+00, 6.60207962e+02],
 [0.00000000e+00, 1.32861840e+03, 5.43963914e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                    dtype=np.double),
    "dist": np.array([[-0.1077022, 0.17465654, -0.0016777, 0.00283067, 0.51751706]], dtype=np.double)

}
#---------------------------------------------------------------------
#以下代码用于结算角度

obj_points = np.reshape(obj_points,(48,3))
img_points = np.reshape(img_points,(48,2))
success, rvecs, tvecs = cv2.solvePnP(obj_points, img_points,MDV_NewCamera_intrinsic["mtx"],
                                         MDV_NewCamera_intrinsic["dist"])


def rotateMatrixToEulerAngles2(RM):
    theta_x = np.arctan2(RM[0, 0], RM[2, 0]) / np.pi * 180
    theta_y = np.arctan2(RM[1, 0], np.sqrt(RM[0, 0] * RM[0, 0] + RM[2, 0] * RM[2, 0])) / np.pi * 180
    print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}")
    return theta_x, theta_y

# tvecs = np.array(tvecs)
print(tvecs)
for i in tvecs:
    rotateMatrixToEulerAngles2(i)