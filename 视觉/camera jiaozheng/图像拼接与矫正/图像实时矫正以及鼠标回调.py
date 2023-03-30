import cv2
import numpy as np
import glob
import math
import serial

WIN_NAME = 'pick_points'


def onmouse_pick_points(event, x, y, flags, param):
    serialPort = "COM2"  # 串口
    baudRate = 9600  # 波特率
    ser = serial.Serial(serialPort, baudRate, timeout=0.5)
    if event == cv2.EVENT_LBUTTONDOWN:
        z = [x, y]
        center = [0, 0]
        r = math.sqrt(math.pow(z[0] - center[0], 2) + math.pow(z[1] - center[1], 2))
        theta = math.atan2(z[1] - center[1], z[0] - center[0]) / math.pi * 180  # 转换为角度
        ser.write(('r = %f, theta = %f' % (r, theta) + '\n').encode())
        # print(ser.readline())  # 可以接收中文
        print(ser.readall().decode())
    ser.close()


def dabiaocamera1():
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    images = glob.glob("D:/biancheng\opencv python/*.jpg")
    i = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

    # 800*600
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    return ret, mtx, dist, rvecs, tvecs


ret, mtx, dist, rvecs, tvecs = dabiaocamera1()
capture = cv2.VideoCapture(1)
while (True):
    ret, frame = capture.read()
    img = cv2.flip(frame, 1)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    dst1 = dst[0:600, 100:700]
    cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, dst)
    cv2.imshow(WIN_NAME, dst)
    c = cv2.waitKey(1)
    if c == 27:
        break
