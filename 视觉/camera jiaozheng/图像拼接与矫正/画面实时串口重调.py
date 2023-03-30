import cv2
import numpy as np
import math
import serial


def onmouse_pick_points(event, x, y, flags, param):
    serialPort = "COM2"  # 因为是COM2，所以要打开COM1串口
    baudRate = 9600  # 波特率
    ser = serial.Serial(serialPort, baudRate, timeout=0.5)
    if event == cv2.EVENT_LBUTTONDOWN:
        z = [x, y]
        center = [w / 2, h / 2]
        r = math.sqrt(math.pow(z[0] - center[0], 2) + math.pow(z[1] - center[1], 2))
        theta = -(math.atan2(z[1] - center[1], z[0] - center[0]) / math.pi * 180)  # 转换为角度
        ser.write(('r = %f, theta = %f' % (r, theta) + '\n').encode())
        # print(ser.readline())  # 可以接收中文
        print(ser.readall().decode())
    ser.close()


WIN_NAME = 'pick_points'
capture = cv2.VideoCapture(0)
while (True):
    ret, image = capture.read()
    h, w = image.shape[:2]
    cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, image)
    image = cv2.flip(image, 1)
    cv2.line(image, (0, int(h / 2)), (int(w), int(h / 2)), (0, 255, 0), 1)
    cv2.line(image, (int(w / 2), 0), (int(w / 2), int(h)), (0, 255, 0), 1)
    cv2.imshow(WIN_NAME, image)
    c = cv2.waitKey(1)
    if c == 27:
        break
