#霍夫直线检测
import cv2 as cv
import numpy as np


# 标准霍夫线变换
def lines_detection(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3,apertureSize表示孔径尺寸，即Sobel算子的尺寸大小
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)  #threshold参数表示检测一条直线所需最少的曲线交点，阈值。
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的
        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x0 = int(a * rho)  # 代表x = r * cos（theta）
        y0 = int(b * rho)  # 代表y = r * sin（theta）
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))  # 注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 点的坐标必须是元组，不能是列表
    cv.imshow("lines_detection", image)


# 统计概率霍夫线变换
def line_detection_possible(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    edges = cv.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    for line in lines:
        print(type(line))
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv.imshow("line_detection_possible", image)


src=cv.imread("D:\opencv\sources\samples\data/building.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
lines_detection(src)

cv.waitKey(0)
cv.destroyAllWindows()