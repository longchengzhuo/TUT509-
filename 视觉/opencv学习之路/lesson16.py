#霍夫圆检测
import cv2 as cv
import numpy as np


def detect_circles_demo(image):
    dst  = cv.pyrMeanShiftFiltering(image, 10, 100)  # 边缘保留滤波EPF
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=23, param2=30, minRadius=0, maxRadius=0)#param1参数表示Canny边缘检测的高阈值，低阈值会被自动置为高阈值的一半。param2参数表示圆心检测的累加阈值，参数值越小，可以检测越多的圆圈，但返回的是与较大累加器值对应的圆圈。
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), int(i[2]), (0, 0, 255), 2)  # 画圆
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)  # 画圆心
    cv.imshow("circles", image)


src=cv.imread("D:\opencv\sources\samples\data\smarties.png")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
detect_circles_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()