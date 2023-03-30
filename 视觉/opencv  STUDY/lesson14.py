#此篇为canny边缘检测
# 1.使用高斯滤波器，以平滑图像，消除噪声
# 2.计算每个像素点的梯度大小和方向
# 3.应用非极大值抑制，以确保此处为正真的边缘
# 4.应用双阈值来确定真实的和潜在的边缘

import cv2 as cv
import numpy as np


def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    cv.imshow("Canny Edge", edge_output)

    dst = cv.bitwise_and(image, image, mask = edge_output)
    cv.imshow("Color edge", dst)


def Canny_direct(image):
    v1 = cv.Canny(image, 80, 150)
    v2 = cv.Canny(image, 50, 100)
    res = np.hstack((v1, v2))
    cv.imshow("direct", res)


src=cv.imread("D:\opencv\sources\samples\data\lena.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
Canny_direct(src)

cv.waitKey(0)
cv.destroyAllWindows()