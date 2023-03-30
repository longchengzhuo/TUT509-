#轮廓发现
import cv2 as cv
import numpy as np


def edge_demo(image):
    blur = cv.GaussianBlur(image, (5, 5), 0)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    # X gradient
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # Y gradient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # Canny
    edge = cv.Canny(xgrad, ygrad, 50, 150)
    cv.imshow("edge", edge)
    return edge


def contour_demo(image):
    '''dst = cv.GaussianBlur(image, (9, 9), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  #用大律法、全局自适应阈值方法进行图像二值化
    cv.imshow("binary", binary)'''
    binary = edge_demo(image)
    contours, heriachy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 此处与教程不同，如果加上cloneImage电脑会报错
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 1)
        print(i)
    cv.imshow("detect contours", image)


src=cv.imread("D:\opencv\sources\samples\data\pic5.png")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
edge_demo(src)
contour_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()