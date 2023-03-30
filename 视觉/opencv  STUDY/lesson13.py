#此篇为图像梯度
#右减左，下减上
import cv2 as cv
import numpy as np


def sobel_demo(image):  # sobel算子
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回uint8形式，说白了就是负数取绝对值
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient_x", gradx)
    cv.imshow("gradient_y", grady)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0) #图片融合
    cv.imshow("gradient", gradxy)


def laplace_demo(image):  # 拉普拉斯算子
    grad = cv.Laplacian(image, cv.CV_32F)
    gradxy = cv.convertScaleAbs(grad)
    cv.imshow("grad", gradxy)


def hand_lpls(image):  # 手动拉普拉斯4邻域算子
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # [[1, 1, 1], [1, -8, 1], [1, 1, 1]]8邻域算子
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("laplace", lpls)


src=cv.imread("D:\opencv\sources\samples\data\lena.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
sobel_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()