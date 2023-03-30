#此篇为图像金字塔(好难阿mmp）
import cv2 as cv
import numpy as np


def pyramid_demo(image):
    level = 3
    pyramid_images = []
    for i in range(level):#0，1，2
        dst = cv.pyrDown(image)
        pyramid_images.append(dst)
        cv.imshow("pyramid_down_"+str(i), dst)
        image = dst
    return pyramid_images


def laplace_demo(image):  # 拉普拉斯金字塔
    pyramid_images = pyramid_demo(image)    #做拉普拉斯金字塔必须用到高斯金字塔的结果
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):     #关于range的三个参数：第一个是起始数字, 第二个是终止数字(不包含), 第三个是步进值.此处指 [3-1， 1， 0]
        if i-1 < 0:  # 当i=0时，i-1<0,会导致数组溢出，因此要单独讨论
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("pyramid_images"+str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])  # 将尺寸设置为上一层图像的尺寸
            lpls = cv.subtract(pyramid_images[i-1], expand)  # 相减
            cv.imshow("pyramid_images" + str(i), lpls)





src=cv.imread("D:\opencv\sources\samples\data\lena.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
pyramid_demo(src)
laplace_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()