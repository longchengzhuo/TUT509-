#各种形态学操作
import cv2 as cv
import numpy as np


def erode_demo(image):  # 腐蚀，用低值替代中心点，黑色变多
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 取反，大于阈值的设为0， 小于阈值的设为Maxval
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))  #第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
    dst = cv.erode(binary, kernel)
    cv.imshow("erode_demo", dst)


def dilate_demo(image):  # 膨胀，让高值替代中心点，白色变多
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))  # 返回指定形状和尺寸的结构元素（内核矩阵）,第一个参数表示内核的形状
    dst = cv.dilate(binary, kernel)
    cv.imshow("dilate_demo", dst)


# 先腐蚀后膨胀叫开运算（因为先腐蚀会分开物体，这样容易记住），其作用是：分离物体，消除小区域。
def open_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # 椭圆结构
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow("open_demo", binary)

# 闭运算则相反：先膨胀后腐蚀（先膨胀会使白色的部分扩张，以至于消除/"闭合"物体里面的小黑洞，所以叫闭运算）
def close_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 矩形结构
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow("close_demo", binary)


#顶帽
#原图像减去开运算结果
#开运算可以消除暗背景下的较亮区域，所以顶帽变换可以得到原图中灰度较亮的区域。
def top_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    cimage = np.array(gray.shape, np.uint8)
    cimage = 100
    dst = cv.add(dst, cimage)  # 增加亮度
    cv.imshow("tophat", dst)


#黑帽
#原图片减去闭运算结果
#闭运算可以删除亮度较高背景下的较暗区域，所以黑帽变换可以得到原图片中灰度较暗的区域
def Black_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    cimage = np.array(gray.shape, np.uint8)
    cimage = 100
    dst = cv.add(dst, cimage)  # 增加亮度
    cv.imshow("Black_hat", dst)


#二值图像的顶帽操作
def hat_binary_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel)
    cv.imshow("hat", dst)


#基本梯度
#基本梯度是用膨胀后的图像减去腐蚀后的图像得到差值图像
def gradient1_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)
    cv.imshow("hat", dst)


#内外梯度
def gradient2_demo(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dm = cv.dilate(image, kernel)
    em = cv.erode(image, kernel)
    dst1 = cv.subtract(image, em) # internal gradient内梯度
    dst2 = cv.subtract(dm, image) # external gradient外梯度
    cv.imshow("internal", dst1)
    cv.imshow("external", dst2)


src=cv.imread("D:\opencv\sources\samples\data\lena.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
gradient2_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()