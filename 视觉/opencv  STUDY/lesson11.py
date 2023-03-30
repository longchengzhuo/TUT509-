#此篇为超大图像二值化
import cv2 as cv
import numpy as np


def huge_image1(image):
    h, w = image.shape[:2]
    cw = 128
    ch = 128
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):  # 以ch为步长进行循环
        for col in range(0, w, cw):  # 以cw为步长进行循环
            roi = gray[row:row+ch, col:col+cw]  # 从灰度图中获取感兴趣区域
            print(np.std(roi), np.mean(roi))  # 打印ROI区域的标准差及均值
            # dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)  # 局部阈值
            ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 全局阈值
            gray[row:row+ch, col:col+cw] = dst  # 将灰度图中的对应小块替换为二值化后的图像
    cv.imwrite("D:/binary1.jpg", gray)


def huge_image2(image):
    h, w = image.shape[:2]
    cw = 43
    ch = 43
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]
            # dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)  # 局部阈值
            print(np.std(roi), np.mean(roi))
            dev = np.std(roi)
            if dev < 15:
                gray[row:row + ch, col:col + cw] = 255  # 直接设为白色
            else:
                ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 全局阈值
                gray[row:row+ch, col:col+cw] = dst
    cv.imwrite("D:/binary2.jpg", gray)


src = cv.imread("D:\chaoda.jpeg")
print(src.shape)
huge_image1(src)


cv.waitKey(0)
cv.destroyAllWindows()