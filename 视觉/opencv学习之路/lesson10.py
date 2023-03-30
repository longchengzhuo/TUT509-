#此篇为图像二值化
import cv2 as cv
import numpy as np

# cv::InputArray src, // 输入图像
# cv::OutputArray dst, // 输出图像
# double thresh, // 阈值
# double maxValue, // 向上最大值
# int adaptiveMethod, // 自适应方法，平均或高斯
#自动找到阈值算法（只要用了这个算法，则前面自己设定的阈值就没用）


def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value %s"%ret)
    cv.imshow("binary", binary)





# cv::InputArray src, // 输入图像
# cv::OutputArray dst, // 输出图像
# double maxValue, // 向上最大值
# int adaptiveMethod, // 自适应方法，平均或高斯
# int thresholdType // 阈值化类型
# int blockSize, // 块大小
# double C // 常量


def local_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow("binary", binary)


def custom_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    print("mean = ", mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    print("ret = ", ret)
    cv.imshow("binary", binary)


src=cv.imread("D:/aaaaaa.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
custom_threshold(src)

cv.waitKey(0)
cv.destroyAllWindows()