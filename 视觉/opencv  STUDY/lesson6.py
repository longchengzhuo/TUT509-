import cv2 as cv
import numpy as np


def blur_demo(image):
    dst=cv.blur(image,(5,5))
    cv.imshow("blur_demo",dst)


def median_blur_demo(image):
    dst = cv.medianBlur(image, 5)
    cv.imshow("median_blur_demo", dst)


def custom_blur_demo(image):
    kernel=np.ones([5,5],np.float32)/25
    dst = cv.fliter2D(image, -1,kernel=kernel)
    cv.imshow("custom_blur_demo", dst)


def clamp(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


def gaussian_noise(image):
    h,w,c=image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            image[row, col, 0] = clamp(b+s[0])
            image[row, col, 1] = clamp(g+s[1])
            image[row, col, 2] = clamp(r+s[2])
    cv.imshow("noise image",image)


def bi_demo(image):
    dst = cv.bilateralFilter(image,0,100,10)
# InputArray src: 输入图像，可以是Mat类型，图像必须是8位或浮点型单通道、三通道的图像。
# int d: 表示在过滤过程中每个像素邻域的直径范围。如果这个值是非正数，则函数会从第五个参数sigmaSpace计算该值。
# double sigmaColor: 颜色空间过滤器的sigma值，这个参数的值月大，表明该像素邻域内有越宽广的颜色会被混合到一起，产生较大的半相等颜色区域。
# double sigmaSpace: 坐标空间中滤波器的sigma值，如果该值较大，则意味着越远的像素将相互影响，从而使更大的区域中足够相似的颜色获取相同的颜色。
# 当d>0时，d指定了邻域大小且与sigmaSpace无关，否则d正比于sigmaSpace. （这个参数可以理解为空间域核w_d的\sigma_d）
# int borderType=BORDER_DEFAULT: 用于推断图像外部像素的某种边界模式，有默认值BORDER_DEFAULT.
    cv.imshow("bi_demo",dst)


src=cv.imread("D:/aaaaaa.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
gaussian_noise(src)
dst = cv.GaussianBlur(src, (0, 0),5)
cv.imshow("1",dst)




cv.waitKey(0)
cv.destroyAllWindows()