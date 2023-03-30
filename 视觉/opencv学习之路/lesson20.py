#分水岭算法（好难啊我去）
import cv2 as cv
import numpy as np


def watershed_demo(img):
    blur = cv.pyrMeanShiftFiltering(src, 10, 100)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # 降噪
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # 寻找确定前景区域
    #关于距离变换函数cv.distanceTransform
    #距离变换的基本含义是计算一个图像中非零像素点到最近的零像素点的距离，也就是到零像素点的最短距离个最常见的距离变换算法就是通过连续的腐蚀操作来实现，腐蚀操作的停止条件是所有前景像素都被完全腐蚀。这样根据腐蚀的先后顺序，我们就得到各个前景像素点到前景中心呗Ⅵ像素点的距离。根据各个像素点的距离值，设置为不同的灰度值。这样就完成了二值图像的距离变换
    #为啥要用距离变换？这样可以选出距离变换中的最大值作为初始标记点（如果是反色的话，则是取最小值）来注水
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 寻找未知区域
    sure_fg = np.uint8(sure_fg)
    unknow = cv.subtract(sure_bg, sure_fg)

    # 标记标签
    #连通域处理函数cv2.connectedComponents
    #连通区域一般是指图像中具有相同像素值且位置相邻的前景像素点组成的图像区域。连通区域分析是指将图像中的各个连通区域找出并标记。
    #参数介绍如下：
    #image：也就是输入图像，必须是二值图，即8位单通道图像。（因此输入图像必须先进行二值化处理才能被这个函数接受）
    #返回值：
    #num_labels：所有连通域的数目
    #labels：图像上每一像素的标记，用数字1、2、3…表示（不同的数字表示不同的连通域）
    ret, markers = cv.connectedComponents(sure_fg)

    # 在所有标签中添加一个，确保背景不是0，而是1
    markers = markers + 1

    # 现在，将未知区域标记为0
    markers[unknow == 255] = 0
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv.imshow("result", img)


src=cv.imread("D:\opencv\sources\doc\js_tutorials\js_assets\coins.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
watershed_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()