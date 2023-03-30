import cv2 as cv
import numpy as np


def template_demo():
    tpl = cv.imread("D:/zise.png")#tpl就是你要找的那个对象，就是那个小的图
    target = cv.imread("D:/zhifangtufanxiang.jpg")
    cv.imshow("template", tpl)
    cv.imshow("target", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCOEFF_NORMED, cv.TM_CCORR_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        #minVal参数表示返回的最小值，如果不需要，则使用NULL。
        #maxVal参数表示返回的最大值，如果不需要，则使用NULL。
        #minLoc参数表示返回的最小位置的指针（在2D情况下）； 如果不需要，则使用NULL。
        #maxLoc参数表示返回的最大位置的指针（在2D情况下）； 如果不需要，则使用NULL。
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc  # tl是矩阵左上角的点的坐标
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)  #br是矩形右下角的点的坐标
        """""
        rectangle
        用于绘制矩形
        rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img 
        img参数表示源图像。
        pt1参数表示矩形的一个顶点。
        pt2参数表示与pt1相对的对角线上的另一个顶点 。
        color参数表示矩形线条颜色 (RGB) 或亮度（灰度图像 ）。
        thickness参数表示组成矩形的线条的粗细程度。取负值时（如 CV_FILLED）函数绘制填充了色彩的矩形。
        lineType参数表示线条的类型。
        shift参数表示坐标点的小数点位数。
        """""
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.imshow("match-"+np.str(md), target)


src=cv.imread("D:/aaaaaa.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
template_demo()

cv.waitKey(0)
cv.destroyAllWindows()