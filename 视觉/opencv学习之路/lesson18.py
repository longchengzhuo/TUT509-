#对象测量
import cv2 as cv
import numpy as np


def measure_object(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 返回的ret是阈值， 该处需取反
    print("threshold value : %s"%ret)
    cv.imshow("binary", binary)
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    # 找二值图binary的轮廓，cv.RETR_EXTERNAL(只检索外部轮廓)、cv.RETR_TREE（检索全部轮廓）
    contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)  # 获取轮廓面积
        x, y, w, h = cv.boundingRect(contour)  # 获取轮廓外接矩形，返回四个参数（x，y）为矩形左上角的坐标，（w，h）是矩形的宽和高
        rate = min(w, h)/max(w, h)  # w和h里小的除以大的
        print("rectangle rate %s"%rate)
        mm = cv.moments(contour)  # 求取轮廓的几何矩
        print(type(mm))  # mm是字典类型
        cx = mm['m10']/mm['m00']
        cy = mm['m01']/mm['m00']
        cv.circle(dst, (np.int(cx), np.int(cy)), 3, (0, 255, 255), -1)  # 根据几何矩获取的中心点，画出中心圆
        # cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 根据轮廓外接矩形数据画出矩形
        print("contour area : %s"%area)
        approxCurve = cv.approxPolyDP(contour, 4, True)  # 4是与阈值的间隔大小，越小越易找出，True是是否找闭合图像
        print(approxCurve.shape)  # 打印该点集的shape，第一个数是代表了点的个数，也就是边长连接逼近数
        if approxCurve.shape[0] > 5:  # 多个的是圆形
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)  # 画出轮廓
        if approxCurve.shape[0] == 4:  # 4个的是矩形
            cv.drawContours(dst, contours, i, (0, 255, 255), 2)
        if approxCurve.shape[0] == 3:  # 3个的是三角形
            cv.drawContours(dst, contours, i, (255, 0, 255), 2)
    cv.imshow("measure contours", dst)


#关于cv.approxPolyDP函数的用法：
#（1）在曲线首尾两点A，B之间连接一条直线AB，该直线为曲线的弦；
#（2）得到曲线上离该直线段距离最大的点C，计算其与AB的距离d；
#（3）比较该距离与预先给定的阈值threshold的大小，如果小于threshold，则该直线段作为曲线的近似，该段曲线处理完毕。
#（4）如果距离大于阈值，则用C将曲线分为两段AC和BC，并分别对两段取信进行1~3的处理。


src=cv.imread("D:\opencv\sources\samples\data\detect_blob.png")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
measure_object(src)

cv.waitKey(0)
cv.destroyAllWindows()