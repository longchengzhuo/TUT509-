import cv2 as cv
import numpy as np

def color_space_demo(image):
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    cv.imshow("gray",gray)
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    cv.imshow("hsv",hsv)
    yuv=cv.cvtColor(image,cv.COLOR_BGR2YUV)
    cv.imshow("yuv",yuv)
    ycrcb=cv.cvtColor(image,cv.COLOR_BGR2YCrCb)
    cv.imshow("ycrcb",ycrcb)


def extarce_object_demo():
    capture=cv.VideoCapture("D:/feipan.mp4")
    while(True):
        ret,frame=capture.read()
        if ret==False:
            break
        hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        lower_hsv = np.array([26,43,46])
        higher_hsv = np.array([34,255,255])
        mask1=cv.inRange(hsv,lowerb=lower_hsv,upperb=higher_hsv)
        dst=cv.bitwise_and(frame,frame,mask=mask1)
        cv.imshow("video",frame)
        cv.imshow("mask",dst)
        c=cv.waitKey(2)
        if c==27:
            break


src=cv.imread("D:/aaaaaa.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
extarce_object_demo()
cv.waitKey(0)

cv.destroyAllWindows()