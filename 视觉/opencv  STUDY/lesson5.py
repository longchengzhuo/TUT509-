import cv2 as cv
import numpy as np


def fill_color_demo(image):
    h,w=image.shape[:2]
    mask=np.zeros([h+2,w+2],np.uint8)
    cv.floodFill(image,mask,(30,30),(0,255,255),(100,100,100),(50,50,50),cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill_color_demo",image)


def fill_binary():
    image=np.zeros([200,200,3],np.uint8)
    image[100:150,100:150,:]=255
    cv.imshow("fill_binary",image)
    mask=np.ones([202,202,1],np.uint8)
    mask[100:151,101:151]=0
    cv.floodFill(image,mask,(102,102),(0,2,255),cv.FLOODFILL_MASK_ONLY)
    cv.imshow("fill_binary1",image)


src=cv.imread("D:/aaaaaa.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)

face=src[50:250,100:300]
gray=cv.cvtColor(face,cv.COLOR_BGR2GRAY)
backface=cv.cvtColor(gray,cv.COLOR_GRAY2BGR)
src[50:250,100:300]=backface
fill_binary()
cv.waitKey(0)
cv.destroyAllWindows()