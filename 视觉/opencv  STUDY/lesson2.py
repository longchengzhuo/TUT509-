import cv2 as cv
import numpy as np


def access_piexls(image):
    print(image.shape)
    height=image.shape[0]
    width=image.shape[1]
    channels=image.shape[2]
    print("width:%s,height:%s,channels:%s"%(width,height,channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv=image[row,col,c]
                image[row,col,c]=255-pv
    cv.imshow("pixels_demo",image)


def create_image():
    img=np.zeros([400,400,1],np.uint8)
    #img[:,:,0]=np.ones([400,400])*255
    img[:, :, 0] = np.ones([400, 400]) * 127
    cv.imshow("new img",img)


src=cv.imread("D:/aaaaaa.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
t1=cv.getTickCount()
create_image()
t2=cv.getTickCount()
time=(t2-t1)/cv.getTickFrequency()
print("time:%s s"%(time))
cv.waitKey(0)


cv.destroyAllWindows()