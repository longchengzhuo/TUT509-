import cv2
import numpy as np

img = cv2.imread('D:/biancheng\opencv python\camera jiaozheng\left_01.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SURF_create(6000)
keypoints, descriptor = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image= img, outImage= img, keypoints= keypoints, flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT, color= (0, 0, 255))

cv2.imshow('surf', img)
cv2.waitKey()
cv2.destroyAllWindows()
