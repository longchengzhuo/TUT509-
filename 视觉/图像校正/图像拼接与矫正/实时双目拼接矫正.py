from Stitcher import Stitcher
import cv2
import numpy as np
import glob

stitcher = Stitcher()

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

obj_points1 = []
img_points1 = []

images1 = glob.glob("D:/biancheng/mmmp/*.jpg")
for fname in images1:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size1 = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret:
        obj_points1.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if [corners2]:
            img_points1.append(corners2)
        else:
            img_points1.append(corners)

obj_points2 = []
img_points2 = []

images2 = glob.glob("D:/biancheng/mmmp2/*.jpg")
for fname in images2:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size2 = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret:
        obj_points2.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if [corners2]:
            img_points2.append(corners2)
        else:
            img_points2.append(corners)

#800*600
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(obj_points1, img_points1, size1, None, None)
capture1 = cv2.VideoCapture(0)
#733*600
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(obj_points2, img_points2, size2, None, None)
capture2 = cv2.VideoCapture(2)
while (True):
    ret, frame1 = capture1.read()
    img1 = cv2.flip(frame1, 1)
    h1, w1 = img1.shape[:2]
    newcameramtx1, roi = cv2.getOptimalNewCameraMatrix(mtx1,dist1,(w1,h1),0,(w1,h1))
    dst1 = cv2.undistort(img1,mtx1,dist1,None,newcameramtx1)
    dst11 = dst1[0:600, 100:700]

    ret, frame2 = capture2.read()
    img2 = cv2.flip(frame2, 1)
    h2, w2 = img2.shape[:2]
    newcameramtx2, roi = cv2.getOptimalNewCameraMatrix(mtx2,dist2,(w2,h2),0,(w2,h2))
    dst2 = cv2.undistort(img2,mtx2,dist2,None,newcameramtx2)
    dst22 = dst2[0:600, 0:600]

    (result, vis) = stitcher.stitch([dst22, dst11], showMatches=True)
    cv2.imshow("Result", result)
    c = cv2.waitKey(1)
    if c == 27:
        break



cv2.waitKey(0)
cv2.destroyAllWindows()