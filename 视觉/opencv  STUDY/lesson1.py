import cv2 as cv


def video_1demo():
    capture=cv.VideoCapture(0)
    while(True):
        ret, frame=capture.read()
        frame = cv.flip(frame, 1)
        cv.imshow("video1", frame)
        c = cv.waitKey(1)
        if c == 27:
            break


def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)


# src=cv.imread("D:/aaaaaa.jpg")
# cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
# cv.imshow("input image",src)
# gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
# cv.imwrite("D:/aa.png",gray)
# get_image_info(src)
video_1demo()

cv.waitKey(0)

cv.destroyAllWindows()


