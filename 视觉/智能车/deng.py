import cv2
import numpy as np
def empty(a):
    pass

path = 'deng.jpg'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img = cv2.imread("debug.jpg")
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.inRange(imgHSV,lower,upper)
    r = cv2.erode(mask, kernel)
    r = cv2.dilate(r, kernel)

    r_contours, hierarchy1 = cv2.findContours(
        r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # HSV 红色
    # lower_red = np.array([150, 108, 0]) 45
    # upper_red = np.array([179, 209, 255])
    # red = cv2.inRange(imgHSV, lower_red, upper_red)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # r = cv2.erode(red, kernel)
    # r = cv2.dilate(r, kernel)
    #
    #
    #
    # # HSV 绿色
    # lower_green = np.array([19, 41, 54])
    # upper_green = np.array([91, 153, 255])
    # green = cv2.inRange(imgHSV, lower_green, upper_green)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # g = cv2.erode(green, kernel)
    # g = cv2.dilate(g, kernel)
    #
    # g_contours, hierarchy = cv2.findContours(
    #     g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # r_contours, hierarchy1 = cv2.findContours(
    #     r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # if (g_contours != []) or (r_contours != []):
    #     if g_contours == []:
    #         cv2.putText(img, "RED", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    #     elif r_contours == []:
    #         cv2.putText(img, "GREEN", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    #     else:
    #         print("what")
    #         print("r_contours", r_contours)
    #         RR = 0
    #         GG = 0
    #         for r_contour in r_contours:
    #             rect_r = cv2.minAreaRect(r_contour)  # 获取最小包围矩形
    #             xrCache, yrCache = rect_r[1]
    #             RR = RR + xrCache + yrCache
    #         for g_contour in g_contours:
    #             rect_g = cv2.minAreaRect(g_contour)
    #             xgCache, ygCache = rect_g[1]
    #             GG = GG + xgCache + ygCache
    #
    #         if RR > GG:
    #             cv2.putText(img, "RED", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    #
    #         else:
    #             cv2.putText(img, "GRENN", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)

    cv2.imshow("red", r)
    # cv2.imshow("green", g)
    cv2.imshow("debug", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()