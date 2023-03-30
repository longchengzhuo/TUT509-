import cv2
import numpy as np


def calculate_slope(line):
    """
    计算线段line的斜率
    :param line: np.array([[x_1, y_1, x_2, y_2]])
    :return:
    """
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)


edge_img = cv2.imread('masked_edge_img.jpg', cv2.IMREAD_GRAYSCALE)
# 获取所有线段
lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 15, minLineLength=40,
                        maxLineGap=20)
# 按照斜率分成车道线
left_lines = [line for line in lines if calculate_slope(line) > 0]
right_lines = [line for line in lines if calculate_slope(line) < 0]
