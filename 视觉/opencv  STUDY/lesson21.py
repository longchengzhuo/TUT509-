import cv2 as cv
import numpy as np
from PIL import Image


def medianBlur(img, kernel):
    # 检测传入的kernel是否为一个合法数组.
    if kernel % 2 == 0 or kernel is 1:
        print('kernel size need 3, 5, 7, 9....')
        return None

    # 通过kernel的大小来计算paddingSize的大小
    paddingSize = kernel // 2

    # 获取图片的通道数

    # 获取传入的图片的大小
    height, width = img.shape[:2]

    matBase = np.zeros((height + paddingSize * 2, width + paddingSize * 2), dtype=img.dtype)

    matBase[paddingSize:-paddingSize, paddingSize:-paddingSize] = img

    # 创建用于输出的矩阵
    matOut = np.zeros((height, width), dtype=img.dtype)
    # 这里是遍历矩阵的每个点
    for x in range(height):
        for y in range(width):
            # 获取kernel X kernel 的内容,并转化成队并列
            line = matBase[x:x + kernel, y:y + kernel].flatten()
            # 队列排序处理.
            line = np.sort(line)
            # 取中间值赋值
            matOut[x, y] = line[(kernel * kernel) // 2]
    return matOut


c = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 5, 5, 5, 5, 5, 5, 1, 1],
              [1, 1, 5, 5, 5, 5, 5, 5, 1, 1],
              [1, 1, 5, 5, 8, 8, 5, 5, 1, 1],
              [1, 1, 5, 5, 8, 8, 5, 5, 1, 1],
              [1, 1, 5, 5, 5, 5, 5, 5, 1, 1],
              [1, 1, 5, 5, 5, 5, 5, 5, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], np.uint8)

myself = medianBlur(c, 3)
print(myself)
# [[0 1 1 1 1 1 1 1 1 0]
#  [1 1 1 1 1 1 1 1 1 1]
#  [1 1 1 5 5 5 5 1 1 1]
#  [1 1 5 5 5 5 5 5 1 1]
#  [1 1 5 5 5 5 5 5 1 1]
#  [1 1 5 5 5 5 5 5 1 1]
#  [1 1 5 5 5 5 5 5 1 1]
#  [1 1 1 5 5 5 5 1 1 1]
#  [1 1 1 1 1 1 1 1 1 1]
#  [0 1 1 1 1 1 1 1 1 0]]
