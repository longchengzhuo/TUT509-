import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



def plot_demo(image):#直方图
    plt.hist(image.ravel(),256,[0,256])  # “image.ravel()”的作用是一维化图像数组
    plt.show()

#plt.hist和cv.calcHist的区别？？？？？？
#一般情况下，我们是使用连线将直方图的频数给连接起来，
# 只要我们获得了直方图中每个小图的数据以及相应频数（x, y坐标），
# 我们就能用plt.plot()函数给勾勒出来了。而plt.hist()函数正好是可以返回每个小图的频数的。
def image_hist(image):#三通道直方图
    color = ('b', 'g', 'r')
    for i, color in enumerate(color):
        # 计算出直方图，calcHist(images, channels, mask, histSize(有多少个bin), ranges[, hist[, accumulate]]"ranges是用来限制直方图在电脑显示屏窗口的大小的")
        # hist 是一个 256x1 的数组，每一个值代表了与该灰度值对应的像素点数目。
        hist = cv.calcHist(image, [i], None, [180], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

#全局直方图均衡化
def equalhist_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow("equalhist_demo",dst)


#局部直方图均衡化
#在有些情况下直接对图像进行直方图均衡化操作会使得背景的对比度上升而某些重要区域的对比度下降
# 自适应直方图均衡化通过将图像分割成块并对每一块进行直方图均衡化来避免了这种现象，
# 但这样又放大了噪声的影响。因此，CLAHE在自适应直方图均衡化的基础上加入了对比度限制，
# 将超过一定对比度限制的像素的值裁剪并均匀分摊到整个区间，使得直方图总面积不变：
def clahe_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    #clipLimit：对比度限制值，默认为40.0
    #tileGridSize：分块大小，默认为Size(8, 8)
    clahe = cv.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
    dst = clahe.apply(gray)
    cv.imshow("equalhist_demo",dst)

#直方图的应用
def create_rgb_hist(image):
    h, w, c = image.shape
    # 创建一个（16*16*16,1）的初始矩阵，作为直方图矩阵
    # 16*16*16的意思为三通道每通道有16个bins
    rgbhist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            # 人为构建直方图矩阵的索引，该索引是通过每一个像素点的三通道值进行构建
            index = int(b / bsize) * 16 * 16 + int(g / bsize) * 16 + int(r / bsize)
           	# 该处形成的矩阵即为直方图矩阵
            rgbhist[int(index), 0] += 1
    return rgbhist


def hist_compare(image1, image2):
	# 创建第一幅图的rgb三通道直方图（直方图矩阵）
    hist1 = create_rgb_hist(image1)
    # 创建第二幅图的rgb三通道直方图（直方图矩阵）
    hist2 = create_rgb_hist(image2)
    # 进行三种方式的直方图比较
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离：%s, 相关性：%s, 卡方：%s" %(match1, match2, match3))


src = cv.imread ("D:/aaaaaa.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
cv.imshow("input image",src)
image_hist(src)
cv.waitKey(0)
cv.destroyAllWindows()