import cv2
import time
import math

mode = "BLUE"       # 填写敌方阵营的颜色，可以是 RED 和 BLUE
debug = False       # 一键开启调试模式

cam = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素
kernel += 1

# 定义准星位置
sightX = 330
sightY = 245

# while(True):
#
timeStamp = time.time()
#     if debug:
#         print("")
#         print("--New F:")
#     ret, img = cam.read()  # 获取一帧图像
img = cv2.imread("lcz(1).jpg")

# 图像通道分离
blueImg, greenImg, redImg = cv2.split(img)  # 分离图像的RGB通道
if mode == "BLUE":                          # 分析识别模式
    img2 = cv2.subtract(blueImg, redImg)  # B通道-R通道
else:
    img2 = cv2.subtract(redImg, blueImg)  # R通道-B通道
img2 = cv2.subtract(img2, greenImg)  # 上一步运算结果-G通道
cv2.imshow('raw2', img2)
ret, img2 = cv2.threshold(img2, 50, 255, cv2.THRESH_BINARY)  # 图像二值化处理
cv2.imshow('raw3', img2)
img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)  # 开运算，减少暗斑
# cv2.imshow('raw1', img2)
binary, contours, hierarchy = cv2.findContours(
    img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # 在原始图像上绘制轮廓以进行显示
# print(1)
n = 0
x = []
y = []
longSide = []
shortSide = [] 
angle = []  # 初始化变量
for contour in contours:
    rect = cv2.minAreaRect(contours[n])  # 获取最小包围矩形
    xCache, yCache = rect[0]  # 获取矩形的中心坐标
    widthC, heigthC = rect[1]  # 获取矩形的长宽
    angleCache = rect[2]  # 获取矩形的角度 (-90, 0]

    if heigthC > widthC:
        # if heigthC / widthC >= 2.1:  # 长宽比满足条件
        if widthC < heigthC and heigthC / widthC >= 2.1:  # 灯条是竖直放置，长宽比满足条件
            x.append(int(xCache))
            y.append(int(yCache))
            longSide.append(int(heigthC))
            # shortSide.append(widthC)
            angle.append(angleCache)
            print(1)
            if debug:
                print("·X:", xCache)
                print(" Y:", yCache)
                print(" Long:", heigthC)
                print(" Short:", widthC)
            n = n + 1  # 有效矩形计数
    else:
        if widthC / heigthC >= 2.1:  # 长宽比满足条件
            x.append(int(xCache))
            y.append(int(yCache))
            longSide.append(int(widthC))
            # shortSide.append(heigthC)
            angle.append(angleCache + 90)
            print("x")
            if debug:
                print("·X:", xCache)
                print(" Y:", yCache)
                print(" Long:", widthC)
                print(" Short:", heigthC)
            n = n + 1  # 有效矩形计数
print(n)
target = []  # 存储配对的两个灯条的编号 (L1, L2)
locX = []
locY = []  # 存储计算得到的中心点坐标
dis = []  # 存储中心点与准星的距离
pairNum = 0  # 初始化计数变量
if n >= 2:  # 图像中找到两个以上的灯条
    for count in range(0, n):
        findCache = count + 1  # 初始化计数变量
        while findCache < n:  # 未超界，进行匹配运算
            calcCache = math.sqrt(
                (x[findCache] - x[count]) ** 2 + (y[findCache] - y[count]) ** 2)  # 求中心点连线长

            calcCache = calcCache / (longSide[count] + longSide[findCache])  # 求快捷计算单位
            if debug:
                print("·Scale:", calcCache)
            print(abs(angle[count] - angle[findCache]), calcCache)
            if abs(angle[count] - angle[findCache]) < 10 and (1.0 < calcCache < 1.4) and (x[findCache] - x[count]) ** 2 > (y[findCache] - y[count]) ** 2:  # 满足匹配条件

                # if abs(angle[count] - angle[findCache]) < 10 and (0.8 < calcCache < 1.2 or 1.8 < calcCache < 5.2):  #满足匹配条件
                target.append((count, findCache))
                #loc.append((int((x[count] + x[findCache]) / 2), int((y[count] + y[findCache]) / 2)))
                locX.append(int((x[count] + x[findCache]) / 2))
                locY.append(int((y[count] + y[findCache]) / 2))
                dis.append(
                    math.sqrt((locX[pairNum] - sightX) ** 2 + (locY[pairNum] - sightY) ** 2))
                cv2.circle(
                    img, (locX[pairNum], locY[pairNum]), 3, (0, 0, 255), -1)
                # 画两个圆来显示中心点的位置
                cv2.circle(
                    img, (locX[pairNum], locY[pairNum]), 8, (0, 0, 255), 2)
                print("··Group", pairNum, ":")
                if debug:
                    print("·L1:", target[pairNum][0])
                    print("·L2:", target[pairNum][1])
                print("")

                pairNum = pairNum + 1  # 计数变量自增
                break
            findCache = findCache + 1
    print(locX)
    print(locY)
    # ···在此处添加数字验证真假装甲板代码

    if pairNum != 0:
        # 寻找离准星最近的装甲板
        disCalcCache = dis[0]
        targetNum = 0  # 存储距离准星做进的装甲板编号
        for count in range(0, pairNum):
            if dis[count] < disCalcCache:
                targetNum = count
                disCalcCache = dis[count]

        if debug:
            print("···FIND TARGET !!!")
            print("·Target:", targetNum)
            print("X:", locX[targetNum])
            print("Y:", locY[targetNum])
        cv2.line(img, (locX[targetNum], 0),
                 (locX[targetNum], 480), (0, 255, 0), 1)  # 画竖线
        cv2.line(img, (0, locY[targetNum]),
                 (720, locY[targetNum]), (0, 255, 0), 1)  # 画横线
        cv2.line(img, (x[target[targetNum][0]], y[target[targetNum][0]]), (
            x[target[targetNum][1]], y[target[targetNum][1]]), (255, 255, 255), 2)  # 画连接线
        cv2.line(img, (locX[targetNum], locY[targetNum]),
                 (sightX, sightY), (255, 0, 255), 2)  # 画指向线

# 画准星
cv2.line(img, (sightX, sightY - 4),
         (sightX, sightY - 9), (50, 250, 100), 2)  # 上
cv2.line(img, (sightX, sightY + 4),
         (sightX, sightY + 9), (50, 250, 100), 2)  # 下
cv2.line(img, (sightX - 4, sightY),
         (sightX - 9, sightY), (50, 250, 100), 2)  # 左
cv2.line(img, (sightX + 4, sightY),
         (sightX + 9, sightY), (50, 250, 100), 2)  # 右

timeStamp = time.time() - timeStamp + 0.0001
fpsCalc = int(100 / timeStamp) / 100.0
cv2.putText(img2, str(timeStamp), (0, 30),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, 4)
cv2.putText(img2, str(fpsCalc), (0, 50),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, 4)
cv2.putText(img2, 'FPS MAX', (50, 50),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, 4)
cv2.imshow('raw', img)
cv2.imshow('ter', img2)  # 进行显示
cv2.waitKey(0)
cv2.destroyAllWindows()
# if cv2.waitKey(1) == ord('1'):
#     break
