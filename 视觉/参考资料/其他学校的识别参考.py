#-*-coding:UTF-8-*-
import cv2
import math
import numpy as np
import serial
import time
import threading

"""
read_txt 共四个参数，后面各跟一个英文逗号
第一个参数  c:小车程序	w:大风车程序
第二个参数  b:蓝色	r:红色
第三个参数  跳帧
第四个参数  延迟时间 （只有大风车需要加，用来确定预测点位置）
若 read_txt 第一个参数为f则退出程序
"""
global read_txt
#read_txt="c,b,1"
read_txt="w,b,1,3,"

"""
data_ary板子性能不足，用来补足帧
"""
#global data_ary
#data_ary=np.zeros((3,2),"int")

"""
 point 存储最终打击点或预测点信息
 	格式：w:x,y 或 c:x,y
 	如：w:100,120
 		c:320:311
"""
global point
point = ""

#小车识别主方法
def car_main(video,color_flg,skip):
    if (video.isOpened()):
        result, frame = video.read()
        skip_num=0
        while result:
            result, frame = video.read()
            if frame is not None:
                skip_num+=1
                if skip_num % skip==0:
                    #print(frame.shape)
                    #frame = cv2.resize(frame,(1200,676)) 
                    find_lamp(frame,color_flg)
                    print(point)
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)
    else:
        print('Fail to open!')
    cv2.destroyAllWindows()#关闭窗口
def find_lamp(image,color_flg):
    # 根据灰度筛选
    #blur = cv2.GaussianBlur(image, (3, 3), 0)
    image_b, image_g, image_r = cv2.split(image)
    #cv2.imshow('windows', binary)
    # cv2.waitKey(1)

    if (color_flg == 'b'):
        retval, binary = cv2.threshold(image_b, 210, 255, cv2.THRESH_BINARY)
        # 根据颜色筛选
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # HSV BLUE
        lower = np.array([100, 43, 46], dtype="uint8")   # [70, 0, 250]
        upper = np.array([124, 255, 255], dtype="uint8")  # [150, 80, 255]
        hsv_mask = cv2.inRange(hsv_image, lower, upper)

        # 白色
        lower = np.array([0, 0, 221], dtype="uint8")    # 240, 240, 225
        upper = np.array([180, 30, 255], dtype="uint8")
        mask_white = cv2.inRange(hsv_image, lower, upper)
        
    else:
        # 根据颜色筛选
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # HSV 红色
        lower = np.array([0, 43, 46], dtype="uint8")  # 175, 205, 170
        upper = np.array([10, 255, 255], dtype="uint8")  # 40 ,155, 255
        hsv_mask = cv2.inRange(hsv_image, lower, upper)
        # 白色
        lower = np.array([0, 0, 221], dtype="uint8")    # 240, 240, 225
        upper = np.array([180, 30, 255], dtype="uint8")
        mask_white = cv2.inRange(hsv_image, lower, upper)
    kernel = np.ones((2,1),np.uint8)
    opening = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5,5),np.uint8)
    dilate_white = cv2.dilate(opening,kernel)

    kernel = np.ones((10,10),np.uint8)
    closing = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3,2),np.uint8)
    opening_color = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


    finish_img = cv2.bitwise_and(opening_color, dilate_white)
    kernel = np.ones((4,4),np.uint8)
    finish_img = cv2.erode(finish_img,kernel)
    (_, contours,_) = cv2.findContours(finish_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rect1 = []
    rect2 = []

    min_bias = 10000
    for i in range(len(contours)):
        area1 = math.fabs(cv2.contourArea(contours[i]))
        rect = cv2.minAreaRect(contours[i])
        x=rect[1][0]
        y=rect[1][1]
        
        if area1 >= 3 and area1 <= 200 and len(contours[i]) >= 5 and max([x,y])/min([x,y])>1.2:
            for j in range(i + 1, len(contours)):
                area2 = math.fabs(cv2.contourArea(contours[j]))
                rect = cv2.minAreaRect(contours[j])
                x2=rect[1][0]
                y2=rect[1][1]
                if area2 >= 3 and area2 <= 200 and len(contours[j]) >= 5 and max([x2,y2])/min([x,y])>1.2:
                    difference = math.fabs(area1 - area2) * 0.8
                    distance = ellipse_distance(contours[i], contours[j]) * 30 # 300
                    angle = ellipse_angle(contours[i], contours[j]) * 3
                    
                    bias = difference + distance  + angle
                    if (bias < min_bias  and distance >=0):
                        min_bias = bias
                        rect1 = contours[i]
                        rect2 = contours[j]
    
    if len(rect1) > 0 and len(rect2) > 0:
        largest_rect(image, rect1, rect2)

def largest_rect(image, rect1, rect2):
    x1, y1, w1, h1 = cv2.boundingRect(rect1)
    x2, y2, w2, h2 = cv2.boundingRect(rect2)
    if abs(x1 - x2) > abs(y1 - y2):
        if x1 < x2:
            x = x1
            y = y1
            w = x2 + w2 - x
            h = y2 + h2 - y
        else:
            x = x2
            y = y2
            w = x1 + w1 - x
            h = y1 + h1 - y
        y -= h // 2
        h *= 2
    else:
        if y1 < y2:
            x = x1
            y = y1
            w = x2 + w2 - x
            h = y2 + h2 - y
        else:
            x = x2
            y = y2
            w = x1 + w1 - x
            h = y1 + h1 - y
        x -= w // 2
        w *= 2

    x = int((x + w / 2))
    y = int((y + h / 2))
    w = int(w)
    h = int(h)
    
    #data_ary[:2]=data_ary[1:]
    #data_ary[-1]=np.array([x,y])
    global point
    point = "c:"+str(x)+","+str(y)+"\n"
    
    cv2.rectangle(image, (int(x-w/2), int(y-h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
    cv2.circle(image, (int(x), int(y )), 2, (0, 255, 0), 2)

def ellipse_distance(contours1, contours2):
    '''求两个灯之间的差距比例跟预设比例2.4的差距 3.5'''
    center1, size1, angle1 = cv2.fitEllipse(contours1)
    center2, size2, angle2 = cv2.fitEllipse(contours2)
    height = (size1[1] + size2[1]) / 2
    distance = abs(center1[0] - center2[0])
    result= math.fabs(distance / height - 2.4)
    if result<1:
        return result if result >= 0.3 else 0
    else:
        return -1
def ellipse_angle(contours1, contours2):
    '''求两灯角度差距'''
    center1, size1, angle1 = cv2.fitEllipse(contours1)
    center2, size2, angle2 = cv2.fitEllipse(contours2)
    angle1 = angle1 if angle1 <= 90 else  math.fabs(angle1 - 180)
    angle2 = angle2 if angle2 <= 90 else math.fabs(angle2 - 180)
    result = math.fabs(angle1 - angle2)
    return result if result >= 5 else 0
    
    
########大风车识别程序######

#图像处理
def get_good_img(frame,min_hsv,max_hsv):
    #转变成hsv
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #hsv颜色范围
    lower_hsv = np.array(min_hsv,np.uint8)
    upper_hsv = np.array(max_hsv,np.uint8)
    #获取mask掩模
    mask = cv2.inRange(hsv,lower_hsv,upper_hsv)
    #图像开闭运算处理，获取较好二值图
    kernel = np.ones((4,4),np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closing
   
#获取打击中心点
def get_center_point(box,wheel_center_point,center_point):
    dis_ary = np.zeros(4)
    for num in range(len(box)) :
        dis_dis = np.linalg.norm(box[num] - center_point)
        dis_ary[num]=dis_dis
    sort_dis_ary = np.sort(dis_ary)
    first_point_idx = np.where(dis_ary==sort_dis_ary[-1])[0][0]
    second_point_idx = np.where(dis_ary==sort_dis_ary[-2])[0][0]
    #获取到外侧两个点
    first_point = box[first_point_idx]
    second_point = box[second_point_idx]
    #计算出打击中心点
    f_point = np.int0(np.mean(np.array([wheel_center_point,first_point,second_point]),0))
    return f_point

#获取被打击臂极其信息
def get_wheel_main(wheel_cnt,wheel_center,get_num,contours,center_point,y):
    max_idx=0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)  # 最小外接矩形
        areas = rect[1][1]*rect[1][0]
        if areas>15 and areas<100 and abs(rect[1][1]-rect[1][0]) < min([rect[1][1],rect[1][0]]) and rect[0][1]<y/3:#近似正方形 且 位于图像上半部分（因为反光问题）
            center_point = np.array(rect[0])
        if areas>500 and rect[0][1]<y/3 :#面积大于5000 且 位于图像上半部分（因为反光问题）
            # 矩形的四个角点取整
            box = np.int0(cv2.boxPoints(rect)) 
            wheel_dis = np.zeros((5),dtype = 'int64')
            
            for i in range(len(wheel_center)):
                diff_dis = np.linalg.norm(wheel_center[i]-np.array(rect[0]))
                wheel_dis[i]=diff_dis

            min_dis_idx = np.where(wheel_dis==np.min(wheel_dis))[0][0]
            if max_idx<min_dis_idx:
                max_idx=min_dis_idx

            if np.min(wheel_dis)>50:
                if get_num <5:
                    wheel_cnt[get_num]=box
                    wheel_center[get_num] = np.array(rect[0])
                    get_num+=1
                else:
                    break#已经找完五个，退出
            else:
                wheel_cnt[min_dis_idx]=box
                wheel_center[min_dis_idx] = np.array(rect[0])
                
    return wheel_cnt,wheel_center,get_num,np.int0(center_point),max_idx

#获取预测点的位置
def get_anticipation_point(origin_point,delay_time,center_point):
    #旋转半径
    r = np.linalg.norm(origin_point-center_point)
    #原打击点在平面中的角度
    origin_agl = math.acos((origin_point[0]-center_point[0])/r)
    #位于负y轴时角度为2π-原角度
    if origin_point[1]<center_point[1]:
        origin_agl=2*math.pi-origin_agl
    #旋转角速度
    anticipation_agl =origin_agl+36*math.pi/180*delay_time
    #预测x点
    anticipation_x = center_point[0]+r*math.cos(anticipation_agl)
    #预测y点
    anticipation_y = center_point[1]+r*math.sin(anticipation_agl) 
    #预测点
    anticipation_point = [int(anticipation_x),int(anticipation_y)]
    return anticipation_point
    
#主运行函数
def wheel_main(cap,color_flg,delay_time,skip):
    #数据初始化
    wheel_cnt=np.zeros((5,4,2),dtype = 'int64')
    wheel_xy=np.zeros((5,2),dtype = 'int64')
    wheel_center = np.zeros((5,2),dtype = 'int64')
    get_num=0
    center_point=np.zeros(2,dtype = 'int64')
    min_color=max_color=max_color=min_color=list()
    skep_num=0#与skip合用进行跳帧
    tnum=0
    while(True):
        
#         if ser.inWaiting()>0:
#             read_txt = ser.readline(100)
#             if len(read_txt) !=0:
#                 break

        #判断用那种颜色
        if color_flg=="r":
            min_color = [0,90,90]
            max_color = [20,255,255]
        else:
            min_color = [90,150,150]
            max_color = [115,255,255]
        ret, frame = cap.read()
        
        if ret==False:
            break
        
        skep_num+=1
        if skep_num %skip == 0:#跳帧用
            frame = cv2.resize(frame,(600,450)) #600,338
            tnum+=1
            #图像处理
            img = get_good_img(frame,min_color,max_color)
            opening, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #获取被打击臂极其信息
            wheel_cnt,wheel_center,get_num,center_point,max_idx = get_wheel_main(wheel_cnt,wheel_center,get_num,contours,center_point,opening.shape[0])
            #获取打击点
            origin_point = get_center_point(wheel_cnt[max_idx],wheel_center[max_idx],center_point)
            #根据延迟时间获取预测点
            try:
                anticipation_point = get_anticipation_point(origin_point,delay_time,center_point)
            except:
            	anticipation_point = [0,0]
            #print(anticipation_point)
            anticipation_point_ser="w:"+str(anticipation_point[0])+","+str(anticipation_point[1])+"\n"
            point = anticipation_point_ser
            #画红 绿点
            cv2.circle(frame, tuple(origin_point), 1, (0, 255, 0), 4)
            cv2.circle(frame, tuple(anticipation_point), 1, (0, 0, 255), 4)
            
            #显示图像
            cv2.imshow('frame',frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()#关闭窗口

class CreateThread (threading.Thread):
    def __init__(self, threadID, name, defName):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.defName = defName
    def run(self):
            print ("***开始线程：" + self.name+"***")
            self.defName()
            print ("***退出线程：" + self.name+"***")

#识别程序线程入口
def run():
    while read_txt[0]!="f":#read_text为全局变量，在数据监听线程中，可以改变其值，当为f时退出
        time.sleep(1)
        runCarOrWheel()
        
#数据监听线程入口
def listenReadText():
    global read_txt
    while read_txt[0]!="f":
        read_txt = input("改变read_txt：")#read_text为全局变量，可以改变其值，进行小车/大风车，颜色，跳帧，延迟时间的实时改变。
        time.sleep(1)
#根据输入内容判断是运行哪个识别程序
def runCarOrWheel():
    if len(read_txt) !=0 and (str(read_txt[0]) == 'w' or str(read_txt[0]) =='c'):
        read_txt_list = read_txt.split(",")
        car_or_wheel = read_txt_list[0]
        color_flg = read_txt_list[1]
        skip = int(read_txt_list[2])
        if car_or_wheel == "c":
            video = cv2.VideoCapture("3.mp4")#"test6.MOV"
            car_main(video,color_flg,skip)#参数:视频，红蓝, 跳帧
        elif car_or_wheel == "w":
            delay_time=float(read_txt_list[3])
            video = cv2.VideoCapture("blue_0.mp4")
            wheel_main(video,color_flg,delay_time,skip)#参数:视频，红蓝，延迟时间/秒, 跳帧
        else:
            if(ser.isOpen()):
                ser.write("first parameter is c or w")
# def writeData(d):
#     while True:
#         time.sleep(0.033)
#         #data_ary=np.array([[1,2],[320,4],[1,2]])
#         #new_data_ary  =  (data_ary[:,0]**2+data_ary[:,1]**2)/(data_ary[:,0]+1)
#         return_data=""
#         for  i in range(len(data_ary)):
#             for  j in range(1,len(data_ary)):
#                 if((data_ary[i]==data_ary[j]).all()):
#                     return_data+=str(data_ary[i][0])+","+str(data_ary[i][1])+"\r\n"
#                     break
#             if return_data!="":
#                 break
#         if return_data=="":
#             return_data+=str(data_ary[-1][0])+","+str(data_ary[-1][1])+"\r\n"
        
# 创建识别线程 和 数据监听线程
threadRun = CreateThread(1, "run", run)
threadListenReadText = CreateThread(2, "listenReadText", listenReadText)

# 开启识别线程threadRun 和 数据监听线程 threadListenReadText
threadRun.start()
threadListenReadText.start()
threadRun.join()
threadListenReadText.join()
print ("退出主线程")