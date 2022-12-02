## 1、转换 onnx模型
yolov5仓库地址（下载v5_6.1版本）： https://github.com/ultralytics/yolov5
### 1) 配置环境
# onnx>=1.9.0  # ONNX export
# onnx-simplifier>=0.3.6  

### 2) export.py 导出 onnx
* --data data/coco128.yaml  # 该参数修改为自己的数据配置文件路径
* --weights ./yolov5s.pt  # 该参数修改为自己的模型路径
python export.py --data data/coco128.yaml --weights ./yolov5s.pt --simplify --include onnx  --device 0


### 3) 可视化onnx 
工具网址： https://netron.app

输出维度： box（x_center，y_center，width，height） + box_score + 类别信息



## 2、下载封装代码并修改
	gitee仓库地址：https://gitee.com/cvmart/ev_sdk_demo4.0_pedestrian_intrusion_yolov5.git
	cp -r ev_sdk_demo4.0_pedestrian_intrusion_yolov5/* ./ev_sdk/

### 1）修改配置文件
 - config/algo_config.json #与训练时配置文件编号顺序对应
	"mark_text_en": ["vehicle", "plate"],
    	"mark_text_zh": ["车辆","车牌"], 
 -  src/Configuration.hpp 
	std::map<std::string, std::vector<std::string> > targetRectTextMap = { {"en",{"vehicle", "plate"}}, {"zh", {"车辆","车牌"}}};// 检测目标框顶部文字

### 2）修改模型路径 src/SampleAlgorithm.cpp
 mDetector->Init("/usr/local/ev_sdk/model/yolov5s.onnx", mConfig.algoConfig.thresh);

### 3）置信度修改
-  
 * 修改src/SampleDetector.cpp 的nms阈值  
	runNms(DetObjs, 0.45);  # 0.45为nms阈值
 * 修改置信度阈值
	config/algo_config.json 和 src/Configuration.hpp 中的 thresh
### 4）报警逻辑修改
 -  src/Configuration.hpp  // 在 struct Configuration 中定义报警类型 {1,2,3}为报警类别对应的类别编号
    std::vector<int> alarmType = {1,2,3};

- src/SampleAlgorithm.cpp  //修改报警逻辑 
  将  //过滤出行人 下方代码替换为如下代码（）
   
    for(auto iter = detectedObjects.begin(); iter != detectedObjects.end();)
    {
        SDKLOG_FIRST_N(INFO, 5) << "iter->label : " << iter->label;
        if(find(mConfig.alarmType.begin(), mConfig.alarmType.end(), iter->label) != mConfig.alarmType.end())
        {
            iter++;
        }
        else
        {
            iter = detectedObjects.erase(iter);
        }
    }
    

## 3、编译测试

### 1）编译
  - 编译SDK库
      mkdir -p /usr/local/ev_sdk/build
      cd /usr/local/ev_sdk/build
      cmake ..
      make install 
  - 编译测试工具
      mkdir -p /usr/local/ev_sdk/test/build
      cd /usr/local/ev_sdk/test/build
      cmake ..
      make install 

### 2）测试
  - 输入单张图片，需要指定输入输出文件
    　　/usr/local/ev_sdk/bin/test-ji-api -f 1 -i /usr/local/ev_sdk/data/vp.jpeg -o result.jpg

## 4、提交封装测试
改好模型目录
models/exp/weights/best.onnx

/usr/local/ev_sdk/model/exp/weights/best.onnx










