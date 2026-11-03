通过训练课堂行为检测数据集建立 基于YOLOv8的课堂学生行为识别检测系统
@[toc]
学生课堂行为数据集。
以下文字及代码仅供参考。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/19ada5531c5843c0b13c427328660938.png)

基于YOLOV8的课堂学生行为识别检测系统 基于深度学习的课堂学生行为识别检测系统
构建流程
软件：Pycharm+Anaconda
环境：python=3.9 opencv-python PyQt5 torch1.9


 支持图片、视频及摄像头进行检测: 界面可实时显示目标位置、目标总数、置信度等信息: 支持检测结果保存;
①选择图片/视频/摄像头
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5df112103d3940fca6d0f20190d8a8f3.png)

数据包含6类学生课堂常见行为：低头、手放在桌子下面、环顾四周、正常、站立、举手。数据集划分好训练集（2111张图片）验证集（528张图片）原图，

基于YOLOv8的课堂学生行为识别检测系统是一个综合性的项目，它结合了深度学习模型、图形用户界面（GUI）和多媒体处理功能。

如何构建这样一个系统的详细步骤和代码示例。

### 1. 环境搭建

确保你的开发环境已经安装了必要的库和工具。

#### 安装依赖

```bash
# 创建并激活虚拟环境
conda create -n student_behavior_detection python=3.9
conda activate student_behavior_detection

# 安装必要的库
pip install opencv-python PyQt5 torch==1.9.0 torchvision ultralytics
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf3b060e4fb649899a331d8b38c5de0a.png)

### 2. 模型加载与推理

假设你已经有了一个训练好的YOLOv8模型，我们可以直接使用它进行推理。

#### 加载模型

```python
from ultralytics import YOLO

def load_model(model_path):
    model = YOLO(model_path)
    return model

model_path = 'path/to/your/best.pt'  # 替换为你的最佳模型路径
model = load_model(model_path)
```

### 3. 图片检测

编写一个函数来处理图片检测，并返回检测结果。

#### 图片检测函数

```python
import cv2

def detect_image(model, image_path):
    results = model(image_path)
    annotated_frame = results[0].plot()
    return annotated_frame, results[0].boxes.data.tolist()

image_path = 'path/to/test_image.jpg'
annotated_frame, detection_results = detect_image(model, image_path)
cv2.imshow('Student Behavior Detection', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 4. 视频检测

编写一个函数来处理视频检测，并实时显示检测结果。

#### 视频检测函数

```python
def detect_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        annotated_frame = results[0].plot()
        
        cv2.imshow('Student Behavior Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

video_path = 'path/to/test_video.mp4'
detect_video(model, video_path)
```

### 5. 摄像头检测

编写一个函数来处理摄像头输入，并实时显示检测结果。

#### 摄像头检测函数

```python
def detect_camera(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        annotated_frame = results[0].plot()
        
        cv2.imshow('Student Behavior Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

detect_camera(model)
```

### 6. GUI设计与实现

使用PyQt5创建一个简单的图形用户界面，用于选择图片、视频或摄像头进行检测。

#### UI设计文件（.ui）

你可以使用Qt Designer来设计UI，并将其保存为`.ui`文件。然后使用`pyuic5`将`.ui`文件转换为Python代码。

```bash
pyuic5 -x your_design.ui -o ui_design.py
```

#### 主程序文件

```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from ui_design import Ui_MainWindow
import cv2

class StudentBehaviorDetection(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.actionOpen_Image.triggered.connect(self.open_image)
        self.actionOpen_Video.triggered.connect(self.open_video)
        self.actionCamera.triggered.connect(self.open_camera)

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg)")
        if file_name:
            annotated_frame, _ = detect_image(model, file_name)
            cv2.imshow('Student Behavior Detection', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi)")
        if file_name:
            detect_video(model, file_name)

    def open_camera(self):
        detect_camera(model)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StudentBehaviorDetection()
    window.show()
    sys.exit(app.exec_())
```

### 总结

基于YOLOv8的课堂学生行为识别检测系统的完整实现步骤和代码示例。通过这些步骤，你可以构建一个支持图片、视频及摄像头输入的学生行为检测系统，并通过图形用户界面进行交互。根据实际需求调整上述代码段，仅供参考
