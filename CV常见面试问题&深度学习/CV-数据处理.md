# Computer Vision高频面试题 - 常见数据处理和特征工程方法



## **1. 常用的数据清洗方法**

**面试考察点：**

- **数据清理在机器学习和计算机视觉任务中的重要性。**
- **如何处理缺失值？**
- **如何处理异常值？**
- **能否使用合适的 Python 代码实现这些方法？**

------

### **1.1 处理缺失值的方法**

| **方法**                                  | **核心思路**             | **适用场景**                     |
| ----------------------------------------- | ------------------------ | -------------------------------- |
| **删除缺失值**                            | 直接删除包含缺失值的样本 | 当缺失值较少且数据量充足时       |
| **均值填补法**                            | 用该列的均值填补缺失值   | 适用于数据服从正态分布           |
| **中位数填补法**                          | 用该列的中位数填补缺失值 | 适用于数据受极端值影响较大的情况 |
| **众数填补法**                            | 用该列的众数填补缺失值   | 适用于分类变量                   |
| **热卡填补法（KNN 或 机器学习模型填充）** | 通过相似数据填充缺失值   | 适用于高维数据或结构化数据       |

------

### **1.2 处理异常值的方法**

| **方法**              | **核心思路**                            | **适用场景**                             |
| --------------------- | --------------------------------------- | ---------------------------------------- |
| **统计分析法**        | 计算最大最小值，判断是否合理            | 适用于数据范围固定的情况（如年龄、温度） |
| **3σ 原则**           | 计算均值 ± 3 * 标准差范围外的值为异常值 | 适用于正态分布数据                       |
| **箱型图（IQR）分析** | 计算四分位距（IQR）筛选异常值           | 适用于数据非正态分布的情况               |
| **基于模型检测**      | 训练分类/回归模型识别异常值             | 适用于异常模式较明显的数据               |
| **基于距离检测**      | 计算数据点之间的欧几里得距离            | 适用于密度均匀的聚类数据                 |
| **基于密度检测**      | DBSCAN、LOF 计算密度差异                | 适用于数据密度变化较大的情况             |

------

### **💡 代码示例**

#### **📌 1. 缺失值处理**

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# 创建示例数据
data = {
    "A": [1, 2, None, 4, 5],
    "B": [None, 2, 3, None, 5],
    "C": [10, 15, 20, 25, None]
}
df = pd.DataFrame(data)

# 1. 删除缺失值
df_dropped = df.dropna()

# 2. 均值填补
imputer_mean = SimpleImputer(strategy="mean")
df_filled = pd.DataFrame(imputer_mean.fit_transform(df), columns=df.columns)

print("原始数据：\n", df)
print("删除缺失值后的数据：\n", df_dropped)
print("填充缺失值后的数据：\n", df_filled)
```



#### **📌 2. 异常值处理（箱型图/IQR 方法）**

```python
import numpy as np

# 计算 IQR
Q1 = df["C"].quantile(0.25)
Q3 = df["C"].quantile(0.75)
IQR = Q3 - Q1

# 确定异常值的范围
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 过滤异常值
df_no_outliers = df[(df["C"] >= lower_bound) & (df["C"] <= upper_bound)]
print("去除异常值后的数据：\n", df_no_outliers)
```

------

## **🎯 总结**

### **📌 数据清洗**

- 缺失值处理：

  - **删除缺失值**：适用于数据量大且缺失比例低的情况。
  - **均值/中位数填补**：适用于数值型数据。
  - **机器学习填补**：适用于高维数据。

- 异常值处理：

  - **3σ 原则**：适用于正态分布数据。
  - **IQR 箱型图**：适用于非正态分布数据。
  - **基于模型/聚类的方法**：适用于复杂数据。

  

## **2. 目标检测中的数据增强方法**

**面试考察点：**

- **数据增强对于目标检测任务的必要性。**
- **能否正确处理 Bounding Box 的变化？**
- **如何使用 Python 代码实现数据增强？**

------

### **2.1 目标检测的常见数据增强方法**

| **方法**                                 | **影响**           | **Bounding Box 处理**         |
| ---------------------------------------- | ------------------ | ----------------------------- |
| **裁剪 (Crop)**                          | 放大或缩小目标区域 | 需要重新计算目标框            |
| **平移 (Translation)**                   | 使目标偏移一定像素 | 目标框随图像移动              |
| **旋转 (Rotation)**                      | 目标倾斜           | 需旋转目标框                  |
| **镜像 (Flip - Horizontal/Vertical)**    | 左右翻转目标       | 水平翻转时需调整目标框 X 坐标 |
| **颜色变化 (Brightness, Contrast, Hue)** | 影响视觉感知       | 不影响目标框                  |
| **添加噪声 (Gaussian Noise)**            | 使模型更鲁棒       | 不影响目标框                  |

------

### **OpenCV 介绍**

OpenCV 是一个用于 **计算机视觉（Computer Vision）和图像处理** 的开源库，支持多种编程语言（Python、C++、Java 等），主要用于：

- 图像处理（滤波、边缘检测、直方图均衡化等）
- 计算机视觉（目标检测、物体跟踪、人脸识别等）
- 视频处理（视频读取、写入、帧处理等）
- pip install opencv-python



### 数据增强：**Albumentations vs OpenCV**

- `Albumentations` 是一个 **高效** 的 **数据增强库**，用于对图像和目标框（Bounding Boxes）进行变换。
- 使用 **Albumentations** 进行 **数据增强（Data Augmentation）**，主要用于 **计算机视觉任务**，特别是**目标检测（Object Detection）和分类**。
- 好处：
  - **更快**：比 OpenCV、PIL 更快（C++ 优化）。
  - **易用**：几行代码即可实现复杂的数据增强。
  - **支持目标检测**：支持 `Pascal VOC`、`COCO`、`YOLO` 格式的目标框变换。
  - **支持语义分割**：可对 **Mask** 进行相同变换。



### **💡 代码示例**

#### 基本数据增强方法：1. 随机翻转（水平/垂直）

```Python
import cv2
import numpy as np

image = cv2.imread("image.jpg")

# 水平翻转
h_flip = cv2.flip(image, 1)

# 垂直翻转
v_flip = cv2.flip(image, 0)

cv2.imshow("Horizontal Flip", h_flip)
cv2.imshow("Vertical Flip", v_flip)
cv2.waitKey(0)
cv2.destroyAllWindows()

```



#### **📌 2. 2D 目标检测中的数据增强**

```python
import albumentations as A
import cv2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5)
], bbox_params=A.BboxParams(format="pascal_voc"))

image = cv2.imread("image.jpg")
bboxes = [[50, 60, 200, 300]]  # (xmin, ymin, xmax, ymax)

transformed = transform(image=image, bboxes=bboxes)

```

在 OpenCV (`cv2`) 中，常用于 **数据增强（Data Augmentation）** 的函数如下

| 方法         | OpenCV 代码                 | 适用于          |
| ------------ | --------------------------- | --------------- |
| **翻转**     | `cv2.flip()`                | 分类 / 目标检测 |
| **旋转**     | `cv2.getRotationMatrix2D()` | 分类 / OCR      |
| **缩放**     | `cv2.resize()`              | 分类 / 目标检测 |
| **裁剪**     | `image[y:y+h, x:x+w]`       | 目标检测        |
| **亮度调整** | `cv2.convertScaleAbs()`     | 分类 / 目标检测 |
| **噪声**     | `np.random.normal()`        | 低光环境增强    |
| **颜色变换** | `cv2.COLOR_BGR2HSV`         | 目标检测        |



## 补充： 3D目标检测

3D 目标检测主要基于 **点云（Point Cloud）** 或 **RGB-D 图像**，通常使用 **激光雷达（LiDAR）、RGB-D 摄像头、毫米波雷达** 等传感器。数据增强需要同时作用于 **3D 点云（Point Cloud）** 和 **3D 边界框（Bounding Box）**。

### **1、3D 目标检测增强方法**

| 方法                           | 作用               | 适用场景         |
| ------------------------------ | ------------------ | ---------------- |
| **点云翻转（Flip）**           | 反转 X/Y/Z 轴      | 适应对称性场景   |
| **点云旋转（Rotation）**       | 旋转点云 & 3D BBox | 适应多角度目标   |
| **点云缩放（Scaling）**        | 放大/缩小点云      | 适应不同尺寸目标 |
| **点云抖动（Jittering）**      | 添加少量随机噪声   | 增强鲁棒性       |
| **点云裁剪（Cropping）**       | 选取局部点云       | 适应局部信息     |
| **高斯扰动（Gaussian Noise）** | 模拟传感器误差     | 低质量点云适应性 |
| **Dropout（随机丢弃点）**      | 使点云稀疏化       | 适应低质量数据   |
| **MixUp & CutMix 3D**          | 组合多个点云       | 适应不同类别     |
| **背景替换**                   | 替换点云背景       | 模拟真实环境     |

### **📌 推荐工具**

1. **Open3D**（常用于点云处理）

2. **mmdetection3d**（支持多种增强策略）

3. **PyTorch3D**（高级 3D 变换）

4. **PointAugment（深度学习自适应增强）**

5. **SECOND & OpenPCDet（LiDAR 3D 检测）**

   

✅ 3D: Open3D 旋转

```Python
import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud("point_cloud.pcd")

# 旋转点云
R = pcd.get_rotation_matrix_from_xyz((0, np.pi/4, 0))  # 旋转45度
pcd.rotate(R, center=(0, 0, 0))

o3d.visualization.draw_geometries([pcd])

```



### **2、机器人视觉（Robotic Perception）**

机器人通常使用 **多模态数据（RGB + 深度 + 点云）**，所以数据增强通常需要**同步**作用于不同模态数据。

### **📌 常见增强方法**

- **同步增强（RGB + 点云）**（保证相机和 LiDAR 传感器数据一致）
- **随机遮挡（Occlusion Augmentation）**（模拟遮挡环境）
- **环境模拟（Domain Adaptation）**（室内/室外不同光照）
- **传感器噪声模拟（Sensor Noise Simulation）**（激光雷达点云误差、相机光照变化）

### **📌 推荐工具**

1. **ROS & Gazebo**（机器人仿真环境）
2. **NVIDIA Isaac Sim**（真实物理仿真）



✅ 机器人目标检测: OpenCV + 深度图

```Python
import cv2
import numpy as np

depth_image = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)

# 归一化深度图
depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

# 颜色映射
depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

cv2.imshow("Depth Image", depth_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()

```



## **总结**

| 目标            | 常用增强方法                    | 推荐工具                         |
| --------------- | ------------------------------- | -------------------------------- |
| **2D 目标检测** | 翻转、旋转、缩放、Mosaic、MixUp | OpenCV, Albumentations, YOLO     |
| **3D 目标检测** | 点云旋转、缩放、噪声、随机遮挡  | Open3D, PyTorch3D, mmdetection3d |
| **机器人视觉**  | RGB+深度同步增强、环境模拟      | ROS, Isaac Sim, AirSim           |

- **Albumentations**（适合 2D）
- **Open3D & PyTorch3D**（适合 3D）
- **ROS/Gazebo**（适合机器人）

💡 **如果你的任务是 2D 目标检测，推荐** `Albumentations`
💡 **如果是 3D 点云，推荐** `Open3D`
💡 **如果是机器人仿真，推荐** `ROS & Gazebo`



## **3. 为什么要对特征做归一化？常用的归一化方法有哪些？**

### **面试考察点：**

- **理解归一化的作用及其对模型训练的影响**
- **不同归一化方法的适用场景**
- **如何在实际数据处理中应用归一化（代码实现）**

------

### **3.1 为什么要做归一化？**

**归一化（Normalization）** 是一种数据预处理方法，用于调整数据的数值范围，使其更适合机器学习模型。主要作用包括：

- **加快模型训练**（减少数值差异带来的梯度问题）
- **提升模型性能**（避免某些特征占主导）
- **增强数值稳定性**（防止数据溢出）

#### **1. 提高梯度下降的收敛速度**

- 归一化可以让所有特征在同一数量级上，从而**避免某些特征对梯度影响过大**，提升训练效率。
- 如果特征值差异很大（比如一个特征是 0-1，另一个是 0-1000），未归一化会导致权重更新速度不均衡。

#### **2. <u>避免数值过大或过小影响模型性能</u>**

- **数据分布不均衡** 可能会导致某些特征在计算时影响过大或过小，影响神经网络的训练。
- **神经网络中，未经归一化的数据可能导致梯度爆炸或梯度消失**。

#### **3. 让模型具有更好的泛化能力**

- 训练集和测试集的数据分布不同会影响模型泛化能力，归一化可以减少这种影响。

------

### **3.2 常用的归一化方法**

| **方法**                              | **计算公式**                | **适用场景**                                                 |
| ------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| **Min-Max 归一化（Min-Max Scaling）** | x′=x−min(x)/max(x)−min(x)x' | 适用于 **数据范围已知**，需要将数据映射到 0-1 之间的情况（如 CNN 输入）。 |
| **Z-score 标准化（标准分数）**        | x′=x−μ/σ                    | 适用于数据服从 **正态分布** 的情况，如 Logistic Regression、SVM。 |
| **Log 归一化**                        | x′=log(x+1)                 | 适用于数据分布偏斜（例如收入、点击次数等）。                 |
| **Robust 归一化（基于中位数和 IQR）** | x′=x−median(x)/IQR          | 适用于有**极端值**的数据，受异常值影响较小。                 |

------



### **3.3 代码示例**

#### **📌 Min-Max 归一化**

📌 **适用场景**：图像像素归一化、加速梯度下降

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 示例数据
data = np.array([[100], [200], [300], [400], [500]])
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print("Min-Max 归一化结果：\n", scaled_data)
```



#### **📌 Z-score 标准化**

📌 **适用场景**：当数据有**不同单位**或**存在异常值**时，比如**神经网络输入**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print("Z-score 标准化结果：\n", scaled_data)
```

------

### 总结：

**📌 选择归一化方法的关键**：

- **深度学习** → 用 **Min-Max 归一化**
- **统计学习** → 用 **Z-score 标准化**
- **文本/特征向量** → 用 **L2 归一化**



