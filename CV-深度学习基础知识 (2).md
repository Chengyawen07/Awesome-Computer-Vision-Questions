# CV-深度学习基础知识 (2)



## 🚀 **6. CNN 典型层次（Layer）及其作用**

卷积神经网络（CNN, Convolutional Neural Network）主要用于**图像处理**任务，它的结构由多个**不同的层（Layer）**组成，每一层有不同的作用，主要包括：

1. **输入层（Input Layer）**
   1. 图像通常表示为 **多维矩阵（Tensor）**
2. <u>**卷积层（Convolutional Layer）**</u>
3. <u>**激活函数层（Activation Layer，例如 ReLU）**</u>
4. <u>**池化层（Pooling Layer，例如 Max Pooling）**</u>
5. **归一化层（Normalization Layer，例如 Batch Normalization）**
6. <u>**全连接层（Fully Connected Layer）**</u>
7. **Softmax 层（用于分类）**
   1. Softmax 层是**神经网络中的输出层**，**适用于多分类任务（mutually exclusive classes）**。
   2. 🚫 **不适用于二分类任务**，二分类通常用 `sigmoid` 代替 `softmax`
   3. **它的作用是：**
      1. **Softmax 把 logits 转换成概率**，适用于 **多分类任务**。
      2. **每个类别的概率加起来等于 1**，方便决策。The probabilities for each class add up to 1


### **📌 关键的层**

CNN 由多个层组成，每一层负责不同的任务，如**特征提取、降维、分类等**。

### **🔹 1. 卷积层（Convolutional Layer）**

**作用**：

- <u>使用**卷积核（filters）**提取特征，如边缘、角点、纹理。</u>
- 每个卷积核负责识别不同的模式，比如猫的耳朵、眼睛等。

**代码示例**

```python
import torch.nn as nn

conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
```

------



### **🔹 2. 激活层（Activation Layer）**

**作用**：

- 引入非线性，提高模型表达能力。
- 解决线性变换的局限性，常用函数：
  - **ReLU**（常见）: 解决梯度消失问题（让负数变为 0，保留正数。）
    - **ReLU（Rectified Linear Unit）**
  - **Sigmoid**（少用）: 适合二分类
  - **Tanh**（少用）: 适合小数据集

**代码示例**：

**作用**：

- 解决**梯度消失问题**（Sigmoid 和 Tanh 会导致梯度消失）。
- **加速收敛**，减少计算量。

```python
activation = nn.ReLU()
activated_output = activation(output)
```

------



### **🔹 3. 池化层（Pooling Layer）**

**作用**：

- **降低数据维度**，减少计算量，防止过拟合。
- 过滤不重要的信息，增强特征的鲁棒性。

**常见池化方式**：

- **最大池化（Max-Pooling）**: 选取局部最大值，保留关键特征。
- **均值池化（Average-Pooling）**: 计算区域均值，平滑特征。

**代码示例**

```python
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

------



### **🔹 4. 全连接层（Fully Connected Layer, FC）**

**作用**：

- 通过神经元进行最终分类。
- 将卷积层提取的特征转换成具体类别。
- **输入是拉平（Flatten）的特征**，转换成一维向量。

**代码示例**

```python
fc = nn.Linear(in_features=128, out_features=10)  # 128 个输入特征，输出 10 类
```

------

### CNN 典型结构示例

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 个类别
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

```

## **Conv 总结**

| **层**                    | **作用**                 |
| ------------------------- | ------------------------ |
| **输入层**                | 读取图片数据             |
| **卷积层**                | 提取特征（边缘、纹理）   |
| **激活函数（ReLU）**      | 让模型学习非线性特征     |
| **池化层（Max Pooling）** | 降低数据维度，减少计算量 |
| **归一化层（BatchNorm）** | 加速收敛，提高稳定性     |
| **全连接层**              | 负责分类                 |
| **Softmax**               | 输出分类概率             |





## 🚀 **7. 传统机器学习 vs. 深度学习的选择**

#### **🔹 什么时候用机器学习？**

1. **数据量小（< 10万条）**：深度学习通常需要大量数据，而 ML 对小数据更友好。
2. **特征明显**：如果数据本身特征明显（如信用评分、医疗数据），ML 的效果可能更好。
3. **计算资源有限**：如果没有 GPU 或云计算资源，ML 训练成本更低。
4. **需要解释性**：如金融、医疗等领域，业务需要理解模型的决策过程，ML 更容易解释。

💡 **适合的算法**：

- **回归（Regression）**：房价预测、市场趋势分析
- **决策树（Decision Tree）**、**XGBoost**：金融风控、信用评分
- **随机森林（Random Forest）**：医疗诊断、欺诈检测
- **SVM（支持向量机）**：文本分类、小数据任务
- **KNN（K-最近邻）**：推荐系统



#### **🔹 什么时候用深度学习？**

1. **数据量大（>10万条）**：深度学习需要大量数据才能学到有效特征。
2. **数据是非结构化的（图片、文本、音频、视频）：**
   - 传统 ML **无法直接处理图片、文本**，DL 在 CNN、RNN、Transformer 等模型下表现更好。
3. 任务复杂，**需要自动特征学习：**
   - 例如人脸识别、自动驾驶，人工设计特征太难，DL 直接学习特征更有效。
4. 对预测精度要求极高：
   - 语音识别、机器翻译、目标检测等，DL 比传统 ML 更准确。

💡 **适合的算法**：

- **CNN（卷积神经网络）**：图像分类、目标检测（YOLO, Faster R-CNN）
- **RNN/LSTM/GRU**：自然语言处理（机器翻译、文本生成）
- **Transformer（BERT, GPT）**：NLP 任务（ChatGPT）
- **GAN（生成对抗网络）**：图像生成（DeepFake）
- **强化学习（Reinforcement Learning）**：自动驾驶、机器人控制

📝 **示例**：

- **自动驾驶（目标检测、语义分割）** ✅ 选择 **CNN / YOLO**
- **语音助手（语音识别、对话）** ✅ 选择 **RNN / Transformer**
- **医学影像分析（CT 诊断）** ✅ 选择 **CNN**

**示例代码（使用传统 ML 和 DL 解决问题）**

```python
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn

# 传统机器学习
rf = RandomForestClassifier(n_estimators=100)

# 深度学习（适用于大数据）
deep_model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.Softmax(dim=1)
)
```

### **结论**

| **标准**     | **选择传统 ML**      | **选择深度学习**             |
| ------------ | -------------------- | ---------------------------- |
| **数据量**   | 小数据 (<10万条)     | 大数据 (>10万条)             |
| **数据类型** | 结构化（表格、数值） | 非结构化（图像、文本、视频） |
| **特征工程** | 需要人工特征选择     | 自动学习特征                 |
| **计算资源** | 低，CPU 运行即可     | 高，需要 GPU                 |
| **可解释性** | 强，业务可理解       | 黑盒，难解释                 |
| **应用场景** | 银行、医疗、信用评分 | 语音识别、自动驾驶、NLP      |



