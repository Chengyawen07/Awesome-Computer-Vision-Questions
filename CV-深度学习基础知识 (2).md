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







## 🚀 **8. BN（Batch Normalization）的作用**

### 🎯**关键作用**

1. <u>**BN 通过 mini-batch 计算均值和方差，对输入数据进行归一化**，减少内部特征偏移，提高模型稳定性。</u>
2. **BN 可以加快训练速度，提高模型的泛化能力，并减少梯度消失问题。**
3. **在 PyTorch 中，`nn.BatchNorm1d()` 适用于全连接层，`nn.BatchNorm2d()` 适用于 CNN**。
4. **在小 batch 任务（如 RNN）中，LayerNorm 比 BatchNorm 更合适！**



💡 **举个生活中的例子**：

- 假设你每天的**生活节奏不稳定**（有时候 7 点起床，有时候 10 点起床），你的学习效率就会受影响。
- 但如果你每天**固定 7:30 起床**、**8:00 开始学习**，你的学习状态会更稳定，提高效率。
- **BN 就是在神经网络训练过程中，<u>调整每层输入的分布，使它更加稳定**，就像调整生活作息一样。</u>

**因此：**

- <u>The role of BN is to "standardize" the input data</u> of each layer, keep it stable, and improve training efficiency!
- **BN 的作用就是让每一层的输入数据“标准化”，保持稳定，提高训练效率！**



### 🎯**用 PyTorch 实现 BN**

BN 在 PyTorch 里很简单，我们直接用 `nn.BatchNorm1d()` 或 `nn.BatchNorm2d()`：

 **BN 用于 CNN（二维卷积）**

```python
import torch
import torch.nn as nn

# 假设输入数据形状为 (batch_size=8, channels=16, height=32, width=32)
input_tensor = torch.randn(8, 16, 32, 32)

# 定义 BN 层（适用于 CNN）
bn = nn.BatchNorm2d(16)  # 16 表示输入通道数

# 经过 BN 层
output_tensor = bn(input_tensor)

print(output_tensor.shape)  # 仍然是 torch.Size([8, 16, 32, 32])
```

- 为什么 `BatchNorm2d(16)` 只填 `16`？
  - 因为 CNN 处理的是**通道维度（Channel）**，BN 会对 `16` 个通道分别计算均值和方差。



### 🎯 面试时如何回答 BN?

面试官：**“你能解释一下 Batch Normalization 吗？”**

🔥 **最佳回答（简短清晰）**：

> Batch Normalization（BN）是一种用于神经网络的归一化方法，它的作用是：
>
> 1️⃣ **减少数据分布的变化（减少 Internal Covariate Shift）**，让每一层的输入更加稳定；
>
> 2️⃣ **加快训练速度，提高梯度稳定性**； 
>
> 3️⃣ **减少梯度消失和梯度爆炸**，特别适用于深度神经网络。 
>
> 在 PyTorch 中，我们可以用 `nn.BatchNorm2d()` 处理 CNN 任务，`nn.BatchNorm1d()` 处理全连接任务。

🔥 **如果考官继续问：** **“BN 为什么能加速训练？”**

> **因为 BN 让数据归一化，使得神经网络的梯度更加稳定，这样学习率可以设得更大，提高收敛速度。**

🔥 **如果考官问：** **“BN 适用于 RNN 吗？”**

> **不适合！因为 RNN 处理序列数据，mini-batch 计算均值/方差会导致时间步不一致，所以 RNN 一般用 LayerNorm。**

PS：

- Mini-Batch 训练是一种优化方法，它比全批量梯度下降更快，比随机梯度下降更稳定。
- 在 PyTorch 里，我们用 `DataLoader` 进行 Mini-Batch 训练，batch_size 影响训练效果，一般选择 `32~512` 之间，以提高训练速度和稳定性。



## **BN vs. 其他归一化**

| **方法**               | **适用场景**               | **特点**                             |
| ---------------------- | -------------------------- | ------------------------------------ |
| **BatchNorm（BN）**    | CNN / 全连接层             | 依赖 mini-batch，训练更快            |
| **LayerNorm（LN）**    | RNN / NLP                  | 对每个样本单独归一化，适用于小 batch |
| **InstanceNorm（IN）** | 风格迁移（Style Transfer） | 对每个样本的通道归一化               |
| **GroupNorm（GN）**    | 小 batch CNN               | 适用于小 batch 任务                  |







## 🚀 **9. 神经网络训练过程中，我们一般会调整哪些超参数？**

神经网络训练时，我们通常需要调整**超参数（Hyperparameters）**，以提高模型性能。这些超参数主要包括：

- **优化器相关**（学习率、批量大小）
- **模型结构相关**（隐藏层、激活函数、正则化）
- **训练策略相关**（训练轮数、早停策略）



## 1.**主要超参数调整**

| **类别**         | **超参数**                      | **作用**                       |
| ---------------- | ------------------------------- | ------------------------------ |
| **优化器相关**   | **学习率（learning rate）**     | 控制参数更新步长，影响收敛速度 |
|                  | **批量大小（batch size）**      | 影响梯度计算稳定性             |
|                  | **优化算法（SGD/Adam）**        | 影响梯度更新方式               |
| **模型结构相关** | **隐藏层数量**                  | 影响模型复杂度                 |
|                  | **隐藏单元数（神经元个数）**    | 影响模型表达能力               |
|                  | **激活函数（ReLU/Sigmoid）**    | 影响梯度传播                   |
|                  | **正则化（Dropout/L2 正则化）** | 影响防止过拟合                 |
| **训练策略相关** | **训练轮数（Epochs）**          | 影响训练充分程度               |
|                  | **早停（Early Stopping）**      | 防止过拟合                     |



##  **2. 常见参数设置**

| **超参数**                 | **作用**         | **推荐值**                                  |
| -------------------------- | ---------------- | ------------------------------------------- |
| **学习率（lr）**           | 控制参数更新大小 | `0.001`（Adam），`0.01`（SGD）              |
| **批量大小（batch size）** | 影响训练稳定性   | `32~512`                                    |
| **优化器**                 | 控制梯度更新方式 | `Adam`（推荐）                              |
| **隐藏层数**               | 影响模型复杂度   | `1~5 层`（简单任务），`5~10 层`（复杂任务） |
| **激活函数**               | 控制非线性变换   | `ReLU`（推荐）                              |
| **正则化**                 | 防止过拟合       | `Dropout = 0.5`                             |
| **训练轮数（Epochs）**     | 控制训练充分性   | `10~100`                                    |

------



## 🚀 **10. Batch Size 对 Loss 影响**

### **1. Batch Size 是什么？**

Batch Size 指的是**每次训练时，用于计算梯度和更新参数的样本数量**。

- **小 Batch（如 16~64）**：每次用较少样本更新参数。
- **大 Batch（如 512~4096）**：每次用更多样本计算梯度。

Batch Size 会直接影响**损失（Loss）\**的\**稳定性、收敛速度和泛化能力**。

### **2. 影响**

- **小 Batch（32~64）** → 适合**不规则数据，泛化能力更强**。
- **中等 Batch（128~512）** → **推荐使用**，在稳定性和速度之间平衡。
- **大 Batch（1024+）** → 适合**超大数据集，计算效率高**，但容易过拟合。

### **代码示例**

```python
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

------

总结：

- Batch Size 影响 Loss 的稳定性和收敛速度。
- 小 Batch 使 Loss 震荡较大，但有更好的泛化能力；大 Batch 让 Loss 更平滑，但可能导致过拟合。
- 在实际训练中，Batch Size = 128~512 是最常见的选择，因为它在计算效率和模型泛化之间取得平衡。



## 🚀 **11. Dropout 为什么能防止过拟合**

### **什么是 Dropout？**

Dropout 让神经网络的**某些神经元随机失活（置 0）**，让模型更健壮：

- **训练时**：<u>随机丢弃一部分神经元（防止依赖特定神经元）。</u>**Randomly drop a portion of neurons (to prevent dependence on specific neurons)**
- **测试时**：所有神经元都工作，但权重会调整。

### **代码示例**

```python
import torch.nn as nn

dropout = nn.Dropout(p=0.5)  # 50% 概率丢弃
output = dropout(input_tensor)
```

------



## **12. 什么是感受野（Receptive Field）**

感受野（Receptive Field, RF）指的是**神经网络中某一层的神经元，在输入图像上能“看到”的区域大小**。

- 在 **卷积神经网络（CNN）** 中，感受野是指**某个神经元在输入层上的感知范围**。
- **感受野越大**，神经元能看到的图像区域越大，学习到的特征越全。



### 🎯 **例子**

- 小感受野（只看近处）：
  - 你的视线只能看**手机屏幕上的一个字**，很难理解整篇文章。
- 大感受野（视野更大）：
  - 你的视线能看整行文字，甚至整篇文章，理解就更容易。

- 在 CNN 中，小感受野只能捕捉细节（比如边缘），而大感受野可以捕捉整体结构（比如人脸）。

### **🎯 计算**

假设：

- 第一层 Conv：**3×3 卷积，步长 1**，感受野 = 3×3
- 第二层 Conv：**3×3 卷积，步长 1**，感受野 = 5×5
- 第三层 Conv：**3×3 卷积，步长 1**，感受野 = 7×7



### 🎯 如何扩大感受野？

| **方法**                                | **作用**                         | **原理**                                 |
| --------------------------------------- | -------------------------------- | ---------------------------------------- |
| **增加卷积层数**                        | 逐层累积感受野                   | 每层卷积的感受野是前一层的扩展           |
| **使用池化层（Pooling）**               | 直接减少特征图大小               | 池化层下采样，使感受野指数增长           |
| **使用更大步长（Stride）**              | 加速感受野扩展                   | 卷积或池化时跳跃计算                     |
| **使用空洞卷积（Dilated Convolution）** | 在不增加计算量的情况下扩大感受野 | 在卷积核中插入空洞，使卷积核覆盖更大区域 |
| **使用全局池化（Global Pooling）**      | 让感受野覆盖整个特征图           | 直接输出全局特征，减少计算量             |



### **🎯 **代码示例 -- Pooling

- **池化（Pooling）** 通过**下采样（Subsampling）**减少特征图的大小，使得每个神经元的输入范围增大。
- **最大池化（Max Pooling）** 和 **平均池化（Average Pooling）** 都能扩大感受野。

```python
pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
```

原理：

`Max Pooling`：在池化窗口内，**取最大值**（保留最重要特征）。

`Average Pooling`：在池化窗口内，**取平均值**（平滑特征）。

作用：

- **降低数据维度**，减少计算量，提高计算效率。
- **扩大感受野**，让 CNN 的每个神经元看到更大的图像区域。
- **提取更关键的特征**，减少模型对不重要细节的关注，提高泛化能力。

