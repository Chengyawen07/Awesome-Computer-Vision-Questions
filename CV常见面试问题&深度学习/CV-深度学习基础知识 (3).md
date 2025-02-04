# CV-深度学习基础知识 (3)

## **📌 13. 为什么先用 Adam，然后再用 SGD？**

✅ **原理**

- **SGD（随机梯度下降）**：每次更新参数时，只用一个小批量（Mini-Batch）的样本计算梯度，并沿着梯度方向下降。
  - lr固定
  - 计算简单，效率高

- **Adam（Adaptive Moment Estimation）**：是**SGD 的升级版**，它结合了**Momentum（动量）和 RMSprop（自适应学习率）**，能**自适应调整学习率**，加速收敛
  - **自适应调整学习率，不同参数有不同的学习率**
  - 能加速收敛





✅ **Adam 收敛快，但最终效果不如 SGD**

- **Adam** 是自适应优化算法，前期收敛速度**比 SGD 快**，适合快速找到一个较优的解。
- 但在训练后期，**Adam 的学习率会变得很小**，可能会卡在次优解附近，无法继续优化。



✅ **SGD 适合精调，让模型收敛到最优解**

- **SGD** 在训练后期可以更稳定地找到最优解，泛化能力更强。
- 所以可以**先用 Adam 快速收敛**，然后**切换到 SGD 进行精调**，获得更好的最终效果。

🔥 **PyTorch 代码示例**

```python
import torch.optim as optim

# 先使用 Adam 训练
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 后期换 SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```



### ✅ **什么时候用 Adam，什么时候用 SGD？**

| **情况**                    | **推荐算法** | **原因**                                         |
| --------------------------- | ------------ | ------------------------------------------------ |
| **前期快速收敛**            | **Adam**     | Adam 适合大数据集，能自适应调整学习率，加速训练  |
| **后期精调**                | **SGD**      | SGD 在训练后期，能更稳定找到最优解，泛化能力更强 |
| **小数据集**                | **SGD**      | Adam 的自适应特性在小数据集上可能没有优势        |
| **NLP 任务（Transformer）** | **Adam**     | NLP 需要快速训练，Adam 效果更好                  |



## **📌 14. 什么是周期学习率（Cyclical Learning Rate, CLR）？**

✅ **学习率是超参数，影响训练效果**

- 学习率太大 → 训练不稳定，难以收敛
- 学习率太小 → 训练速度慢，容易卡住

✅ **周期学习率（CLR）的作用**

- <u>让学习率**在一定范围内周期性变化**，而不是固定的。</u>
- 这样可以**跳出局部最优点，加速收敛**，提高模型泛化能力。

🔥 **CLR 公式**：

- Learning Rate= Min LR + (Max LR−Min LR) × f(cycle position)

其中 `f(cycle position)` 可以是 **三角波、余弦波等**。

🔥 **PyTorch 代码实现**

```python
from torch.optim.lr_scheduler import CyclicLR

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=2000, mode='triangular')
```

------



## **📌 15. 如何判断神经网络过拟合？怎么缓解？**

✅ **如何判断过拟合？**

1. **训练误差下降，但测试误差变大** ✅
2. **学习曲线**：训练 Loss **持续下降**，但验证 Loss **先下降后上升**
3. <u>**训练数据准确率高（95%+），但测试数据准确率低**</u>



✅ **缓解过拟合的方法**

| **方法**                          | **作用**                                     |
| --------------------------------- | -------------------------------------------- |
| **增加训练数据**                  | 让模型学到更普遍的特征                       |
| **数据增强（Data Augmentation）** | 在图像任务中扩充数据，如旋转、镜像           |
| **使用正则化（L1/L2）**           | 限制参数大小，防止模型过拟合                 |
| **减少特征数**                    | 移除不相关特征，提高泛化能力                 |
| **调整超参数**                    | 降低学习率，增加 Batch Size                  |
| **降低模型复杂度**                | 减少神经元数，减少层数                       |
| **使用 Dropout**                  | 随机丢弃部分神经元                           |
| **提前停止（Early Stopping）**    | 训练时如果验证 Loss 不再下降，则提前停止训练 |

🔥 **PyTorch 代码（L2 正则化 + Dropout）**

```python
import torch.nn as nn
import torch.optim as optim

# 使用 Dropout
dropout_layer = nn.Dropout(p=0.5)

# 在 PyTorch 里，我们用 `weight_decay` 来控制 L2 正则化
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
```



✅ **L2 正则化（Weight Decay）**

- 在深度学习中，**L2 正则化**（也称为**权重衰减 Weight Decay**）的作用是**限制模型参数的大小**，防止过拟合。

- 也就是在Loss的函数后，加上一个正则项。比如Loss = MSE + L2正则

- 原理：

  - **模型过拟合 = 过度依赖某些特征，导致权重变得特别大**。
  - **L2 正则化**会**让权重变小**，防止网络学习到过于复杂的模式，增强泛化能力。

- 在 PyTorch 里，我们用 `weight_decay` 来控制 L2 正则化。

- 🔥 **适用于 SGD, Adam, RMSprop 等优化器**

  - optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

  

- 

## **📌 16. 全卷积神经网络（FCN）的优点？**

✅ **全卷积神经网络（FCN, Fully Convolutional Network）**

- FCN 主要用于**图像分割任务（Semantic Segmentation）**，与传统 CNN 相比，它能在**像素级别**做预测，不仅仅是分类。

✅ **全卷积网络（FCN）特点**：

1. **端到端预测**：输入图像 → 输出像素级预测，不需要额外的分类步骤。
   1. 直接输入原始图像，网络自动学习特征，并输出像素级分类

2. **能处理任意大小的输入图像**
   1. **CNN 需要固定大小的输入图像**（如 224×224），而 FCN 可以接受**任意大小的图像**，并直接输出相同尺寸的预测结果。
   2. 传统 CNN 需要**全连接层（FC）**，但 FCN 移除了 FC 层，换成了**卷积层**，这样就可以处理任意大小的输入。

3. **适用于图像分割**：如医学图像、自动驾驶感知任务。
   1. **CNN 只能输出一个类别标签**（如“猫”或“狗”），但 FCN 能**预测每个像素的类别**（如每个像素属于“猫”或“背景”）。
   2. 适用于**语义分割（Semantic Segmentation）\**任务，比如\**自动驾驶、医学影像分析**

4. 使用跳跃连接（Skip Connection），保留细节信息
   1. 由于卷积 + 池化会丢失细节，FCN **使用跳跃连接（Skip Connection）**，将**低级特征与高级特征融合**，提高分割精度。




✅ **FCN 适用于哪些任务？**

| **任务**         | **应用场景**         |
| ---------------- | -------------------- |
| **图像语义分割** | 目标检测、自动驾驶   |
| **医学影像分析** | 肿瘤检测、X-ray 分析 |
| **场景理解**     | 智能安防、无人机视觉 |

🔥 **PyTorch 代码（使用 FCN 进行语义分割）**

```python
import torchvision.models.segmentation as segmentation

# 预训练的 FCN 模型
model = segmentation.fcn_resnet50(pretrained=True)
```

------





## **📌 17. 1×1 卷积的作用**

✅ **1×1 卷积（pointwise convolution）**

- 1×1 卷积的作用本质上是**调整通道数（Channel）**，也就是**改变特征图（Feature Map）的深度维度**，但不改变特征图的空间尺寸（Height × Width）。
- 这里的 **"维度" 指的是通道数（Channels）**，而不是图像的宽度（Width）或高度（Height）。



**1. 主要用于 降维、升维。**

- **减少通道数**，降低计算量，加快推理速度。
- **示例**：
  - **输入通道数：512**
  - **使用 1×1 卷积降维到 128**
  - **输出通道数变小，但 H×W 不变**

```Python
import torch.nn as nn

conv1x1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)  # 降维
```



 **2. 增强特征组合能力（特征融合）**

- **1×1 卷积能重新组合特征通道信息**，类似于神经网络的**全连接层**，但计算量更低。
- 允许不同通道间的信息交互，**提升模型的表达能力**。



✅ **1×1 卷积的核心作用**

| **作用**                  | **解释**                             |
| ------------------------- | ------------------------------------ |
| **升/降维（通道数调整）** | 改变特征图的通道数，提高计算效率     |
| **增加网络深度**          | 增强非线性映射能力，提高学习能力     |
| **组合特征**              | 让不同通道的信息交互，学习更复杂特征 |

## **🚀 总结**

| **问题**                      | **答案总结**                                                |
| ----------------------------- | ----------------------------------------------------------- |
| **13. Adam → SGD**            | Adam 收敛快但效果一般，后期用 SGD 精调模型                  |
| **14. 周期学习率（CLR）**     | 让学习率周期性变化，加速收敛，提高泛化                      |
| **15. 过拟合及解决方案**      | 数据增强、正则化（L2, Dropout）、减少特征数、Early Stopping |
| **16. 全卷积神经网络（FCN）** | 端到端像素级预测，适用于图像分割                            |
| **17. 1×1 卷积的作用**        | 调整通道数、增加非线性、组合特征                            |

------



## **📌 18. 梯度下降法如果陷入局部最优怎么办？**

梯度下降法可能会陷入局部最优（Local Minimum），特别是在**非凸损失函数**的情况下。以下是解决方案：

**✅ 1. 增大学习率（Learning Rate）**

- <u>学习率（Step Size）影响模型更新的幅度：</u>
  - **太小**：容易被卡在局部最优。
  - **太大**：可能会错过最优点，在最优解附近震荡。

🔥 **调整学习率的方法**

- **指数衰减学习率（Exponential Decay）**
- **周期性学习率（Cyclical Learning Rate, CLR）**

```python
import torch.optim.lr_scheduler as lr_scheduler

scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 每 10 轮学习率降低 10 倍
```



**✅ 2. 每个 Epoch 洗牌数据（Shuffling Data）**

- **原因**：如果每次训练都用相同的 batch 数据，可能会陷入局部最优。
- **解决方法**：每个 epoch 后，打乱训练数据，确保每个 batch 的样本不同。

🔥 **PyTorch 实现**

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # 开启 shuffle
```



**✅ 3. 设置不同的参数初始化**

- **原因**：不良的权重初始化可能导致模型卡在局部最优。

- 解决方法

  ：尝试不同的权重初始化方法，例如：

  - **Xavier（Glorot）初始化**
  - **Kaiming（He）初始化**

🔥 **PyTorch 代码**

```python
import torch.nn.init as init

init.xavier_uniform_(model.fc.weight)  # Xavier 初始化
init.kaiming_uniform_(model.conv.weight, nonlinearity='relu')  # Kaiming 初始化
```



**✅ 4. 使用更好的优化算法**

- **Momentum 梯度下降**：让梯度下降有“惯性”，避免停滞在局部最优。
- **Adam（Adaptive Moment Estimation）**：自适应调整学习率，提高收敛能力。

🔥 **PyTorch 代码**

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Momentum
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam
```



## **📌 19. 神经网络中的梯度消失（Vanishing Gradient）和梯度爆炸（Exploding Gradient）**

在深度神经网络中，反向传播过程中梯度会逐层传播，如果层数**过深**，梯度可能会：

- **越来越小（梯度消失）** ⏬ → 训练变慢，甚至停止更新。
  - **在深层神经网络中，越靠近输入层，梯度就越接近 0**，导致**前面的层学不到东西**，网络几乎不更新

- **越来越大（梯度爆炸）** ⏫ → 训练不稳定，模型崩溃。
  - **在深度网络中，梯度会变得越来越大，导致参数更新过猛，网络不稳定**。
  - 学习率太大




**通俗理解** 梯度就像“坡度”：

- 如果梯度大 → 表示“坡很陡”，参数更新幅度大，容易震荡。
- 如果梯度小 → 表示“坡很平”，参数更新幅度小，训练慢甚至停止。



**✅ 1. 为什么会发生梯度消失？**

- <u>主要发生在**使用 Sigmoid 或 Tanh 激活函数**的深层网络。</u>
- **Sigmoid 函数的梯度范围：** σ′(x)=σ(x)(1−σ(x))\sigma'(x) = \sigma(x) (1 - \sigma(x)) 由于**最大梯度值只有 0.25**，<u>当网络层数增加时，梯度会**不断变小**，导致权重更新变慢，甚至完全停止。</u>

🔥 **解决方案** ✅ **使用 ReLU 替代 Sigmoid**

```python
import torch.nn.functional as F

output = F.relu(input_tensor)
```



✅ **使用 Batch Normalization（BN）**

```python
import torch.nn as nn

bn_layer = nn.BatchNorm1d(num_features=128)
```



✅ **使用 LSTM 代替普通 RNN**

```python
rnn = nn.LSTM(input_size=100, hidden_size=128, num_layers=2)
```



**✅ 2. 为什么会发生梯度爆炸？**

- 主要发生在**层数过深**的网络中，梯度在反向传播时指数增长，最终变得极大，导致模型崩溃。

🔥 **解决方案** 

✅ **使用梯度裁剪（Gradient Clipping）**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 设定梯度最大值
```

✅ **使用合适的权重初始化**

```python
init.kaiming_uniform_(model.fc.weight, nonlinearity='relu')  # Kaiming 初始化
```



**🔥  总结**

| **问题**     | **原因**                                 | **影响**               | **解决方案**                 |
| ------------ | ---------------------------------------- | ---------------------- | ---------------------------- |
| **梯度消失** | 深度网络中梯度不断变小，前面层学不到东西 | 训练变慢或无法学习     | 用 ReLU, BatchNorm, ResNet   |
| **梯度爆炸** | 梯度太大，参数更新过猛，训练不稳定       | 损失变成 NaN，训练失败 | 用梯度裁剪、Xavier/He 初始化 |



## **📌 20. 卷积神经网络（CNN）中的 Padding 作用**

<u>在卷积操作中，**如果不使用 Padding，每次卷积后特征图尺寸会缩小**，最终可能丢失重要信息。</u>
 ✅ **Padding 的主要作用：**

1. **保持输入输出尺寸一致**
2. **防止边界信息丢失**
3. **让 CNN 更容易训练不同尺寸的图片**



**✅ 1. 如果不加 Padding**

- 输入尺寸：`5x5`

- `3x3` 卷积核，步长 `1`

- **输出尺寸变成 `3x3`**

  **原图**

  ```
  5x5 → 3x3 → 1x1（最终特征图很小）
  ```

**🔥 PyTorch 代码（无 Padding）**

```python
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
```



**✅ 2. 如果加上 Padding**

- 输入尺寸：`5x5`

- `3x3` 卷积核，步长 `1`，**Padding=1**

- **输出尺寸仍然是 `5x5`** ✅

  **保持尺寸**

  ```
  5x5 → 5x5 → 5x5
  ```

**🔥 PyTorch 代码（有 Padding）**

```python
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
```

------



## **🚀 总结**

| **问题**                   | **答案总结**                                             |
| -------------------------- | -------------------------------------------------------- |
| **18. 梯度下降局部最优**   | 调整学习率、数据洗牌、初始化不同参数、换优化器           |
| **19. 梯度消失和梯度爆炸** | 深层网络引发梯度变小或变大，解决方法：ReLU、BN、梯度裁剪 |
| **20. CNN 中的 Padding**   | 让特征图尺寸保持一致，避免边界信息丢失，提高模型泛化能力 |

- **梯度下降可能陷入局部最优，我们可以调整学习率、打乱数据、换初始化方式或使用 Momentum 优化器。**

- **梯度消失通常发生在深层网络，尤其是使用 Sigmoid 时，我们可以用 ReLU、BN 或 LSTM 解决。**
- **梯度爆炸是梯度过大导致的不稳定问题，可以用梯度裁剪或权重初始化来缓解。**
- **在 CNN 中，Padding 作用是防止特征图尺寸缩小，保护边界信息，提高模型的训练稳定性。**



## **📌 21. 为什么使用 SqueezeNet？SqueezeNet 是怎样的网络？**

<u>SqueezeNet 是 **一种轻量级 CNN 结构**，目标是**减少模型参数量，同时保持较高的识别精度**，适用于**资源受限的设备**（如移动设备、嵌入式系统）。</u>



**🔹 1. 为什么使用 SqueezeNet？**

在深度学习模型中，常见的 CNN（如 AlexNet、VGG）都**参数量巨大，占用存储空间大，计算量高**，不适合在**手机、嵌入式设备**上运行。

SqueezeNet 就是为了解决这个问题，它的目标是： 

✅ **减少模型大小（降低存储需求）**
✅ **减少计算量（加快推理速度）**
✅ **保持高准确率（接近 AlexNet 级别）**



