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

  

