# Computer Vision 高频面试题 - 评价指标

1. 什么是 ROC 曲线？如何计算 AUC？
2. 目标检测中的 mAP 是如何计算的？
3. 召回率 (Recall) 和精准率 (Precision) 的计算公式是什么？
4. IoU（Intersection over Union）是什么？如何计算？mIoU 又是什么？
5. 为什么目标检测问题中通常使用 AUC 而不是召回率和精准率？



### **1. 什么是 ROC 曲线？如何通过 ROC 计算 AUC？**

#### **核心要点**

- **ROC（Receiver Operating Characteristic）曲线** 反映了分类器的性能，通过调整阈值来绘制 **真正类率（TPR）** 和 **假正类率（FPR）**。
  - ROC 曲线是 **衡量分类模型性能** 的方法，特别是在<u>二分类问题（比如检测是否是目标物体）中很常用</u>。它的横轴和纵轴分别代表：
- **TPR（真正类率）** = TP/{TP+FN}（召回率）
- **FPR（假正类率）** = FP/{FP+TN}（1 - 特异度）
- <u>**AUC（Area Under Curve）**是 ROC 曲线下的面积，表示分类器的整体能力</u>：
- **结论：**
  - AUC = 1：完美分类器
  - AUC = 0.5：模型和随机猜测一样，没有区分能力
  - AUC > 0.5：比随机好，越接近 1 越好



#### **代码示例**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 真实标签（0 = 负类, 1 = 正类）
y_true = np.array([0, 0, 1, 1])
# 预测得分（分类器的置信度）
y_scores = np.array([0.1, 0.4, 0.35, 0.8])

# 计算 FPR, TPR
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # 随机分类的参考线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
```

------



### **2. 目标检测中 mAP 怎么计算？**

#### **核心要点**

- mAP（Mean Average Precision，平均精度均值） 是目标检测算法评估的核心指标，<u>它衡量 模型预测的框有多准。</u>
- **AP 计算步骤：**
  1. <u>计算 Precision（查准率） 和 Recall（查全率）</u>
     1. 目标检测模型会给每个预测框一个置信度分数（比如 90%）
     2. 设定不同的置信度阈值，每个阈值下，我们计算 **精准率（Precision）** 和 **召回率（Recall）**
     3. 以 **召回率 Recall 为横轴，精准率 Precision 为纵轴**，绘制曲线
  2. 绘制 PR 曲线（Precision-Recall）
  3. 计算 PR 曲线下的面积，即 AP（每个类别）
  4. 对所有类别 AP 取平均，得到 mAP

**通俗理解**

- <u>**AP 相当于“召回率-精准率”曲线的 AUC**</u>
- **mAP 就是多个类别 AP 的平均值，越高表示模型检测效果越好**
- **mAP = 0.8** 说明<u>模型在所有目标类别上平均能达到 80% 的精度</u>

#### **代码示例**

```python
import numpy as np
from sklearn.metrics import average_precision_score

# 真实标签 (0: 负类, 1: 正类)
y_true = np.array([0, 1, 1, 0, 1, 1, 0])
# 预测得分
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.6, 0.75, 0.05])

# 计算 AP
ap = average_precision_score(y_true, y_scores)
print(f"Average Precision (AP): {ap:.4f}")
```

------



### **3. 什么是查准率（Precision）和查全率（Recall）？**

#### **核心要点**

- <u>**查准率（Precision）**：预测为正样本的结果中，实际正确的比例</u>。 Precision=TP/TP+FP
  - **例子：**
    - 我们的目标检测模型检测 10 个目标，预测 8 个是“猫”。
    - 其中 6 个是正确的猫（TP=6），但有 2 个是错误的（FP=2）。
    - Precision = 6/(6+2)=0.75（模型预测的猫中有 75% 是真的）。
- <u>**查全率（Recall）**：所有真实正样本中，被成功预测出的比例</u>。 Recall=TP/TP+FN
  - **例子：**
    - 现实中其实有 10 只猫，但模型只正确识别了 6 只（TP=6），错过了 4 只（FN=4）。
    - Recall = 6/(6+4) = 0.6（实际的猫中，只有 60% 被正确检测）。
- 两者的权衡：
  - 高 Precision，低 Recall：模型只在非常确信时才预测目标，但容易漏掉一些目标。
  - 高 Recall，低 Precision：模型尽可能地检测目标，但容易出现误检。

#### **代码示例**

```python
from sklearn.metrics import precision_score, recall_score

# 真实标签
y_true = [0, 1, 1, 1, 0, 1, 0]
# 预测标签
y_pred = [0, 1, 1, 0, 0, 1, 1]

# 计算 Precision 和 Recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

------

### **总结**

1. <u>**ROC 曲线** 评估分类器的综合性能，**AUC 值越大模型越好**。</u>

2. <u>**mAP（目标检测）** 计算多个类别的平均检测精度，**mAP 越高，目标检测越准**。</u>

3. 精准率 Precision 和 召回率 Recall

    需要平衡：

   - **Precision 高** → 误报少（但可能漏掉目标）。
   - **Recall 高** → 目标检测全（但误报可能多）。

可以这样记住：

- **ROC-AUC** 评价分类问题。
- **mAP** 评价目标检测。
- **Precision vs Recall** 衡量检测的精准度和全面性。



### **4. 什么是 AUC？**

#### **核心要点**

- **AUC（Area Under Curve）** 代表 ROC 曲线下的面积，衡量分类器对 **正负样本的区分能力**。
- AUC 计算方式：
  1. 计算 ROC 曲线（FPR、TPR）。
  2. 计算曲线下的面积。
  3. **<u>AUC 越接近 1，分类器性能越好。</u>**

#### **代码示例**: roc_auc_score

（与 ROC 代码相同）

```python
from sklearn.metrics import roc_auc_score

# 计算 AUC
auc_value = roc_auc_score(y_true, y_scores)
print(f"AUC Score: {auc_value:.4f}")
```

------



### **总结**

| 指标                   | 公式                     | 说明                       |
| ---------------------- | ------------------------ | -------------------------- |
| **TPR (召回率)**       | TP/(TP+FN)TP / (TP + FN) | 真实正例中被正确预测的比例 |
| **FPR (假正率)**       | FP/(FP+TN)FP / (FP + TN) | 负例被错误预测为正例的比例 |
| **Precision (查准率)** | TP/(TP+FP)TP / (TP + FP) | 预测为正例中实际正确的比例 |
| **Recall (查全率)**    | TP/(TP+FN)TP / (TP + FN) | 所有正例中被正确预测的比例 |
| **AP (平均精度)**      | PR 曲线下的面积          | 衡量目标检测的单类别性能   |
| **mAP (均值平均精度)** | 各类别 AP 的均值         | 衡量目标检测的整体性能     |
| **AUC (曲线下面积)**   | ROC 曲线下的面积         | 衡量分类器的总体性能       |



### **5. 为什么不用召回率 (Recall) 和精准率 (Precision)，而是用 AUC？**

#### **1. 召回率和精准率的矛盾**

- 召回率 (Recall) = 检测出的正样本中，真正是正样本的比例：

  Recall=TPTP+FNRecall = \frac{TP}{TP + FN}

  - **高召回率**：意味着检测的目标比较全，但可能会误判一些负样本（误报多）。

- 精准率 (Precision) = 预测为正的样本中，真正是正样本的比例：

  Precision=TPTP+FPPrecision = \frac{TP}{TP + FP}

  - **高精准率**：意味着误判少，但可能会漏掉很多正样本（漏检多）。

通常，**召回率和精准率是此消彼长的**，如果我们想让模型检测尽可能多的目标（提高 Recall），通常会导致误报增多，Precision 下降；反之，如果我们希望模型更严格（提高 Precision），就会牺牲召回率。

#### **2. AUC 的优势**

<u>AUC（ROC 曲线下的面积）是一个更全面的衡量指标，因为：</u>

1. **适用于类别不均衡的情况**：如果数据集中正负样本的比例严重失衡（比如 99% 是负样本），Precision 和 Recall 会受到影响，而 AUC 可以很好地评估模型的整体表现。
2. **不依赖于单一的分类阈值**：Precision 和 Recall 依赖于特定的阈值（例如 0.5 以上才算正样本），但 AUC 计算的是所有可能阈值下的性能，因此更客观。

**通俗理解**：

- 召回率和精准率只能看某个固定阈值下的性能，比如 “把概率大于 0.5 的预测为正”。
- AUC 考虑了 **所有可能的阈值**，评估模型的整体能力，适用于各种任务，尤其是**类别不均衡时效果更佳**。

------



### **6. 什么是 IoU？mIoU 是如何计算的？**

#### **1. IoU（Intersection over Union，交并比）**

✅ **IoU 是目标检测中最常用的衡量指标**，用于**<u>评估预测框 (Prediction Box) 和真实框 (Ground Truth) 的重叠程度</u>**。

- **IoU 值范围 0~1**，越接近 1，说明预测框与真实框越接近，检测效果越好。

**示例**： 假设真实框面积是 200，预测框面积是 220，它们的重叠部分（交集）是 150：

IoU=150/(200+220-150) = 150/270 = 0.55

表示这个检测结果的准确度为 55%。

------

#### **2. mIoU（Mean IoU，平均 IoU）**

**mIoU 是计算所有类别 IoU 的平均值，**用于评估整个模型的目标检测能力。

计算方式：

1. **对每个类别计算 IoU**（假设有 N 个类别，每个类别都有自己的 IoU）。
2. **取所有类别 IoU 的均值**： mIoU=1N∑i=1NIoUimIoU = \frac{1}{N} \sum_{i=1}^{N} IoU_i

✅ **示例**： 假设检测的是 "猫"、"狗"、"鸟" 三个类别：

- IoU(猫) = 0.6
- IoU(狗) = 0.7
- IoU(鸟) = 0.8

则：

mIoU=(0.6+0.7+0.8) / 3= 0.7 

表示整体检测的准确度为 70%。

------

### **总结**

1. **为什么用 AUC 而不是 Precision / Recall？**
   - Precision 和 Recall 依赖于阈值，容易受类别不平衡影响。
   - AUC 不受单一阈值限制，适用于类别不均衡的情况。
2. **IoU 和 mIoU 是什么？**
   - <u>**IoU** 衡量单个预测框与真实框的重叠程度，越接近 1 越好。</u>
   - **mIoU** 是所有类别 IoU 的均值，衡量整个目标检测模型的性能，越高表示模型检测能力越强。

**简单记住**：

- **AUC 适合分类任务，mIoU 适合目标检测任务！**



### 使用 PyTorch 计算 IoU 代码示例：

```Python
import torch
from torchvision.ops import box_iou

# 预测框（格式 [x1, y1, x2, y2]）
pred_boxes = torch.tensor([
    [50, 50, 200, 200],
    [30, 30, 180, 180],
    [100, 100, 250, 250]
], dtype=torch.float32)

# 真实框（格式 [x1, y1, x2, y2]）
gt_boxes = torch.tensor([
    [60, 60, 190, 190],
    [40, 40, 170, 170],
    [120, 120, 230, 230]
], dtype=torch.float32)

# 计算 IoU
iou_matrix = box_iou(pred_boxes, gt_boxes)

# 打印 IoU 结果
print(iou_matrix)

```

### **解释**

1. `box_iou(pred_boxes, gt_boxes)` 计算 **预测框** 与 **真实框** 之间的 IoU（交并比）。
2. `iou_matrix` 是一个 **NxN 矩阵**，其中 `iou_matrix[i][j]` 代表 **第 i 个预测框 和 第 j 个真实框的 IoU 值**。



### **目标检测 (Object Detection) 常用评价指标** 🎯

| **指标**                                   | **核心用途**                                                 |
| ------------------------------------------ | ------------------------------------------------------------ |
| **IoU（Intersection over Union，交并比）** | 计算**预测框和真实框的重叠程度**，评估单个检测框的准确性。IoU 越高，说明预测框和真实框越接近。 |
| **mIoU（Mean IoU，平均交并比）**           | 计算所有类别 IoU 的平均值，**衡量整体检测精度**。            |
| **Precision（精准率）**                    | 预测出的目标中，正确的比例。**适用于减少误报（误检）**，比如自动驾驶检测行人。 |
| **Recall（召回率）**                       | 真实目标中，模型成功检测出的比例。**适用于减少漏检**，如医疗图像检测癌症。 |
| **AP（Average Precision，平均精度）**      | 计算 Precision-Recall 曲线下的面积，**评估某一类别的检测性能**。 |
| **mAP（Mean Average Precision，平均 AP）** | 所有类别 AP 的均值，**衡量整个目标检测模型的性能**。mAP 越高，说明检测效果越好。 |

------



### **分类任务 (Classification) 常用评价指标** 🎯

| **指标**                                         | **核心用途**                                                 |
| ------------------------------------------------ | ------------------------------------------------------------ |
| **Accuracy（准确率）**                           | 计算**所有预测正确的比例**，适用于类别均衡的数据集。         |
| **Precision（精准率）**                          | 预测为正类的样本中，实际正确的比例，适用于**降低误报**（如垃圾邮件检测）。 |
| **Recall（召回率）**                             | 实际正类的样本中，成功被预测的比例，适用于**减少漏报**（如癌症检测）。 |
| **F1-score**                                     | Precision 和 Recall 的加权平均值，适用于需要**兼顾误报和漏报**的任务。 |
| **ROC（Receiver Operating Characteristic）曲线** | 评估不同阈值下模型的表现，**适用于类别不均衡的场景**。       |
| **AUC（曲线下面积）**                            | 衡量 ROC 曲线的整体性能，AUC 越接近 1，分类性能越好。适用于**评估二分类模型**。 |

------

### **总结**

- **目标检测**：主要关注 **IoU、mIoU、mAP**，以及 **Precision-Recall** 平衡。
- **分类任务**：主要关注 **Accuracy、F1-score、AUC**，适用于不同数据分布的情况。

