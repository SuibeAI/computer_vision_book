---
marp: true
theme: default
paginate: true
math: katex
style: |
  section {
    font-size: 22px;
  }
  h1 { color: #1a5276; }
  h2 { color: #1f618d; border-bottom: 2px solid #aed6f1; padding-bottom: 4px; }
  code { background: #f0f3f4; }
  pre  { background: #f8f9fa; border-left: 4px solid #2e86c1; }
  .highlight { color: #c0392b; font-weight: bold; }
---

# 基于 U-Net 的医学影像分割

## 完整实验案例：消化内镜息肉检测

> 数据集：Kvasir-SEG &nbsp;|&nbsp; 损失函数：Dice Loss &nbsp;|&nbsp; 框架：PyTorch

---

## 目录

1. 医学影像分割背景
2. Kvasir-SEG 数据集
3. 评估指标：Dice 系数
4. 损失函数：Dice Loss
5. U-Net 模型结构
   - 基础组件：双卷积块
   - 编码器（下采样路径）
   - 瓶颈层
   - 解码器（上采样 + 跳跃连接）
6. 数据预处理流水线
7. 训练策略：学习率调度与早停
8. 实验结果可视化
9. 总结与思考

---

## 1. 医学影像分割背景

**图像分割**：将图像中每个像素分配一个类别标签。

```
输入图像 (H×W×3)  →  分割掩码 (H×W×1)
     RGB 内镜图像          0/1 二值掩码（息肉区域）
```

### 为什么医学影像分割重要？

| 应用场景 | 具体任务 |
|----------|----------|
| 消化内镜 | 息肉自动检测与定位 |
| 病理切片 | 细胞/组织区域分割 |
| CT/MRI   | 器官与肿瘤体积测量 |
| 眼底图像 | 视网膜血管分割 |

> **核心挑战**：目标形状不规则、边界模糊、前景/背景极度不平衡

---

## 2. Kvasir-SEG 数据集

### 基本信息

- **来源**：挪威 Vestre Viken 医院内镜中心
- **规模**：1,000 张消化内镜图像 + 对应像素级掩码
- **分辨率**：332×487 到 1920×1072（大小不一）
- **标注**：由 2 名经验丰富的内镜医师手工标注

### 数据组织结构

```
data/Kvasir-SEG/
├── images/          # 原始 RGB 内镜图像（.jpg）
│   ├── cju0qkwl9...jpg
│   └── ...
└── masks/           # 像素级二值掩码（.jpg，白=息肉）
    ├── cju0qkwl9...jpg
    └── ...
```

### 数据划分（对齐 DuckNet 论文协议）

采用与 DuckNet 相同的**两步划分法**（随机种子 58800）：

```python
# 第一步：分出 10% 作为测试集
trainval_paths, test_paths = train_test_split(all_paths, test_size=0.1,  random_state=58800)
# 第二步：从剩余中分出 ~10% 作为验证集
train_paths,   val_paths   = train_test_split(trainval_paths, test_size=0.111, random_state=58800)
```

最终划分：**800 训练 / 100 验证 / 100 测试**

---

## 3. 评估指标

本实验在测试集上报告与 DuckNet 论文一致的五项指标（像素级二值混淆矩阵）：

| 指标 | 公式 | 含义 |
|------|------|---------|
| **Dice** | $\frac{2\,\text{TP}}{2\,\text{TP}+\text{FP}+\text{FN}}$ | 预测与真实的重叠度，主要指标 |
| **IoU** | $\frac{\text{TP}}{\text{TP}+\text{FP}+\text{FN}}$ | Jaccard 系数，比 Dice 更严格 |
| **Precision** | $\frac{\text{TP}}{\text{TP}+\text{FP}}$ | 预测为阳性中真正阳性的比例 |
| **Recall** | $\frac{\text{TP}}{\text{TP}+\text{FN}}$ | 真实阳性被正确检出的比例 |
| **Accuracy** | $\frac{\text{TP}+\text{TN}}{\text{Total}}$ | 全像素分类正确率 |

### 为什么不单用 Accuracy？

设息肉仅占图像面积的 5%，全预测为背景的模型 Accuracy 高达 **95%**，但 Dice = **0**。

> Dice / IoU 对**类别不平衡**更鲁棒，是医学分割的首选评估指标。

```python
def compute_dice(logits, targets, threshold=0.5):
    probs = (torch.sigmoid(logits) > threshold).float()  # 概率→二值掩码
    intersection = (probs.view(-1) * targets.view(-1)).sum()
    return (2.0 * intersection) / (probs.view(-1).sum() + targets.view(-1).sum() + 1e-8)
```

---

## 4. 损失函数：Dice Loss

训练时需要一个可微分的损失函数。将 Dice 系数"软化"（使用概率而非二值）：

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \cdot \sum_i p_i g_i + \varepsilon}{\sum_i p_i + \sum_i g_i + \varepsilon}$$

- $p_i = \sigma(\text{logit}_i)$：sigmoid 输出的像素概率
- $g_i \in \{0, 1\}$：真实掩码标签
- $\varepsilon = 1.0$（平滑项）：防止分母为零，稳定梯度

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)         # 映射到 [0, 1]
        probs, targets = probs.view(-1), targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice                     # 损失 = 1 - Dice
```

> **注意**：Dice Loss 不需要预先对正负样本加权，天然处理类别不平衡。

---

## 5. U-Net 模型结构总览

U-Net（Ronneberger et al., 2015）因其**对称的编解码结构**和**跳跃连接**成为医学分割的标志性架构。

```
输入 (3×256×256)
     │
  [编码器]  enc1(64) → pool → enc2(128) → pool → enc3(256) → pool → enc4(512) → pool
                │                │                  │                  │
             跳跃连接          跳跃连接            跳跃连接           跳跃连接
                │                │                  │                  │
  [瓶颈层]               bottleneck(1024)
                                 │
  [解码器]  up4+cat → dec4(512) → up3+cat → dec3(256) → up2+cat → dec2(128) → up1+cat → dec1(64)
     │
  输出 (1×256×256)  →  sigmoid  →  预测掩码
```

**三个核心设计**：
1. **逐步下采样**：提取多尺度语义特征
2. **跳跃连接**：融合浅层细节与深层语义
3. **逐步上采样**：恢复空间分辨率

---

## 5.1 基础组件：双卷积块（double_conv）

U-Net 中每个层级的特征提取单元：

```
输入 (C_in × H × W)
   │
   ├─ Conv2d(3×3, padding=1)  ← 保持空间尺寸不变
   ├─ BatchNorm2d
   ├─ ReLU
   ├─ Conv2d(3×3, padding=1)
   ├─ BatchNorm2d
   └─ ReLU
   │
输出 (C_out × H × W)
```

```python
def double_conv(in_ch, out_ch):
    """两层 3×3 卷积 + BN + ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
```

> `padding=1` 使卷积前后空间尺寸不变；`BatchNorm` 加速收敛并提升稳定性。

---

## 5.2 编码器（下采样路径）

每经过一个 `MaxPool2d(2)` 空间尺寸减半，通道数加倍：

```python
self.enc1 = double_conv(3,   64)    # 256×256 → 256×256
self.enc2 = double_conv(64,  128)   # 128×128 → 128×128
self.enc3 = double_conv(128, 256)   #  64×64  →  64×64
self.enc4 = double_conv(256, 512)   #  32×32  →  32×32
self.pool = nn.MaxPool2d(2)         # 每次 pool 后尺寸减半
```

前向传播（保存中间特征图 e1–e4，后续跳跃连接使用）：

```python
e1 = self.enc1(x)              # 保存，用于 skip connection
e2 = self.enc2(self.pool(e1))  # 下采样后提特征
e3 = self.enc3(self.pool(e2))
e4 = self.enc4(self.pool(e3))
```

**感受野逐渐扩大**：越深的特征图包含越大范围的上下文信息。

---

## 5.3 瓶颈层

编解码路径之间的过渡层，通道数达到最大（1024）：

```python
self.bottleneck = double_conv(512, 1024)

# 前向传播
b = self.bottleneck(self.pool(e4))  # 16×16×1024
```

瓶颈层汇聚了全局语义信息，是模型"理解"整体场景的关键。

---

## 5.4 解码器（上采样 + 跳跃连接）

### 转置卷积上采样

```python
self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
# 将特征图空间尺寸 ×2，通道数 ÷2
```

### 跳跃连接：拼接（Concatenate）

```python
d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
#              └─上采样特征(512)─┘  └─编码器特征(512)─┘  = 1024 通道输入
```

**为什么需要跳跃连接？**

- 深层特征：语义丰富，但空间细节丢失（低分辨率）
- 浅层特征：空间细节保留，但语义信息弱（高分辨率）
- 拼接两者：同时获得**"在哪里"**（定位）和**"是什么"**（识别）

> 这是 U-Net 超越早期 FCN 的核心创新。

---

## 5.5 输出层

```python
self.out_conv = nn.Conv2d(64, 1, kernel_size=1)  # 1×1 卷积，降维到单通道

# 推理时
pred_mask = (torch.sigmoid(model(img)) > 0.5).float()
```

完整 `forward` 汇总：

```python
def forward(self, x):
    e1 = self.enc1(x)
    e2 = self.enc2(self.pool(e1))
    e3 = self.enc3(self.pool(e2))
    e4 = self.enc4(self.pool(e3))
    b  = self.bottleneck(self.pool(e4))
    d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
    d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
    d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
    return self.out_conv(d1)   # logits，未经 sigmoid
```

---

## 6. 数据预处理流水线

```python
# 图像：统一尺寸 + 归一化（ImageNet 均值/标准差）
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet 均值
                         [0.229, 0.224, 0.225]),   # ImageNet 标准差
])

# 掩码：仅调整尺寸 + 二值化（不做归一化！）
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),   # [0,255] → [0.0,1.0]
])
# __getitem__ 中：(mask_transform(mask) > 0.5).float()  →  严格 0/1
```

**关键细节**：
- 图像使用 ImageNet 预训练均值归一化，有助于迁移学习
- 掩码**不归一化**，最终阈值化为严格 0/1 二值标签
- 所有图像缩放到 256×256，确保批次张量维度一致

---

## 7. 训练策略

### 优化器：Adam

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### 学习率调度：ReduceLROnPlateau

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)
# 若验证损失连续 3 轮不下降，则 lr = lr × 0.5
```

### 早停机制（Early Stopping）

```python
best_val_loss    = float("inf")
patience_counter = 0

if val_loss < best_val_loss:
    best_val_loss    = val_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= args.patience:   # 默认 patience=10
        print("[早停] 提前终止训练。")
        break
```

> 最大训练轮数 **100**，实际轮数由早停决定（`patience=10`）。最优权重按 **Val Dice** 保存。

---

## 8. 训练输出示例

```
[设备] cuda
[数据集] 训练: 800 / 验证: 100 / 测试: 100

Epoch [01/100]  Train Loss: 0.6823  Val Loss: 0.5741  Val Dice: 0.4312  LR: 1.00e-03
  → 最优模型已保存 (Dice=0.4312)
Epoch [02/100]  Train Loss: 0.4915  Val Loss: 0.4102  Val Dice: 0.6021  LR: 1.00e-03
  → 最优模型已保存 (Dice=0.6021)
...
Epoch [52/100]  Train Loss: 0.1823  Val Loss: 0.1654  Val Dice: 0.8712  LR: 2.50e-04

[早停] 验证损失连续 10 轮未改善，提前终止训练。

[测试集评估] 加载最优模型 unet_best.pth ...

==================================================
  Test Results (n=100)
  Dice      : 0.8841
  IoU       : 0.7923
  Precision : 0.8976
  Recall    : 0.8712
  Accuracy  : 0.9654
  Loss      : 0.1159
==================================================

训练完成！共 62 轮，最优 Val Dice: 0.8841，Test Dice: 0.8841
```

**输出文件**：

| 文件 | 内容 |
|------|------|
| `outputs/unet_best.pth`   | 验证 Dice 最高时的模型权重 |
| `outputs/unet_final.pth`  | 最后一轮的模型权重 |
| `outputs/loss_curve.png`  | 训练/验证损失曲线 |
| `outputs/predictions.png` | 来自**测试集**的 5 张预测对比图 |

---

## 8.1 预测可视化

每次训练结束，从**测试集**随机抽取 5 张，对比展示：

```
┌──────────────┬──────────────┬──────────────┐
│  Input Image │ Ground Truth │  Prediction  │
├──────────────┼──────────────┼──────────────┤
│  内镜 RGB    │   白=息肉    │  模型预测    │
│     ...      │     ...      │     ...      │
└──────────────┴──────────────┴──────────────┘
```

```python
@torch.no_grad()
def visualize_predictions(model, dataset, device, save_dir, n=5):
    model.eval()
    indices = random.sample(range(len(dataset)), n)
    for row, idx in enumerate(indices):
        img, mask = dataset[idx]
        logit = model(img.unsqueeze(0).to(device))
        pred  = (torch.sigmoid(logit) > 0.5).float().squeeze().cpu().numpy()
        # 分别显示原图、真实掩码、预测掩码
```

---

## 9. 运行方式

### 快速启动

```bash
cd experiments/annotated_unet

# 默认参数：最大 100 轮，早停 patience=10，80/10/10 划分
python train_unet.py

# 自定义参数示例
python train_unet.py \
    --epochs     100 \   # 最大训练轮数
    --batch      8   \   # 批大小（显存不足可降到 4）
    --lr         1e-3\   # 初始学习率
    --patience   10  \   # 早停耐心轮数
    --test_ratio 0.1 \   # 测试集比例（对齐 DuckNet）
    --val_ratio  0.1 \   # 验证集比例
    --save   outputs     # 结果保存目录
```

### 数据准备

```bash
bash download_kvasir.sh   # 自动下载并解压 Kvasir-SEG
```

---

## 10. 总结

### U-Net 的核心设计思想

| 组件 | 作用 |
|------|------|
| **双卷积块** | 在每个尺度上充分提取局部特征 |
| **编码器 + MaxPool** | 逐步扩大感受野，捕获全局语义 |
| **瓶颈层** | 汇聚最高层抽象语义特征 |
| **转置卷积上采样** | 可学习的特征图放大 |
| **跳跃连接** | 融合多尺度特征，恢复空间细节 |

### Dice Loss 的适用场景

- **前景稀少**（息肉、肿瘤等）时，交叉熵易被背景主导
- Dice Loss 直接优化评估指标，训练目标与验证目标一致

### 思考题

1. 若将跳跃连接去掉，模型性能会如何变化？为什么？
2. 为什么编码器用 MaxPool 而不是带步长的卷积进行下采样？
3. 如何将 U-Net 扩展到多类别分割（如同时分割多种组织）？

---

## 参考资料

- **U-Net 原论文**：Ronneberger O, Fischer P, Brox T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
- **Kvasir-SEG 数据集**：Jha D, et al. *Kvasir-SEG: A Segmented Polyp Dataset*. MMM 2020.
- **Dice Loss**：Milletari F, et al. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV 2016.
- **代码实现**：本目录 `train_unet.py`

---

<!-- 封底 -->

# 动手实践

```bash
python train_unet.py --save outputs
```

> 观察损失曲线的收敛趋势，对比 `unet_best.pth` 与随机初始化的预测差异。

**祝实验顺利！**
