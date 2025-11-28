# 图神经网络 - 节点分类

基于图卷积网络(GCN)的节点分类入门项目，使用Cora论文引用网络数据集。

## 项目简介

- **任务**：节点分类
- **数据集**：Cora（论文引用网络）
- **模型**：图卷积网络（GCN）
- **框架**：PyTorch + PyTorch Geometric

## 特性

- 代码简洁，注释详细
- 自动数据下载和预处理
- 训练过程包含早停机制
- 完整的可视化结果（PCA、t-SNE、混淆矩阵等）
- 支持一键运行

## 项目结构

```
.
├── data/                      # 数据集存储（自动创建）
│   └── Cora/
│       ├── raw/               # 原始数据文件
│       └── processed/         # 处理后的数据
├── outputs/                   # 输出目录
│   ├── images/                # 可视化结果
│   └── models/                # 训练好的模型
├── utils/                     # 工具脚本
│   └── download_dataset.py    # 数据集下载工具
├── main.py                    # 主脚本（一键运行）
├── data_loader.py             # 数据加载和可视化
├── model.py                   # GCN模型定义
├── train.py                   # 训练脚本
├── visualize.py               # 可视化脚本
├── requirements.txt           # 依赖包列表
├── .gitignore                 # Git忽略配置
└── README.md                  # 项目文档
```

## 快速开始

### 方法1：一键运行（推荐）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行完整流程（数据下载 + 训练 + 可视化）
python main.py
```

**可选参数：**
```bash
python main.py --skip-download  # 跳过数据下载（如已下载）
python main.py --epochs 100     # 自定义训练轮数（默认200）
python main.py --skip-viz       # 跳过可视化
```

### 方法2：分步运行

```bash
# 步骤1：下载数据集
python utils/download_dataset.py

# 步骤2：探索数据
python data_loader.py

# 步骤3：训练模型
python train.py

# 步骤4：可视化结果
python visualize.py
```

## 安装说明

### 基础安装

```bash
pip install -r requirements.txt
```

### GPU支持（推荐）

CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric numpy pandas matplotlib seaborn networkx scikit-learn tqdm
```

CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric numpy pandas matplotlib seaborn networkx scikit-learn tqdm
```

## 数据集介绍

### Cora 论文引用网络

- **节点数量**：2,708篇科学论文
- **边数量**：10,556条引用关系
- **节点特征**：1,433维词袋向量
- **类别数量**：7个研究主题
- **任务**：根据论文内容和引用关系预测论文的研究主题

### 数据集划分

- 训练集：140个节点（每类20个）
- 验证集：500个节点
- 测试集：1,000个节点

## 模型架构

```
输入(1433维) → GCN层(64维) → ReLU → Dropout → GCN层(7维) → Softmax
```

### GCN核心思想

每个节点通过聚合邻居节点的特征来更新自己的表示。

数学表达式：
```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

其中：
- H^(l)：第l层的节点特征
- A：邻接矩阵（图结构）
- W^(l)：可学习的权重矩阵
- σ：激活函数（如ReLU）

## 训练过程

### 训练流程

1. 在所有节点上进行前向传播
2. 仅在训练节点上计算损失
3. 反向传播并更新参数
4. 在验证集和测试集上评估
5. 基于验证集准确率的早停机制

### 超参数

- 学习率：0.01
- 权重衰减：5e-4
- Dropout：0.5
- 隐藏层维度：64
- 训练轮数：200（含早停）
- 早停耐心值：20轮

## 预期结果

| 指标 | 训练集 | 验证集 | 测试集 |
|------|--------|--------|--------|
| 准确率 | ~95% | ~80% | ~81% |

## 可视化结果

程序会自动生成以下可视化图表：

1. **图结构** - 论文引用网络的可视化
2. **标签分布** - 论文类别的分布情况
3. **训练历史** - 损失和准确率曲线
4. **节点嵌入(PCA)** - 学习到的节点表示的2D投影
5. **节点嵌入(t-SNE)** - t-SNE降维可视化
6. **预测结果** - 真实标签与预测标签的对比
7. **混淆矩阵** - 各类别的分类性能

所有结果保存在 `outputs/images/` 目录。

## 代码说明

### data_loader.py

- 使用PyTorch Geometric加载Cora数据集
- 提供数据集统计信息
- 可视化图结构和标签分布

### model.py

- 实现2层GCN模型
- 可选的3层Deep GCN变体
- 详细的代码注释

### train.py

- 包含早停机制的训练循环
- Adam优化器和权重衰减
- 自动GPU检测
- 训练历史记录和可视化

### visualize.py

- PCA和t-SNE降维
- 预测结果可视化
- 混淆矩阵生成

## 进阶使用

### 调整超参数

编辑 `train.py` 修改：

```python
model = GCN(
    num_features=data.num_node_features,
    num_classes=num_classes,
    hidden_dim=64,    # 可尝试 32, 128, 256
    dropout=0.5       # 可尝试 0.3, 0.6, 0.7
)

trainer = Trainer(
    model=model,
    data=data,
    lr=0.01,          # 可尝试 0.001, 0.005
    weight_decay=5e-4 # 可尝试 1e-3, 1e-4
)
```

### 尝试不同模型

使用3层Deep GCN：

```python
from model import DeepGCN

model = DeepGCN(
    num_features=data.num_node_features,
    num_classes=num_classes,
    hidden_dim=64,
    dropout=0.5
)
```

## 扩展方向

### 尝试其他数据集

- Citeseer
- PubMed
- 自定义图数据集

### 实现高级模型

- GraphSAGE（基于采样的聚合）
- GAT（图注意力网络）
- 更深的网络架构

### 探索其他任务

- 图分类
- 链接预测
- 社区检测

## 常见问题

### 数据下载失败

如果自动下载失败，可手动下载：

1. 从以下地址下载8个文件：
   ```
   https://github.com/kimiyoung/planetoid/raw/master/data/
   ```

2. 将文件放入 `data/Cora/raw/` 目录：
   - ind.cora.x
   - ind.cora.tx
   - ind.cora.allx
   - ind.cora.y
   - ind.cora.ty
   - ind.cora.ally
   - ind.cora.graph
   - ind.cora.test.index

3. 运行时跳过下载：
   ```bash
   python main.py --skip-download
   ```

### GPU不可用

代码会自动检测GPU，如不可用则使用CPU（速度较慢但功能正常）。

检查GPU可用性：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 依赖要求

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NumPy
- Matplotlib
- scikit-learn
- NetworkX
- tqdm

详见 `requirements.txt`。

## 参考资料

- **论文**：[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- **PyTorch Geometric**：https://pytorch-geometric.readthedocs.io/
- **GNN教程**：https://distill.pub/2021/gnn-intro/

## 许可证

MIT License

## 贡献

欢迎提交Pull Request！

## 引用

如果在研究中使用本代码，请引用：

```bibtex
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```
