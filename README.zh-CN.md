# VitEmoNet

基于 Transformer 的 SEED 脑电情绪识别项目。

## 项目简介

VitEmoNet 面向 3 分类 EEG 情绪识别任务，使用 DE 特征作为输入，重点提供：

- 清晰可复现的训练流程
- 可扩展的模型结构
- 便于实验迭代的工程化脚本

## 项目特点

- Transformer 主干用于时序-通道特征建模
- 支持 `input_data_1d` / `input_data` 两种预处理数据目录
- 支持类别不平衡的加权训练
- 提供训练日志、评估与推理入口

## 目录结构

```text
VitEmoNet/
├── train.py              # 训练入口
├── inference.py          # 推理脚本
├── transformer.py        # Transformer 模型定义
├── vivit.py              # 可选的视频式 transformer 模块
├── callbacks.py          # 回调与 checkpoint
├── metrics.py            # 混淆矩阵与指标工具
├── data.py               # 数据处理辅助脚本
├── data_1d.py            # 1D 特征生成脚本
├── data_2d.py            # 2D 特征生成脚本
└── logger.py             # 日志工具
```

## 数据集

本项目适配 **SEED** 脑电情绪数据（3 分类）。

默认期望的预处理文件：

- `input_data_1d/train_data.npy`
- `input_data_1d/train_label.npy`
- `input_data_1d/val_data.npy`
- `input_data_1d/val_label.npy`
- `input_data_1d/test_data.npy`
- `input_data_1d/test_label.npy`

（若使用另一种特征格式，也可放在 `input_data/` 目录，文件名一致。）

## 环境依赖

建议环境：

- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn
- Matplotlib

示例安装：

```bash
pip install tensorflow numpy scikit-learn matplotlib scipy
```

## 训练

```bash
python3 train.py
```

训练输出保存于：

- `save/<timestamp>/train.log`
- `save/<timestamp>/model_summary.log`
- callbacks 生成的 checkpoint 与可视化结果

## 推理

```bash
python3 inference.py
```

请根据本地环境修改脚本中的模型路径和数据路径。

## 模型说明

`train.py` 默认模型配置：

- 输入尺寸：`(5, 25, 62)`
- 多层 Transformer Encoder
- 池化 + MLP 分类头
- 3 分类 softmax 输出

## 可调参数

可在 `train.py` 中调整：

- 学习率
- batch size
- 训练轮数
- Transformer 深度、头数、MLP 维度

## 引用

如果这个项目对你的研究或工程有帮助，欢迎引用本仓库与相关 SEED 资料。

## License

可在此补充你希望使用的开源许可证信息。
