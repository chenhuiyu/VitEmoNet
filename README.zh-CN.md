# VitEmoNet

基于 Transformer 的 SEED EEG 情绪识别实现。

## 为什么之前 acc 看起来接近随机

我仔细检查后，主要有两个高风险问题：

1. **预处理输入维度不匹配**
   - 旧代码在 `preprocess()` 里加了额外通道轴：`frames[..., tf.newaxis]`。
   - 但当前 transformer 期望输入是 `(5, 25, 62)`，不是 `(5, 25, 62, 1)`。
   - 这会破坏特征结构，训练很容易失稳。

2. **标签索引可能错位**
   - 数据脚本里标签是 `+1`（即 `{1,2,3}`）。
   - 3分类的 `SparseCategoricalCrossentropy` 需要 `{0,1,2}`。
   - 索引错位会直接影响训练与评估有效性。

## 我已修改

- 去掉 `preprocess()` 里的多余维度，保持和模型输入一致。
- 在训练阶段增加标签标准化：
  - 自动将 `{1,2,3}` 映射到 `{0,1,2}`；
  - 若范围异常直接报错，避免“悄悄训练错”。
- 修正 `class_weight` 的类别索引，使其与标准化标签一致。
- 开启随机种子，提升可复现性。
- 增加随机基线（random baseline）日志，便于 sanity check。

## 运行方式

请先准备好 `.npy` 数据文件（默认路径 `./input_data_1d/`）：
- `train_data.npy`, `train_label.npy`
- `val_data.npy`, `val_label.npy`
- `test_data.npy`, `test_label.npy`

运行：

```bash
python3 train.py
```

日志输出到 `save/<timestamp>/train.log`。

## 修改前后指标对比

当前环境里没有你的 `.npy` 数据，因此无法直接在这里跑出你真实数据的前后数值对比。

数据可用后，重点看两项：
- `random baseline accuracy`
- `final test accuracy`

健康状态应是：`final test accuracy` 能稳定高于随机猜测（3分类约 33%）。

## 后续可继续优化

- 做按被试（subject-wise）交叉验证
- 输出每个被试的 Acc/F1/Kappa
- 自动保存混淆矩阵与错误样本分析
