# Issue 1: 训练日志相关问题

**状态**: ✅ 已修复 (2026-01-13)

## 问题列表

### 1. lr_scheduler.step() 顺序问题
- **问题描述**: scheduler.step() 应该在 optimizer.step() 之后调用
- **修复状态**: ✅ 代码中顺序正确（src/train.py:136-137）
- **说明**: 检查后发现代码已经正确，无需修改

### 2. 最佳模型保存逻辑混乱
- **问题描述**:
  - Log里同时出现了负数和正数的val_loss
  - 连续打印了两遍保存信息
- **修复内容**:
  - ✅ 修改 `save_best_model` 函数，添加 `metric_higher_is_better` 参数 (src/utils/checkpoints.py:110-146)
  - ✅ 移除函数内部的重复打印
  - ✅ 修改调用代码，传递正数best_metric并设置 `metric_higher_is_better=False` (src/train.py:363-376)
  - ✅ 现在只有一次打印，显示正确的正数val_loss

### 3. 拼写错误
- **问题描述**: "tarting training..." (少了个S)
- **修复状态**: ✅ 代码中已经是正确的 "Starting training..." (src/train.py:328)
- **说明**: 检查后发现代码已经正确，无需修改

### 4. Epoch计数不统一
- **问题描述**: 标题显示Epoch 1/100，但进度条显示Epoch 0
- **修复内容**:
  - ✅ 修改进度条显示，统一从1开始计数 (src/train.py:73)
  - ✅ 现在标题和进度条都显示从1开始的epoch编号

### 5. Loss计算方式
- **问题描述**: loss计算应该按照epoch而不是step
- **修复状态**: ✅ 代码已经按照epoch计算（优化了注释说明）
- **修复内容**:
  - ✅ 优化了日志记录的注释，明确说明loss是按epoch统计的 (src/train.py:149-159)
  - ✅ 训练loss是当前epoch的累计平均loss
  - ✅ 验证loss在每个epoch后进行评估和保存

## 相关文件修改

1. `src/utils/checkpoints.py` - save_best_model函数
2. `src/train.py` - 最佳模型保存逻辑、进度条显示、日志注释
3. `plan.md` - 更新文件树

## 验证

所有问题已经修复：
- ✅ scheduler.step() 顺序正确
- ✅ 最佳模型保存逻辑统一，无重复打印
- ✅ 拼写正确
- ✅ Epoch计数统一从1开始
- ✅ Loss按epoch计算和记录
- ✅ plan.md文件树已更新