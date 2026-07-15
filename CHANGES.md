# 修改内容

1. `cryo_registration/fine_point_matching.py`
   - 修复 `EquivariantVectorHead` 随机初始化即接近 rank-1 的问题。
   - 保留旋转等变性。
   - 新增通道条件数/协方差正则，明确惩罚通道坍缩。
   - 将正则并入 `fine_equivariant_alignment_loss`。

2. `cryo_registration/model.py`
   - 由 `apply_to_repo.py` 自动修改。
   - 将单点等变向量 SVD 改成多对应点联合协方差 SVD。
   - 加入奇异值比例退化检查。
   - 使用组内平移中位数生成位姿假设。

3. `tests/test_equivariant_channel_diversity.py`
   - 检查随机初始化通道秩。
   - 检查旋转等变性。
   - 检查坍缩损失。
   - 检查多点联合位姿恢复。

应用命令：

```bash
python apply_to_repo.py /fangzhouc/alignmodel702/.worktrees/exclusive-region-labels
cd /fangzhouc/alignmodel702/.worktrees/exclusive-region-labels
/opt/miniconda/envs/cryo/bin/python -m pytest -q \
  tests/test_fine_point_matching.py \
  tests/test_fine_point_matching_training.py \
  tests/test_equivariant_channel_diversity.py
```

修改后必须重新开始 FinePointMatcher 训练，不要恢复旧 optimizer state。
