# `stage6_canonical_calib_v2` 方案

## Summary
基于 [可辨识性坍缩优化.yaml](/D:/github/3dgs-low-light/可辨识性坍缩优化.yaml) 和当前代码库，问题不在“canonical 导出没切干净”，而在 `T_i` 的参数化和 prior 设计天然允许退化解：
- 当前 [view_calibration.py](/D:/github/3dgs-low-light/3DRR_codebase/core/model/view_calibration.py) 仍是 `[log_a, b, u, v]` 的自由光度变换，既能变暗也能变亮。
- 当前 [modules.py](/D:/github/3dgs-low-light/3DRR_codebase/core/losses/modules.py) 里的 `ViewCalibrationIdentityLoss` 仍然把 `T_i` 往 identity 拉。
- 当前 [losses.py](/D:/github/3dgs-low-light/3DRR_codebase/core/libs/losses.py) 的 `exposure_control_loss` 只是全图 mean 锚，太弱。

当前“应检查的问题”里，接线层面基本已经干净：
- `CANONICAL_CALIB.ENABLED` 时，[builder.py](/D:/github/3dgs-low-light/3DRR_codebase/core/losses/builder.py) 没再构建默认 `RGBReconstructionLoss`、`ReconstructionLoss`、`IlluminationRegularizationLoss`。
- [train.py](/D:/github/3dgs-low-light/3DRR_codebase/train.py) 的 `L_obs_photo` 输入确实是 `view_calibrated_hwc`，`L_canon_exposure` 输入确实是 `rgb_base_hwc`，`canonical_target_mean` 也没有复用 augmentation 的 `target_mean`。
- [builder.py](/D:/github/3dgs-low-light/3DRR_codebase/core/losses/builder.py) 的 `required_aux_heads()` 不会在 canonical 模式下请求 `illum`。

因此下一步不该继续调现有权重，而是按 yaml 里的两个关键改法重构 `stage6_canonical_calib`。

## Key Changes
- 把 `ViewCalibrationTable` 从自由校正器改成单向退化器。
  - 将每视角参数从 4 维 `[log_a, b, u, v]` 改为 3 维 `[d, u, v]`。
  - `d = softplus(raw_d) >= 0`。
  - 亮度变换固定为 `Y_obs = sigmoid(logit(Y_canon) - d)`，只允许变暗，不允许变亮或改对比。
  - `u, v` 只保留小色偏，建议 `u = 0.02 * tanh(raw_u)`、`v = 0.02 * tanh(raw_v)`。
  - 第一版不保留 `a`，避免又引回“对比度补偿”自由度。
- 把 `ViewCalibrationIdentityLoss` 改成“低照退化初值 prior”。
  - 删掉当前 identity 形式 `(a-1)^2 + b^2 + rho(u^2+v^2)`。
  - 新增每视角退化初值 `d0_i`，由训练图原始低照观测估计：
    - 用 train split 原图亮度中位数 `m_i`
    - 用 `canonical_target_mean` 或单独的 `CANONICAL_TARGET_MEDIAN` 作为 `m_c`
    - `d0_i = clip(logit(m_c) - logit(m_i), 0, d_max)`
  - 新 prior 形式固定为 `(d - d0_i)^2 + rho(u^2 + v^2)`。
  - `d_max` 建议显式配置，默认 `4.0`。
- 强化 canonical anchor，但不再只是“全图 mean”。
  - 保留复用 [losses.py](/D:/github/3dgs-low-light/3DRR_codebase/core/libs/losses.py) 的 exposure 思路，但扩成新的 robust 版本，不沿用当前 `rendered_luma.mean()`。
  - 第一版采用有效亮度区域统计：
    - `mask = (Y > 0.05) & (Y < 0.95)`
    - 在 mask 上计算 `median` 和 `p75`
  - canonical exposure loss 固定为：
    - `|median(Y_mask) - target_median|`
    - `+ 0.5 * |p75(Y_mask) - target_p75|`
  - 新增配置项：
    - `CANONICAL_TARGET_MEDIAN`
    - `CANONICAL_TARGET_P75`
    - `EXPOSURE_MASK_LOW`
    - `EXPOSURE_MASK_HIGH`
  - 默认值建议：
    - `CANONICAL_TARGET_MEDIAN: 0.38`
    - `CANONICAL_TARGET_P75: 0.55`
    - `EXPOSURE_MASK_LOW: 0.05`
    - `EXPOSURE_MASK_HIGH: 0.95`
- 增加 stage6 的两段式训练调度，避免一开始 joint optimize。
  - Phase A，前 `500` step：
    - 冻结 geometry 和 SH/color
    - 只训练 `view_calibration`
    - 目的：先让 `T_i` 学会解释低照
  - Phase B，剩余 step：
    - geometry 继续冻结
    - 解冻 `sh0/shN`
    - `view_calibration` 学习率保持为 `sh` 的 `5-10x`
  - 不建议在 `stage6_canonical_calib` 里再训练 `means/quats/scales/opacities`。
- 调整 `train.py` 的数据与上下文。
  - 在 train dataset 初始化时预计算 `frame_key -> calib_index` 之外，再预计算 `frame_key -> d0_i`。
  - `context` 中新增：
    - `view_prior_d0`
    - `canonical_target_median`
    - `canonical_target_p75`
  - 日志项改成：
    - `obs`
    - `vpr`
    - `cex`
    - `d`
    - `uv`
  - 去掉当前只适合旧参数化的 `a_mean`、`b_mean`。
- 新增一版并行配置，不覆盖现有 `stage6_canonical_calib`。
  - 新目录建议为 `config/stage6_canonical_calib_v2/`
  - 第一版只提供 [laboratory.yaml](/D:/github/3dgs-low-light/3DRR_codebase/config/stage6_canonical_calib/laboratory.yaml) 的 v2 变体
  - 固定：
    - `AUGMENTATION.ENABLED: false`
    - `PROXY_TARGET.ENABLED: false`
    - `LOSS.LAMBDA_RECONSTRUCTION: 0.0`
    - `PRIORS.DEPTH.ENABLED: false`
    - `PRIORS.MULTIVIEW.ENABLED: false`
    - `PRIORS.STRUCTURE.ENABLED: true`
  - 推荐起始权重：
    - `LOSS.LAMBDA_EXPOSURE: 0.05`
    - `CANONICAL_CALIB.PRIOR_WEIGHT: 0.01`
    - `CANONICAL_CALIB.COLOR_PRIOR_RHO: 8.0`
    - `CANONICAL_CALIB.LR_VIEW_CALIB: 1e-2`
    - `MODEL.LR_SH0/LR_SHN` 保持现 stage6 水平
    - geometry lr 设为 `0` 或直接不进 optimizer

## Test Plan
- 静态检查：
  - canonical 模式下 `required_aux_heads()` 仍不请求 `illum`。
  - `T_i` 新参数化下不会产生增亮，验证 `Y_obs <= Y_canon` 在数值上成立。
  - `view_prior_d0` 对每个 train frame 都能生成，且在 `[0, d_max]` 内。
- 单元/小样本验证：
  - `raw_d=0, raw_u=0, raw_v=0` 时，`view_calibrated_hwc` 与 `rgb_base_hwc` 接近相同。
  - 增大 `raw_d` 时，输出单调变暗。
  - robust exposure anchor 在少量高亮像素存在时，比 mean 版本更稳定。
- 训练行为：
  - Phase A 前 `500` step 内只有 `view_calibration` 参数在更新。
  - Phase B 解冻后，`R_canon` 不再直接塌回低照，`d_mean` 明显大于 `0`。
  - `test` 输出仍保存 canonical `rgb`，不是 calibrated 图。
- 对照实验：
  - 当前 `stage6_canonical_calib` vs `stage6_canonical_calib_v2`
  - 同一 `stage5b_ft` checkpoint 起跑，同样 `5000` step
  - 重点检查：
    - canonical 输出是否不再是低照
    - 黑屏、反射地面、线缆区域是否至少不比 `shadow_blend` 更差
    - train-view 是否由 `T_i` 吸收低照差异，而不是 canonical 本体变暗

## Assumptions
- 第一版只做 `Laboratory`，不立刻推广到其他场景。
- 第一版不再尝试补 proxy、illum head、multiview 或 depth prior，避免重新引入冲突目标。
- `canonical_target_mean=0.38` 可继续作为初始化参考，但 robust anchor 以 `median/p75` 为主，不再依赖全图 mean。
- 如果 v2 仍然学成低照 canonical，下一步应考虑“少量最亮 train views 作为固定弱 reference anchor”，而不是再回头加大旧式 exposure weight。
