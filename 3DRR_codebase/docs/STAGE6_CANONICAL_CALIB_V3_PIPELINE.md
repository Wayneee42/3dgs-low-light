# Stage6 Canonical Calib V3 Pipeline（与当前代码一致版）

本文档描述的是 `stage6_canonical_calib_v3`，对应当前代码路径：
- `train.py`
- `core/losses/builder.py`
- `core/losses/modules.py`
- `core/model/view_calibration.py`
- `config/stage6_canonical_calib_v3/laboratory.yaml`

不是 v2，也不是 `stage6_shadow_ft_v2`。

## 1. 设计目标

v3 的核心是把旧链路：

`proxy_target -> recon loss -> illum head`

替换为：

`canonical 渲染 -> per-view 退化映射 T_i -> 拟合原始低照观测`

数学形式：

`I_i^obs ≈ T_i(R_i^canon)`

其中：
- `R_i^canon`：3DGS 基础渲染（`rgb_base_hwc`）
- `T_i`：每个训练视角的低维退化映射（`ViewCalibrationTable`）
- `I_i^obs`：原始观测（`reference_hwc`）

## 2. v3 默认开关（Laboratory 配置）

来自 `config/stage6_canonical_calib_v3/laboratory.yaml`：

- `CANONICAL_CALIB.ENABLED: true`
- `CANONICAL_CALIB.VIEW_CALIB_MODE: degradation_only`
- `PROXY_TARGET.ENABLED: false`
- `AUGMENTATION.ENABLED: false`
- `LOSS.LAMBDA_RECONSTRUCTION: 0.0`
- `LOSS.LAMBDA_ILLUM_REG: 0.0`
- `PRIORS.DEPTH.ENABLED: false`
- `PRIORS.MULTIVIEW.ENABLED: false`
- `PRIORS.STRUCTURE.ENABLED: true`

Warmstart：
- `MODEL.WARMSTART_CHECKPOINT: outputs/stage5b_ft/Laboratory/step_5000/step_5000.pt`

## 3. 训练数据流（当前实现）

每个 step：

1. 从 train dataset 取一帧，`prepare_low_light_batch(...)` 生成：
- `reference`（原图）
- `supervision`（v3 默认与 reference 一致，因为 augmentation 关闭）
- `proxy_target`（保留在 context，但 v3 主损失不使用）

2. 模型前向：
- `render_outputs["rgb"]` 作为 canonical（`rgb_base_hwc`）
- `view_calibration.apply(rgb, view_id)` 得到 `view_calibrated_hwc`
- `context["rendered"]` 在 canonical 模式下被替换成 `view_calibrated_hwc`

3. loss 计算：
- `obs_photo`：`view_calibrated_hwc` 对齐 `reference_hwc`
- `view_prior`：约束 per-view 退化参数
- `canon_exp`：约束 canonical 亮度分布
- `teacher_color`（可选，当前 Laboratory v3 开启）
- `teacher_chroma`（当前 Laboratory v3 关闭，权重为 0）
- `structure_prior`（当前配置开启）

## 4. ViewCalibration（degradation_only）真实参数化

代码在 `core/model/view_calibration.py`。

每视角参数是 4 维：`[raw_d, raw_s, raw_u, raw_v]`，解码为：

- `d = clamp_min(softplus(raw_d) - offset, 0)`
- `s = exp(-clamp_min(softplus(raw_s) - offset, 0))`
- `u = chroma_scale * tanh(raw_u)`
- `v = chroma_scale * tanh(raw_v)`

注意：
- `offset = ln(2)`，保证 `raw=0` 时 `d=0, s=1`。
- `s` 的实现包含 `clamp_min`，所以 `s <= 1`，不会放大色度，只会衰减色度。

在 YCbCr 空间：

- `y_obs = sigmoid(logit(y_canon) - d)`
- `cb_obs = s * cb + u`
- `cr_obs = s * cr + v`

这保证了亮度是“只退化不增亮”的方向。

## 5. 损失项与接线（builder 实际行为）

`core/losses/builder.py` 中，`CANONICAL_CALIB.ENABLED=true` 时：

- 不构建默认 `RGBReconstructionLoss(supervision)` 主链路
- 不构建 `ReconstructionLoss`
- 不构建 `IlluminationRegularizationLoss`
- 不请求 `illum` aux head

而是构建：

1. `CanonicalObservationLoss`（名为 `obs_photo`）
- 输入：`view_calibrated_hwc`
- 目标：`reference_hwc`
- 形式：`L1 + SSIM`

2. `ViewCalibrationPriorLoss`（`degradation_only` 下）
- 约束：`(d-d0)^2 + rho*((1-s)^2 + u^2 + v^2)`

3. `CanonicalExposureAnchorLoss`（`canon_exp`）
- 作用于 `rgb_base_hwc`
- robust 统计：masked median 与 p75

4. `TeacherChromaConsistencyLoss`（`teacher_chroma`）
- 当前 Laboratory v3 为 0（不生效）

5. `TeacherColorAnchorLoss`（`teacher_color`）
- 当前 Laboratory v3 开启
- 在有效 mask 上约束 Lab：
  - `|a_s-a_t| + |b_s-b_t|`
  - 可选 `|c_s-c_t|`（由 `LAMBDA_C` 控制）

## 6. Teacher 分支真实行为

`train.py` 中 teacher 只在需要 teacher loss 时启用：

- `teacher_enabled = teacher_chroma_enabled or teacher_color_enabled`
- teacher 模型通过 `build_teacher_model(..., warmstart_checkpoint, ...)` 构建
- teacher 与 student 使用同一 warmstart checkpoint 初始化
- teacher 全部参数 `requires_grad=False`

mask 细节（`TeacherColorAnchorLoss`）：
- 亮度 mask：`MASK_L_LOW < L_t < MASK_L_HIGH`
- 色度 mask：`C_t > MASK_CHROMA_MIN`
- alpha mask：同时使用 student `alphas` 与 `teacher_alphas`，阈值 `ALPHA_MIN`

## 7. 优化器步进门控（v3 关键）

`should_step_optimizer()` 在 `degradation_only` 下：

- `view_calibration`：始终 step
- `sh0` / `shN`：仅当 `current_step > VIEW_ONLY_STEPS` 才 step
- 其他（`means/quats/scales/opacities/depth_feat/prior_feat/illum_feat`）：不 step

当前 Laboratory v3 配置是 `VIEW_ONLY_STEPS: 0`，所以：
- 从第 1 步开始，`view_calibration` 和 `sh0/shN` 都在更新
- 几何相关参数保持冻结

## 8. 验证与测试输出是不是 canonical

是。`train.py` 在 canonical 模式下调用：

- `validate(..., save_canonical=True)`
- `evaluate(..., save_canonical=True)`

`save_render_outputs(...)` 在 `save_canonical=True` 时保存 `render_outputs["rgb"]`，不是 `recon_rgb`。

## 9. 当前 v3（Laboratory）可写成的有效损失

可近似写成：

`L_total = L_obs + w_prior*L_view_prior + w_exp*L_canon_exp + L_teacher_color + w_struct*L_structure`

对应配置：
- `w_prior = 0.005`
- `w_exp = 0.05`
- `teacher_color`：`LAMBDA_AB=0.02`, `LAMBDA_C=0.01`
- `teacher_chroma = 0`
- `w_struct = 0.01`

说明：
- `structure_prior` 虽开启，但它主要影响被冻结参数时，训练驱动力会受限。

## 10. 进度条 postfix 字段（当前实现）

canonical 模式常见字段：

- `loss`：总损失
- `n_gs`：高斯数量
- `obs`：`obs_photo`
- `psnr`：`view_calibrated_hwc` vs `reference_hwc`
- `vpr`：`view_prior`
- `d`：`view_prior_d_mean`
- `cs`：`view_prior_s_mean`
- `uv`：`view_prior_uv_mean`
- `cex`：`canon_exp`
- `tchr`：`teacher_chroma`（权重大于 0 才显示）
- `tclr`：`teacher_color`（启用时显示）
- `st`：`structure_prior`

## 11. 与 v3 对齐时最容易混淆的点

1. `proxy_target` 仍会出现在 context，但不是 v3 主监督目标。
2. `rendered` 在 canonical 模式下是 `view_calibrated_hwc`，不是 `recon_hwc`。
3. test 导出图在 canonical 模式下应是 `rgb`。
4. `degradation_only` 的 `s` 是色度衰减系数（含 clamp），不是自由增益。
5. `VIEW_ONLY_STEPS=0` 代表没有独立“只训练 view_calib”前期。

---

如果后续改了 `VIEW_ONLY_STEPS`、teacher 权重或输出保存逻辑，需要同步更新本文件对应章节（第 7、8、10 节）。
