# 3DRR Depth Back-Projection 初始化落地方案（stage4/5a/5b）

## Summary

将当前随机 `10w` 高斯初始化替换为“基于 train 深度图 + 官方位姿”的几何初始化，用于 `stage4/5a/5b`，并保持其他阶段基线不变。  
目标是降低多先验联训的早期几何漂移，提升稳定性与可复现性（尤其是 `stage5b`）。

已锁定策略：

- 启用范围：仅 `stage4/5a/5b`
- 缺失深度：任何 train 帧缺失深度即报错终止

## Implementation Changes

### 1. 初始化入口与配置接口

在模型初始化路径新增“可选 depth back-projection 初始化”，默认行为不变（随机初始化），仅在配置开启时走深度初始化。

新增 `MODEL` 配置字段（向后兼容）：

- `INIT_MODE`: `random`（默认）| `depth_backproject`
- `INIT_FROM_SPLIT`: 固定 `train`
- `INIT_BACKPROJECT_SAMPLE_STRIDE`: 默认 `4`
- `INIT_BACKPROJECT_MAX_POINTS`: 默认等于 `NUM_INIT_POINTS`
- `INIT_BACKPROJECT_DEPTH_EPS`: 默认 `1e-3`
- `INIT_BACKPROJECT_NEAR`: 默认 `0.2`
- `INIT_BACKPROJECT_FAR`: 默认 `6.0`
- `INIT_BACKPROJECT_DEPTH_INVERT`: 默认 `false`

实现约束：

- `stage4/5a/5b` 模板配置改为 `INIT_MODE: depth_backproject`
- 其他阶段保持 `INIT_MODE: random`

### 2. 几何构建算法（基于 3DGS/Blender 约定）

使用 `transforms_train.json` 的官方 `c2w`（OpenGL 约定）与内参，逐像素反投影生成世界坐标点，再采样到 `NUM_INIT_POINTS`：

1. 读取 train 图像对应深度图（`auxiliaries/depth/<frame_key>.png`，0-1）。
2. 对深度图做有效掩码：`depth > INIT_BACKPROJECT_DEPTH_EPS`。
3. 按 `INIT_BACKPROJECT_SAMPLE_STRIDE` 下采样像素网格。
4. 深度映射到初始化尺度：
   - `z = near + depth * (far - near)`（若 `INIT_BACKPROJECT_DEPTH_INVERT=true`，改为 `1-depth`）
5. OpenGL 相机坐标：
   - `x = (u-cx)/fx * z`
   - `y = -(v-cy)/fy * z`
   - `z_cam = -z`
6. 用 `c2w` 变换到世界坐标，累计所有 train 帧点。
7. 若点数不足 `NUM_INIT_POINTS`：报错（不回退随机补点）；若过多：随机无放回采样。
8. 用该点集写入 `splats["means"]`，其余参数仍沿用现有默认初始化（`quats/scales/opacities/sh/depth_feat/prior_feat`）。

### 3. 数据与训练接线

训练构建模型时传入初始化所需训练记录（仅 train）：

- 训练入口负责把 train split 的 `transform + frame_key + depth path` 组织为初始化输入。
- 初始化过程不依赖 val/test，避免 development 集合缺 test GT 带来的耦合问题。
- 若 `INIT_MODE=depth_backproject` 且任一 train 帧缺深度图，训练在启动阶段直接报错并标出 `frame_key`。

### 4. validation / development 双数据集适配规则

- `validation`：按现有流程运行；初始化只读 train 深度，不读取 test GT。
- `development`：同样只依赖 train 深度与 train 位姿；即使 test 缺增强图/GT，不影响初始化。
- 两类数据集共享同一初始化逻辑和参数接口，不做分支实现。

## Test Plan

### 1. 功能正确性

- `INIT_MODE=random` 时行为与当前一致（均值随机初始化）。
- `INIT_MODE=depth_backproject` 时：
  - `means` 由深度反投影生成，不再是立方体随机点。
  - `means.shape == [NUM_INIT_POINTS, 3]`。
  - 初始化阶段日志打印“使用 train 深度初始化”与有效点统计。

### 2. 错误路径

- train 任意帧缺深度：启动即报错，错误信息包含 `frame_key`。
- 深度有效点不足以采样到 `NUM_INIT_POINTS`：报错终止（不 silent fallback）。

### 3. 实验对照（最小集合）

在 BlueHawaii 至少跑：

1. `stage4`：`random` vs `depth_backproject`
2. `stage5b`：`random` vs `depth_backproject`

固定其余超参数不变，仅替换初始化方式。  
验收标准：`depth_backproject` 至少提升稳定性（不同 run 波动变小）并在 `stage5b` 的中期 checkpoint 不劣于随机初始化。

## Assumptions

- 深度图来源是 Marigold 导出的 `auxiliaries/depth/*.png`，其绝对尺度不可信，仅用于几何先验形状初始化。
- 本轮不引入 COLMAP 初始化，不引入 illumination/denoiser/L_rec。
- 本轮不改变 loss 权重与 densify 超参，只改变初始化来源。
- 初始化仅影响 `means`，其余高斯参数保持当前实现，避免一次改动过多变量。
