# 3DRR x LITA-GS 先验迁移重构计划: `D_r` / `P_r` 显式辅助渲染头

## Summary

将当前“对渲染 RGB 再做先验约束”的迁移方式，重构为更接近论文与 LITA-GS 代码真实机制的“显式辅助渲染头”方案：

- `stage4`: 只启用 `D_r` 分支
- `stage5a`: 只启用 `P_r` 分支
- `stage5b`: 联合启用 `D_r + P_r`，采用分阶段启用，避免早期把几何拉偏
- 明确不迁 `illumination feature=3`、denoiser、tone mapper、progressive module；这些保持为后续阶段议题，不进入本次实现

核心思想固定为：

- `depth` 与 `structure` 都不再是“对 RGB 的后验正则”
- 它们都是每个 Gaussian 上独立的可学习标量 latent feature
- 对给定视角，通过 alpha blending 直接渲染出 `D_r`、`P_r`
- 分别与离线先验 `D`、`P` 对齐
- RGB 主干继续只负责颜色重建，不承担结构/深度先验的派生约束

## Implementation Changes

### 1. 模型与渲染接口

将当前 `Simple3DGS` 的“单 RGB / 几何深度渲染”改为“RGB 主头 + 辅助先验头”结构。

固定设计：

- 每个 Gaussian 新增两个 learnable feature：
  - `depth_feat: [N, 1]`
  - `prior_feat: [N, 1]`
- 不引入 `illumination_feat`
- RGB SH 参数、means、quats、scales、opacities 保持现有定义
- 前向接口改为返回字典，而不是位置元组，避免再把“几何深度”和“辅助深度”混成同一个 `rendered_depth`

统一前向返回结构：

```python
{
  "rgb": Tensor[H, W, 3],
  "depth_aux": Tensor[H, W, 1] | None,
  "prior_aux": Tensor[H, W, 1] | None,
  "alphas": Tensor[H, W] | Tensor[H, W, 1],
  "info": Any,
}
```

渲染实现固定为“双通道逻辑，但不与 RGB 同一语义混算”：

- RGB 保持当前 SH 渲染路径
- `depth_aux`、`prior_aux` 走单独的 auxiliary rasterization path
- auxiliary path 与 RGB path 必须共享：
  - 相同的 means / quats / scales / opacities
  - 相同的 camera / intrinsics / image size
  - 相同的 rasterization visibility 逻辑
- auxiliary path 的输出语义定义为：
  - 每像素对 Gaussian scalar feature 的 alpha blending 结果
- 实现上统一通过一个 renderer adapter 封装：
  - `render_rgb(...)`
  - `render_aux_heads(..., heads=["depth", "prior"])`
- 不允许在 loss 里通过 `CIConv(rendered_rgb)` 间接构造 `P_r`

说明：

- 本计划不要求保留当前 `render_mode="RGB+ED"` 作为训练主监督来源
- 当前 stage4 的几何 expected depth 实现视为旧方案；新方案中的 `D_r` 明确定义为 auxiliary feature head 渲染结果

### 2. 训练与损失重构

训练入口统一基于返回字典构造 context，loss 不再直接依赖“渲染 RGB 的派生先验”。

固定 loss 设计：

- `RGBReconstructionLoss`: 保持现有实现
- `DepthPriorLoss`: 监督 `depth_aux`
- `StructurePriorLoss`: 监督 `prior_aux`

`DepthPriorLoss` 固定规则：

- 监督对象：`depth_aux` 对齐离线深度先验 `depth`
- 输入都先做逐帧标准化
- loss 形式沿用当前 stage4 的稳定部分：
  - 全局 Pearson correlation loss
  - 局部 Pearson correlation loss
- 默认组合：
  - `global_weight = 1.0`
  - `local_weight = 1.0`
- 不再使用当前模型输出中的“几何 expected depth”作为训练监督主对象

`StructurePriorLoss` 固定规则：

- 监督对象：`prior_aux` 对齐离线 structure prior `structure`
- target 使用离线 CIConv 提取得到的 `P`
- `prior_aux` 与 `P` 都做逐帧 min-max normalization 到 `[0, 1]`
- loss 固定为 L1
- 不在 loss 内对 `rendered_rgb` 再做 CIConv

分阶段启用规则固定为：

- `stage4`
  - 启用 `depth_aux`
  - 禁用 `prior_aux`
  - `depth` loss 从 `START_STEP=5000` 开始
- `stage5a`
  - 启用 `prior_aux`
  - 禁用 `depth_aux`
  - `structure` loss 从 `START_STEP=5000` 开始
- `stage5b`
  - 同时启用 `depth_aux` 与 `prior_aux`
  - `depth` loss 从 `START_STEP=5000` 开始
  - `structure` loss 从 `START_STEP=10000` 开始
  - 不允许两个先验在 step 0 同时开启

默认权重固定为：

- `stage4`
  - `PRIORS.DEPTH.WEIGHT = 0.02`
- `stage5a`
  - `PRIORS.STRUCTURE.WEIGHT = 0.01`
- `stage5b`
  - `PRIORS.DEPTH.WEIGHT = 0.02`
  - `PRIORS.STRUCTURE.WEIGHT = 0.01`

日志固定输出：

- `rgb`
- `depth_prior`
- `structure_prior`
- `depth_prior_global`
- `depth_prior_local`
- `structure_prior_available`
- `total`

### 3. 数据协议与离线先验

数据协议保持官方 Blender 帧绑定，不迁 LITA-GS 的目录耦合。

固定数据约定：

- `depth` 来源：
  - `scene_root/auxiliaries/depth/<frame_key>.png`
- `structure` 来源：
  - `scene_root/auxiliaries/structure/<frame_key>.png`
- `prior/` 仅作为 legacy alias，保留读取兼容，但文档和新实验不再使用该命名

离线提取固定方式：

- `depth`
  - 继续使用当前 Marigold 提取脚本
- `structure`
  - 继续使用当前 CIConv-based extraction script
  - `INVARIANT='W'`
  - `KERNEL_SIZE=3`
  - `SCALE=0.8`

数据加载固定策略：

- `depth`：
  - 若 `PRIORS.DEPTH.ENABLED=true` 且样本缺失 depth，则训练直接报错
- `structure`：
  - 若 `PRIORS.STRUCTURE.ENABLED=true` 且样本缺失 structure，则该样本跳过 structure loss，但不阻塞训练

### 4. 配置与阶段定义

保留当前 `PRIORS` 配置组，但语义改为“辅助渲染头监督”，不再表示“对 RGB 的正则”。

固定配置接口：

```yaml
PRIORS:
  DEPTH:
    ENABLED: true|false
    WEIGHT: float
    START_STEP: int
    GLOBAL_WEIGHT: float
    LOCAL_WEIGHT: float
    BOX_SIZE: int
    SAMPLE_RATIO: float

  STRUCTURE:
    ENABLED: true|false
    WEIGHT: float
    START_STEP: int
    INVARIANT: 'W'
    KERNEL_SIZE: 3
    SCALE: 0.8
```

新增阶段配置集：

- `config/stage4/*`: `D_r` only
- `config/stage5a/*`: `P_r` only
- `config/stage5b/*`: `D_r + P_r`

阶段语义固定：

- `stage2`: 无 depth / structure 先验，作为稳定基线
- `stage4`: 仅 `D_r`
- `stage5a`: 仅 `P_r`
- `stage5b`: `D_r + P_r`
- 本次计划不再把当前“RGB 派生结构损失”的实现当作正式 stage5 方案

文档必须同步更新：

- stage4 文档改为“显式 `D_r` auxiliary head”
- stage5 文档拆成 `stage5a` 与 `stage5b`
- 明确说明当前方案与 LITA-GS 的差异：
  - 迁移 `D_r / P_r` 机制
  - 不迁 illumination feature / denoiser / tone mapper

## Test Plan

必须覆盖以下测试与验证场景。

### 1. 形状与接口测试

- model forward 在三种配置下输出正确：
  - `stage2`: `rgb` 有值，`depth_aux=None`，`prior_aux=None`
  - `stage4`: `depth_aux` 有值，`prior_aux=None`
  - `stage5a`: `depth_aux=None`，`prior_aux` 有值
  - `stage5b`: 两者都有值
- auxiliary head 输出分辨率与 RGB 一致
- 关闭所有先验时，训练仍可回退到 stage2 逻辑

### 2. 损失行为测试

- `DepthPriorLoss` 只读取 `depth_aux`，不再读取几何 expected depth
- `StructurePriorLoss` 不允许对 `rendered_rgb` 再做 CIConv
- `START_STEP` 前：
  - depth / structure loss 必须为 0
- `START_STEP` 后：
  - 对应 loss 才生效
- 缺失 structure 时：
  - `structure_prior=0`
  - 训练不中断
- 缺失 depth 且 depth enabled 时：
  - 训练报错并明确提示 frame key

### 3. BlueHawaii 验证顺序

统一按下面顺序做实验，不跳步：

1. `stage2/BlueHawaii` 重新作为新主干基线
2. `stage4/BlueHawaii` 仅 `D_r`
3. `stage5a/BlueHawaii` 仅 `P_r`
4. `stage5b/BlueHawaii` 联合 `D_r + P_r`

每个实验记录：

- `results.json`
- `per_view.json`
- `config.yaml`
- `val_step*.jpg`
- 训练日志中的各 loss 曲线

通过标准固定为：

- `stage4` 至少不应显著破坏 RGB 重建
- `stage5a` 必须优于“当前 RGB 派生结构损失方案”，否则 `P_r` 分支视为无效
- `stage5b` 若低于 `stage5a` 或明显低于 `stage2`，则判定“联合先验当前不适合作为主线”

## Assumptions

- 本轮迁移固定不实现 `illumination feature=3`
- 本轮迁移固定不实现 denoiser / tone mapper / PDM
- `depth_feat` 与 `prior_feat` 维度固定为 1，不做可配置化
- 旧 stage4/stage5 checkpoint 不要求向前兼容；新实验默认从头训练
- 现有离线先验生成方式继续沿用：
  - depth 用 Marigold
  - structure 用 CIConv(`W`, `k=3`, `scale=0.8`)
- 若 `gsplat` 不能稳定在单次调用中同时处理 RGB SH 与 auxiliary scalar head，则实现必须固定采用“RGB pass + auxiliary pass”的 renderer adapter；不得为了省一次渲染调用而把辅助先验重新塞回 RGB loss 路径
