<proposed_plan>
# 3DRR x LITA-GS 迁移优化计划：严格低照自监督版

## Summary

目标是在 **不使用真实正常光照 GT** 的前提下，继续向 LITA-GS 靠拢，并修正当前 `3DRR_codebase` 的关键偏差。

基于两个代码库当前实际：

- `3DRR_codebase` 当前最稳主线是：
  - `stage5b`
  - `D_r / P_r` 显式辅助头
  - `hybrid_anchor`
  - `densify 5000/200`
  - `low_light = 0`
  - 旧版均值 `exposure`
- LITA-GS underexposed 公开代码的实际训练主线是：
  - **只用低照图 `low_image`**
  - 从低照图生成 `target_image = low_image * (target / mean(low_image))`
  - `D_r / P_r` 也来自低照图链
  - 没有在公开 underexposed 主训练里直接用真实正常光照 GT 做主监督

因此，本轮迁移优化的核心不是继续沿当前失败的 `stage6(L_rec -> low-light reference)` 往前走，而是：

- 保留当前 `D_r / P_r` 设计
- 放弃“直接拿原始 reference 做 `L_rec`”的错误路径
- 明确引入 **低照图派生的 brighter proxy target**
- 如果继续加 illumination 分支，也只能对齐这个 proxy，而不是对齐真实正常光照图

## Key Conclusions From Both Codebases

### 1. 当前 `stage6` 失败的根因

当前 `3DRR_codebase` 的 `stage6` 里：

- `L_rec` 监督对象是 `reference`
- 但 `reference` 在当前数据协议里就是低照 train 图
- 于是 `recon_rgb` 被显式拉回低照域
- 最终输出当然会变暗，和目标“正常光照 test 风格输出”相反

所以当前 `stage6` 不是“illumination 思路错”，而是 **监督域错了**。

### 2. LITA-GS 并不是用真实正常光照 GT 做主训练监督

LITA-GS underexposed 公开代码里，训练主监督是：

- `low_image -> target_image`
- `rendered_image` 对齐 `target_image`

而不是直接对齐真实正常光照图。

这说明如果要保持“不作弊”的前提，3DRR 后续所有 illumination / reconstruction 迁移都必须遵守：

- 训练期主监督只能来自低照图本身或其派生 proxy
- 不能直接引入真实正常光照 GT 作为主监督

### 3. 当前最值得迁的不是 denoiser，而是“正确监督域下的 illumination decomposition”

但 illumination 分支只有在以下前提下才成立：

- 它的 reconstruction target 必须是低照图派生的 brighter proxy
- 不能是当前数据集里的低照 `reference`
- 也不能是假设存在的真实正常光照 GT

## Migration Direction

本轮固定采用一条更接近 LITA-GS underexposed 公开代码、也更符合当前 3DRR 数据协议的路线：

- **保留 `stage5b` 主线作为稳定基线**
- 新增一个严格低照自监督的 illumination 试验阶段，记为 `stage6_proxy`
- `stage6_proxy` 的所有监督都只能由低照图派生
- 暂缓 denoiser / PDM
- 暂缓再次尝试“真实 normal-light `L_rec`”

## Implementation Plan

### Phase 0. 固定当前稳定主线

先把当前最稳版本固定为正式 baseline：

- `stage5b`
- `hybrid_anchor`
- `densify_start = 500`
- `densify_stop = 5000`
- `densify_interval = 200`
- `low_light = 0`
- 保留旧版均值 `exposure`
- 输出仍以 `test/` 下正常光照风格结果为主

作用：

- 给后续 `stage6_proxy` 提供明确对照
- 避免 illumination 分支继续污染当前主线

### Phase 1. 数据协议重构：显式区分三类图像语义

在 `train.py` / `augment.py` / 数据上下文中统一拆出三类图像：

- `low_image`
  - 当前输入低照图
- `supervision`
  - 当前 3DRR 的增强监督图，保留现有机制以兼容 baseline
- `proxy_target`
  - 新增，严格由 `low_image` 经曝光提升得到
  - 形式仿照 LITA-GS：
    - `proxy_target = clamp(low_image * (target / mean(low_image)), 0, 1)`

固定规则：

- `proxy_target` 只能来自低照图
- 不允许从 test GT、正常光照真值或外部 paired 图像生成
- 后续 illumination reconstruction 只允许对齐 `proxy_target`

说明：

- 这一步不是引入真实 GT
- 是把“增强 supervision”与“LITA 风格 brighter proxy”在语义上分开

### Phase 2. 纠正 illumination 分支语义，形成 `stage6_proxy`

保留 `illum_feat` 和 `illum_aux`，但重构训练目标：

- `rgb_base`
  - 继续输出基础 RGB
- `illum_aux`
  - 继续输出 illumination map
- `recon_rgb`
  - 定义为最终乘法重建结果

监督规则改为：

- `L_rgb_base`
  - 监督 `rgb_base -> supervision`
- `L_rec_proxy`
  - 监督 `recon_rgb -> proxy_target`
- `L_depth`
  - 监督 `depth_aux -> depth`
- `L_prior`
  - 监督 `prior_aux -> structure`

明确禁止：

- `L_rec(recon_rgb, low_image)`
- `L_rec(recon_rgb, real normal-light GT)`

### Phase 3. Reconstruction 形式最小修正

沿用当前最小实现，不迁完整 tone mapper，但固定约束其语义为“可增亮、可压暗、默认中性”。

当前保留：

- `recon_rgb = clamp(rgb_base * illum_factor, 0, 1)`

其中：

- `illum_factor` 默认中性为 `1`
- 允许大于 `1`
- 允许小于 `1`

本轮不新增：

- tone mapper MLP
- noise head
- denoiser stages

理由：

- 先验证“监督域纠正后 illumination 是否有效”
- 不一次引入多个变量

### Phase 4. Loss 组合收敛

新的 `stage6_proxy` 总损失固定为：

- `L_total = L_rgb_base + w_rec * L_rec_proxy + w_depth * L_depth + w_prior * L_prior + w_exp * L_exp`

固定策略：

- `low_light_consistency = 0`
- 旧版均值 `exposure` 保留，但降为次级项
- 不再让 `low_light` 参与主线

建议默认值：

- `w_rec = 0.5`
- `w_depth = 0.02`
- `w_prior = 0.01`
- `w_exp = 0.02`

理由：

- 避免 reconstruction 一上来压过 RGB base
- 先让 illumination 分支作为“辅助修正”，不是主导分支

### Phase 5. 配置与阶段定义

新增阶段：

- `stage6_proxy`

语义定义：

- `stage2`: baseline
- `stage4`: `D_r`
- `stage5a`: `P_r`
- `stage5b`: `D_r + P_r`
- `stage6_proxy`: `D_r + P_r + L_r + proxy reconstruction`

`stage6_proxy` 默认：

- 延续当前最优初始化与 densify：
  - `INIT_MODE = hybrid_anchor`
  - `INIT_ANCHOR_RATIO = 0.1`
  - `INIT_ANCHOR_MAX_DEPTH_GRAD = 0.08`
  - `INIT_ANCHOR_BRIGHTNESS_QUANTILE = 0.7`
  - `densify 5000 / 200`
- `low_light = 0`
- `exposure = 0.02`
- `reconstruction = 0.5`

### Phase 6. 日志与导出修正

训练日志必须显式拆开：

- `rgb_base`
- `reconstruction`
- `depth_prior`
- `structure_prior`
- `exposure`
- `total`

PSNR 日志也必须拆成两类，避免再次混淆：

- `psnr_base`
  - `rgb_base` vs `supervision`
- `psnr_recon`
  - `recon_rgb` vs `proxy_target`

验证导出固定保存：

- `test/base/*.png`
- `test/illum/*.png`
- `test/recon/*.png`

但正式 `results.json/per_view.json` 仍然以最终输出图为主。

## Test Plan

### 1. 行为正确性

必须先验证：

- `proxy_target` 的来源仅为低照图
- `stage6_proxy` 中不存在任何对真实正常光照 GT 的训练监督
- `L_rec_proxy` 不再读取当前低照 `reference` 作为 target
- `stage5b` 行为不变，不受新阶段污染

### 2. BlueHawaii 验证顺序

固定顺序：

1. 当前最佳 `stage5b` baseline
2. `stage6_proxy` with `w_rec = 0.5`
3. `stage6_proxy` with `w_rec = 1.0`

看三件事：

- 是否仍然明显变暗
- 是否比当前 `stage6` 稳定得多
- 是否比 `stage5b` 更接近正常光照 test 风格

### 3. Development 验证规则

Development 无 GT，只看：

- 是否出现前景遮挡壳层
- 是否出现 illumination 图塌缩为常数
- `recon_rgb` 是否比 `rgb_base` 更自然，而不是更暗
- 高斯数是否保持在当前可接受范围

## What Not To Do Next

本轮明确不建议：

- 再次使用真实正常光照图做 `L_rec`
- 继续沿当前失败版 `stage6` 调权重
- 先上 denoiser / PDM
- 再次大改 exposure loss 形式
- 再次回到全量 depth initialization

## Expected Outcome

这条计划的目标不是立刻超过当前最佳 `stage5b`，而是先解决两个关键问题：

1. illumination 分支的监督域错误
2. 训练语义与“不作弊”的约束不一致

如果 `stage6_proxy` 仍然明显不如 `stage5b`，结论会很清楚：

- 在当前 3DRR 数据协议下，仅迁 illumination decomposition 仍不足以带来净收益
- 后续才值得单独评估更完整的 LITA-GS 模块，例如 denoiser/PDM 或更强的 proxy 生成策略
</proposed_plan>