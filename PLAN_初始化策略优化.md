# PLAN\_初始化策略优化.md（重写版）

## Summary

结论：

1. Marigold 深度在跨视图上通常存在 scale/shift 漂移，确实会影响训练稳定性，尤其在 `stage5b` 联训时更容易放大冲突。
2. LITA-GS 没有显式做统一的跨视图 metric 对齐；它主要通过 Pearson 系列深度损失降低绝对尺度依赖，并配合其初始化与渲染链条。
3. 对当前代码库最稳妥的落地方式是：先做 scene-level 深度对齐，再做 depth back-projection 初始化；训练主损失保持现有定义不变。

## Key Changes

### A. 新增深度对齐预处理（初始化前）

新增工具：`3DRR\_codebase/tools/align\_depth\_scales.py`

输入输出：

* 输入：`scene\_root/auxiliaries/depth/<frame\_key>.png`
* 输出：`scene\_root/auxiliaries/depth\_aligned/<frame\_key>.png`
* 报告：`scene\_root/auxiliaries/depth\_aligned/depth\_align\_report.json`

固定规则：

* 仅处理 `transforms\_train.json` 中的 train 帧。
* 以 anchor 帧为基准（默认第 0 帧，`s=1,b=0`）。
* 对每个目标帧拟合仿射：`d\_aligned = s\_i \* d\_i + b\_i`。
* 约束来自相机已知位姿下的跨视角重投影重叠区域。
* 使用鲁棒拟合（Huber 风格迭代加权最小二乘）。
* 若某帧无有效重叠约束，直接报错，不静默回退。

### B. 接入 depth back-projection 初始化

改动文件：`3DRR\_codebase/core/model/simple\_3dgs.py`

新增配置：

* `INIT\_MODE: random | depth\_backproject`
* `INIT\_DEPTH\_DIR`（默认 `auxiliaries/depth\_aligned`）
* `INIT\_BACKPROJECT\_NEAR/FAR`
* `INIT\_BACKPROJECT\_SAMPLE\_STRIDE`
* `INIT\_BACKPROJECT\_MIN\_VALID\_POINTS`
* `INIT\_BACKPROJECT\_DEPTH\_EPS`
* `INIT\_BACKPROJECT\_DEPTH\_INVERT`

固定流程：

1. 读取 train 帧深度与官方位姿。
2. 按 stride 采样像素。
3. 将归一化深度映射到 `\[near, far]`。
4. 反投影到世界坐标并聚合多帧点云。
5. 采样到 `NUM\_INIT\_POINTS`，写入 `splats\["means"]`。

失败策略：

* 缺少 train 深度：报错并包含 `frame\_key`。
* 有效点不足：报错，不做随机补点。

### C. 训练接线

改动文件：`3DRR\_codebase/train.py`

* 从 train dataset 构建 `init\_context`（`scene\_root` + `frame\_key/transform\_matrix`）。
* 训练实例化改为：`Simple3DGS(cfg, data\_info, init\_context=init\_context)`。
* 不改当前主损失定义（RGB/DepthPrior/StructurePrior）。

### D. 阶段配置默认

已将 `stage4/stage5a/stage5b` 配置切到：

* `INIT\_MODE: 'depth\_backproject'`
* `INIT\_DEPTH\_DIR: 'auxiliaries/depth\_aligned'`

其他 stage 保持 random（默认行为）。

## Test Plan

### 1\) 预处理验证

命令：

```bash
python tools/align\_depth\_scales.py <scene\_root>
```

验收：

* `depth\_aligned` 覆盖所有 train 帧。
* 生成 `depth\_align\_report.json`。
* 报告包含每帧 `s\_i,b\_i`、overlap、对齐前后误差（median/p90）。
* 多数帧对齐后误差下降。

### 2\) 初始化行为验证

* `INIT\_MODE=random`：行为与旧版一致。
* `INIT\_MODE=depth\_backproject`：日志打印 `used\_frames/dropped\_frames/total\_points/sampled\_points`。
* 缺深度或有效点不足：训练启动即报错。

### 3\) Validation 对照

固定其余超参，仅比较初始化方式：

1. `stage4`: random vs depth\_backproject(aligned)
2. `stage5a`: random vs depth\_backproject(aligned)
3. `stage5b`: random vs depth\_backproject(aligned)

至少比较 `step\_15000` 与 `step\_30000` 的 `results.json/per\_view.json`。

### 4\) Development 落地

* 仅用 train 帧做对齐与初始化。
* 不依赖 test GT。
* 保持现有 step 目录结构导出。

## Assumptions

* Marigold 深度并非天然跨视图 metric 一致。
* 本轮仅做“深度对齐 + 初始化优化”，不引入 LITA-GS 之外新模块。
* 暂不联动改 loss 或 densify 超参，避免多变量耦合。

