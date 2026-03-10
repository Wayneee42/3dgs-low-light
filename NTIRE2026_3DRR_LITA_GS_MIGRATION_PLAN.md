# NTIRE 2026 3D 修复与重建挑战赛迁移方案

## 1. 文档目标

本文档用于指导将 `LITA-GS` 中对低照度场景有价值的能力，按阶段迁移到 `3DRR_codebase` 主干中。迁移的唯一主线是：

- 以 `3DRR_codebase` 作为训练、评测、提交流程的唯一主干。
- 以 `3DRR_codebase/dataset` 中的官方 Blender 数据组织方式作为唯一严格参考。
- 不反向修改官方数据去适配 `LITA-GS`，而是把 `LITA-GS` 的思想拆解后，重新适配到官方主干。

这意味着后续所有设计判断都应遵守以下原则：

1. 官方 `transforms_train.json`、`transforms_val.json`、`transforms_test.json` 是相机定义与划分真值。
2. 官方 `train/`、`val/`、`test/` 目录结构是图像组织真值。
3. 任意新增模块都不能破坏 `3DRR_codebase/train.py` 和 `3DRR_codebase/eval.py` 的现有最小可运行链路。
4. `LITA-GS` 只能作为“能力来源”，不能继续作为“工程骨架”。

---

## 2. 当前两套代码的本质差异

### 2.1 `3DRR_codebase` 的工程假设

`3DRR_codebase` 当前是一套非常干净的最小闭环：

- 数据入口固定在 [`3DRR_codebase/core/data/blender.py`](D:/github/3dgs-low-light/3DRR_codebase/core/data/blender.py)
- 训练入口固定在 [`3DRR_codebase/train.py`](D:/github/3dgs-low-light/3DRR_codebase/train.py)
- 推理入口固定在 [`3DRR_codebase/eval.py`](D:/github/3dgs-low-light/3DRR_codebase/eval.py)
- 模型主体固定在 [`3DRR_codebase/core/model/simple_3dgs.py`](D:/github/3dgs-low-light/3DRR_codebase/core/model/simple_3dgs.py)

它的优点是：

- 数据格式简单且与官方数据一致。
- 训练、验证、测试分离明确。
- checkpoint 与渲染输出形式清晰，便于后续整理挑战赛提交链路。
- 依赖较轻，便于快速迭代。

它的不足也很明确：

- 目前低照增强仅在训练中用了一个非常粗糙的 `gamma_augment`。
- 没有深度、结构先验、去噪器、曝光建模等更强能力。
- 仍是“最小 3DGS baseline”，距离竞赛级系统还有明显差距。

### 2.2 `LITA-GS` 的工程假设

`LITA-GS` 的入口和数据假设与官方主干明显不一致：

- 训练入口是 [`LITA-GS/train_underexposed.py`](D:/github/3dgs-low-light/LITA-GS/train_underexposed.py)
- 场景组织依赖 [`LITA-GS/scene/__init__.py`](D:/github/3dgs-low-light/LITA-GS/scene/__init__.py)
- 数据读取依赖 [`LITA-GS/scene/dataset_readers.py`](D:/github/3dgs-low-light/LITA-GS/scene/dataset_readers.py)
- 默认源数据是 COLMAP 目录，而不是官方 Blender JSON 目录

`LITA-GS` 默认要求的数据资产包括：

- `sparse/0/*.bin|*.txt`
- `images/`
- `images_low/`
- `depth_maps/`
- 结构先验目录，例如 `W_0.8/`
- 额外配置参数，如曝光目标、深度损失、先验损失、去噪器相关超参数

也就是说，`LITA-GS` 强在方法，但弱在与本次挑战赛官方基线的格式一致性。

### 2.3 不能直接“拿来即用”的根因

`LITA-GS` 不能直接替换为比赛主干，核心不是“代码复杂”，而是“假设不兼容”：

1. 官方数据是 Blender JSON 相机，而 `LITA-GS` 依赖 COLMAP 相机和稀疏点云。
2. 官方划分由 `transforms_*.json` 决定，而 `LITA-GS` 通过 `eval_index` 和图像顺序划分训练/测试。
3. 官方 baseline 的评测输出是清晰的 `test/*.png`，而 `LITA-GS` 的输出组织偏论文实验风格。
4. 官方 baseline 的模型和依赖更轻，`LITA-GS` 包含额外渲染分支、先验分支、去噪器和更多状态，迁移成本高。

因此，合理路线不是“把 `LITA-GS` 改成能跑官方数据”，而是“把 `LITA-GS` 的有效能力逐个移植到 `3DRR_codebase`”。

---

## 3. 迁移总原则

### 3.1 主干冻结原则

在整个迁移周期内，以下内容视为不可轻易改动的主干契约：

- 数据根目录组织方式
- `transforms_train.json` / `transforms_val.json` / `transforms_test.json`
- `train.py` 的基本训练流程
- `eval.py` 的单独渲染流程
- 输出目录基本结构

如果某个 `LITA-GS` 模块需要破坏上述任意一项，默认不迁移，除非能证明它对竞赛成绩有显著收益并且能被重新封装。

### 3.2 先对齐接口，再迁移能力

迁移顺序必须是：

1. 先对齐数据接口。
2. 再对齐相机与渲染接口。
3. 再迁移损失与训练策略。
4. 最后才迁移附加网络和先验模块。

不能一开始就把深度损失、结构先验、去噪器一起塞进主干，否则调试成本会失控，且无法判断收益来源。

### 3.3 每阶段必须可回退

每个阶段结束时都要满足两个条件：

- `3DRR_codebase` 可以独立训练与渲染。
- 新增能力可以通过配置开关关闭，退回上一个稳定版本。

这对挑战赛非常重要，因为最后冲榜时需要稳定性，不是功能堆叠。

### 3.4 先做低侵入，再做高收益高复杂度

迁移优先级建议按“收益/复杂度比”排序：

1. 低照增强策略
2. 亮度与曝光一致性建模
3. 深度先验
4. 结构先验
5. 去噪器、多分支、额外网络

这比按论文模块顺序照搬更适合竞赛工程。

---

## 4. 模块映射关系

下面这张表的目的，是把“应该迁移什么”和“不要迁移什么”讲清楚。

| 来源 | `LITA-GS` 现状 | 对应迁移目标 | 建议策略 |
| --- | --- | --- | --- |
| 数据入口 | `scene/dataset_readers.py` 依赖 COLMAP 与附加目录 | `3DRR_codebase/core/data/blender.py` | 不直接迁移，实现 Blender 主导的可选辅助模态读取 |
| 场景组织 | `scene/__init__.py` 负责相机、点云、pose 优化等 | `3DRR_codebase` 现有数据和模型边界 | 拆分思想，不整体复制 |
| 训练主循环 | `train_underexposed.py` 包含曝光、深度、先验、去噪器等混合逻辑 | `3DRR_codebase/train.py` | 只迁移有收益的损失与增强策略，主循环保持简洁 |
| 深度监督 | `pearson_depth_loss`、`local_pearson_loss` | 新增可选 `DepthPriorLoss` | 在主干中做成插件式损失 |
| 结构先验 | 依赖离线先验图与 `CIConv2d` 相关逻辑 | 新增可选 `StructurePriorLoss` | 保留监督思想，不保留目录假设 |
| 去噪器 | 与主模型和渲染分支有较强耦合 | 可选后期增强模块 | 放到最后评估是否值得迁移 |
| 数据划分 | `eval_index` 控制测试集 | 官方 `transforms_*.json` | 完全以官方划分为准，不迁移 `eval_index` 机制 |

可以概括成一句话：

- 迁移的是“能力”。
- 不迁移的是“原始工程结构”和“原始目录假设”。

---

## 5. 推荐的分阶段迁移路线

### 阶段 0：冻结官方基线，建立“可比较”的起点

#### 目标

在不引入 `LITA-GS` 任何功能之前，先把 `3DRR_codebase` 当作唯一实验平台，并建立一组可重复的 baseline 结果。

#### 输入

- 官方 Blender 数据目录
- `3DRR_codebase` 当前默认配置

#### 要做的事

1. 固定官方数据目录，不再围绕 `LITA-GS` 的 `data/` 或 COLMAP 目录做实验。
2. 对 `3DRR_codebase` 的每个官方场景建立一份配置文件，确保都能独立训练。
3. 记录每个场景的：
   - 训练命令
   - checkpoint 路径
   - test 渲染输出路径
   - 基础指标或可视化结果
4. 明确后续任何迁移都必须与这个 baseline 对比。

#### 代码落点

- [`3DRR_codebase/config/laboratory.yaml`](D:/github/3dgs-low-light/3DRR_codebase/config/laboratory.yaml)
- 其他场景配置文件

#### 产出物

- 一份统一实验表
- 一套稳定可复现的官方 baseline 输出

#### 验收标准

- 官方 Blender 数据可以在 `3DRR_codebase` 中完整跑通训练和测试渲染。
- 不依赖 `LITA-GS` 任何脚本。

#### 停止条件

- 至少一个官方场景获得稳定 baseline 输出。
- 训练命令、checkpoint 和渲染路径已经固定。

---

### 阶段 1：把官方 Blender 数据规范“写死”为唯一标准

#### 目标

把当前默认存在于 [`3DRR_codebase/core/data/blender.py`](D:/github/3dgs-low-light/3DRR_codebase/core/data/blender.py) 中的隐式假设，升级成显式的数据协议。后续所有迁移模块都必须依赖这个协议，而不是各自私自读文件。

#### 为什么这一阶段必须先做

目前 `blender.py` 已经能工作，但它更像“能跑就行”的 loader，而不是严格的数据协议层。要迁移 `LITA-GS` 能力，就必须先让所有模块知道：

- 图像从哪里来
- 相机内外参从哪里来
- train/val/test 的定义是什么
- 允许哪些可选辅助数据

#### 推荐改造方向

在 `3DRR_codebase/core/data/` 内建立统一的数据样本定义，至少统一以下字段：

- `images`
- `transforms`
- `infos.frame_name`
- `infos.split`
- 可选字段：`low_light_image`、`depth`、`prior`

关键点是：

- 这些附加字段必须是“可选”，默认不存在也能训练。
- 它们的索引必须严格绑定官方 `frame_name` 或 `file_path`，不能按文件排序猜测匹配。

#### 建议新增能力

1. 数据完整性检查脚本
   - 检查 `transforms_*.json` 与图像文件是否一一对应。
   - 检查分辨率、焦距、主点是否完整。
2. 辅助模态注册规则
   - 例如如果未来要引入深度图，就规定命名规则必须跟官方帧名一一对应。
3. 相机约定文档化
   - 明确当前 `Simple3DGS.forward()` 中使用的坐标系变换约定。

#### 代码落点

- [`3DRR_codebase/core/data/blender.py`](D:/github/3dgs-low-light/3DRR_codebase/core/data/blender.py)
- [`3DRR_codebase/core/data/__init__.py`](D:/github/3dgs-low-light/3DRR_codebase/core/data/__init__.py)
- 建议新增 `3DRR_codebase/core/data/validators.py`

#### 产出物

- 一份正式的数据协议说明
- 一套对官方帧名进行严格绑定的辅助模态规范
- 一个数据检查脚本

#### 验收标准

- 数据层能够在不影响现有 baseline 的前提下，支持可选附加字段。
- 没有任何训练逻辑直接去扫描自定义目录猜测数据关系。

#### 停止条件

- 数据样本字段已经稳定。
- 训练脚本对未来深度/先验接入不需要再改数据协议。

---

### 阶段 2：只迁移“低照建模最小闭环”，不要先迁移深度和先验

#### 目标

先把 `LITA-GS` 中最接近竞赛目标、同时工程侵入性最小的一部分迁入主干。这个阶段只关注“低照图像建模本身”，不引入深度图、结构先验图、去噪器多分支。

#### 推荐优先迁移的内容

1. 曝光目标建模
   - `LITA-GS` 中存在通过均值亮度把低照图拉到目标曝光的处理思路。
   - 可以把当前 `train.py` 中过于简化的 `gamma_augment` 升级为更合理的可配置曝光增强策略。
2. 低照一致性损失
   - 在不改变主渲染骨架的前提下，引入更合理的低照重建损失，而不是只对增强后的 RGB 做单一路径监督。
3. 训练期亮度扰动策略
   - 用于增强模型对不同低照程度的鲁棒性。

#### 推荐的最小实现路径

第一步：保留 `Simple3DGS` 不动，只改 `train.py` 中的监督逻辑。  
第二步：把当前 `gamma_augment` 替换成配置驱动的亮度变换模块。  
第三步：把 RGB loss 与低照增强 loss 拆开记录。  
第四步：只在验证确认收益后，再考虑模型侧改动。

#### 代码落点

- [`3DRR_codebase/train.py`](D:/github/3dgs-low-light/3DRR_codebase/train.py)
- 建议新增 `3DRR_codebase/core/libs/augment.py`
- 建议新增 `3DRR_codebase/core/libs/losses.py`

#### 产出物

- 一版“官方主干 + 更合理低照训练”的稳定版本
- 与原始 gamma baseline 的对照实验

#### 验收标准

- 不依赖任何 COLMAP 目录。
- 仅基于官方 Blender 数据即可训练。
- 相比原始 gamma baseline，低照场景的可视质量有明确提升。

#### 停止条件

- 已经出现稳定正收益。
- 低照训练链路不再依赖临时硬编码。

---

### 阶段 3：在 `3DRR_codebase` 中重建“可插拔先验接口”

#### 目标

为后续迁移深度先验、结构先验、额外监督信号建立统一扩展接口，但此阶段先做接口，不急着把所有先验都接上。

#### 关键思想

不要把 `LITA-GS` 的 `use_depth`、`use_prior`、`use_denoiser` 这种耦合式逻辑原样搬进来，而是要在 `3DRR_codebase` 中做成插件式能力。

推荐结构：

- `core/data/` 负责提供可选模态
- `core/model/` 仍保持主模型最小化
- `core/losses/` 或 `core/modules/` 负责新增监督项
- 配置文件负责显式打开或关闭模块

#### 建议接口形式

每个先验模块至少包含：

- 输入需求
- 前向计算
- loss 权重
- 开关项
- 日志项

例如可以定义：

- `RGBLoss`
- `ExposureConsistencyLoss`
- `DepthPriorLoss`
- `StructurePriorLoss`

#### 代码落点

- [`3DRR_codebase/train.py`](D:/github/3dgs-low-light/3DRR_codebase/train.py)
- 建议新增 `3DRR_codebase/core/losses/`
- 建议扩展配置文件的 `LOSS` 和 `PRIORS` 区块

#### 这样做的收益

- 后面引入新先验时不会反复重写训练主循环。
- 任何模块收益都可单独做 ablation。
- 最后冲榜时更容易裁剪无收益模块。

#### 验收标准

- 训练主循环不再堆满 `if use_xxx` 的分叉逻辑。
- 任意先验都能通过配置单独开关。

#### 停止条件

- 新增一个先验模块只需要改配置和模块注册，不需要重写主循环。

---

### 阶段 4：迁移深度先验，但必须遵守官方数据索引

#### 目标

将 `LITA-GS` 中深度相关能力迁移到 `3DRR_codebase`，但深度图不再作为“原始训练入口前提”，而是作为绑定到官方 Blender 帧上的附加监督。

#### 正确迁移方式

1. 保持官方 `transforms_*.json` 和 `train/val/test` 不变。
2. 为每个官方帧生成或收集对应深度图。
3. 按官方帧名建立深度图映射。
4. 在 `Blender` loader 中把深度作为可选字段读入。
5. 在损失层中增加深度一致性监督。

#### 错误方式

以下做法都不建议：

- 重新跑 COLMAP 并用 COLMAP 结果替换官方相机。
- 用 `LITA-GS` 的 `eval_index` 重新定义训练测试划分。
- 按文件顺序而不是官方帧名去匹配深度图。

#### 迁移注意点

1. 深度的作用应是“辅助几何稳定”，不是重写主数据协议。
2. 深度图质量不足时，loss 权重必须可以快速降到 0。
3. 需要先验证深度图与官方视角是否严格一一对应。

#### 代码落点

- [`3DRR_codebase/core/data/blender.py`](D:/github/3dgs-low-light/3DRR_codebase/core/data/blender.py)
- 建议新增 `3DRR_codebase/core/losses/depth_prior.py`
- 配置文件中增加 `PRIORS.DEPTH` 组

#### 产出物

- 官方帧名绑定的深度图组织规范
- 深度监督实验结果

#### 验收标准

- 关闭深度分支时，系统退化为阶段 2 的稳定版本。
- 开启深度分支后，几何一致性有改善，但不破坏基础颜色重建。

#### 停止条件

- 深度先验已经能稳定提升至少一部分场景。
- 如果没有提升，能够干净回退。

---

### 阶段 5：迁移结构先验，但只迁移“监督思路”，不要复制目录假设

#### 目标

将 `LITA-GS` 中结构先验的有效思想迁入主干，但不复用其原始目录组织和命名耦合。

#### 原则

`LITA-GS` 中结构先验依赖特定目录和特定生成流程，这套东西在比赛工程里不能直接当基础设施。更合理的做法是：

- 只保留“结构先验作为额外监督”的思路。
- 重新定义先验文件如何与官方帧名绑定。
- 把先验提取脚本视为离线预处理，而不是训练入口前提。

#### 推荐流程

1. 先为官方数据建立结构先验生成脚本。
2. 产物命名与官方帧名严格一致。
3. 训练时仅读取已生成的先验文件。
4. 缺失先验时自动跳过，不阻塞主训练。

#### 代码落点

- 建议新增 `3DRR_codebase/tools/extract_structure_prior.py`
- 建议新增 `3DRR_codebase/core/losses/structure_prior.py`
- 数据层继续只认官方帧键

#### 为什么不能直接搬目录

因为 `LITA-GS` 当前目录组织更像论文复现实验环境，不适合作为挑战赛工程主干。比赛工程必须强调：

- 数据关系清楚
- 可批量运行
- 可快速定位错误
- 结果可复现

#### 验收标准

- 结构先验的存在不会改变官方主数据集结构。
- 结构先验分支可以独立做消融实验。

#### 停止条件

- 结构先验要么产生稳定收益，要么被明确排除出主线。

---

### 阶段 6：最后再评估是否迁移去噪器或额外分支网络

#### 目标

只在前面阶段已经稳定且确有收益的情况下，再决定是否迁移 `LITA-GS` 的去噪器、多分支渲染或其他重模块。

#### 为什么必须放到最后

因为这类模块往往带来以下问题：

- 训练状态更多
- checkpoint 更复杂
- 推理路径更重
- 调参空间迅速膨胀
- 更难定位到底是哪一部分在提升或拖累结果

对于挑战赛工程来说，这些模块不是不能上，而是必须满足两个前提：

1. 前面阶段的主干已经稳定。
2. 通过消融确认它带来的收益大于复杂度成本。

#### 决策标准

只有在满足以下条件时才建议迁移：

- 没有它时结果已经稳定。
- 加上它后验证集质量持续提升，而不是偶发提升。
- 额外显存、训练时间、checkpoint 管理成本在可接受范围内。

#### 代码落点

- 若需要，再单独引入 `core/modules/denoiser.py`
- 不直接污染 `Simple3DGS` 的基本渲染职责

#### 验收标准

- `eval.py` 仍能在清晰、稳定的路径下输出最终 test 渲染。
- 新增状态不会破坏 checkpoint 的可管理性。

#### 停止条件

- 若收益不稳定，直接停止迁移，不进入最终参赛主线。

---

## 6. 建议的工程重构顺序

为了让迁移真正可落地，建议按下面顺序改代码，而不是想到什么改什么。

### 第一步：稳住数据层

优先处理：

- [`3DRR_codebase/core/data/blender.py`](D:/github/3dgs-low-light/3DRR_codebase/core/data/blender.py)
- [`3DRR_codebase/core/data/__init__.py`](D:/github/3dgs-low-light/3DRR_codebase/core/data/__init__.py)

要达到的结果：

- 官方 Blender 数据读取逻辑成为唯一入口。
- 支持可选附加模态，但不改变默认行为。

### 第二步：稳住训练主循环

优先处理：

- [`3DRR_codebase/train.py`](D:/github/3dgs-low-light/3DRR_codebase/train.py)

要达到的结果：

- 把增强、渲染、loss、日志分层。
- 后续新增监督项只改 loss 组合，不改训练骨架。

### 第三步：稳住模型边界

优先处理：

- [`3DRR_codebase/core/model/simple_3dgs.py`](D:/github/3dgs-low-light/3DRR_codebase/core/model/simple_3dgs.py)

要达到的结果：

- 主模型只负责高斯参数与渲染。
- 不要把所有低照逻辑和先验逻辑都塞进模型内部。

### 第四步：新增模块化配置

优先处理：

- [`3DRR_codebase/config/laboratory.yaml`](D:/github/3dgs-low-light/3DRR_codebase/config/laboratory.yaml)
- 以及其他场景配置文件

建议新增配置组：

- `AUGMENTATION`
- `LOSS`
- `PRIORS`
- `EXPERIMENT`

这样后面开关模块才有清晰控制面。

---

## 7. 推荐的数据规范设计

为了兼容未来的深度图、结构先验和低照辅助图，建议在官方 Blender 目录旁边增加“可选附加模态目录”，但绝不改官方基础目录。

推荐形式如下：

```text
scene_root/
├─ train/
├─ val/
├─ test/
├─ transforms_train.json
├─ transforms_val.json
├─ transforms_test.json
├─ auxiliaries/
│  ├─ depth/
│  │  ├─ train_0001.png
│  │  ├─ train_0002.png
│  │  └─ ...
│  ├─ prior/
│  │  ├─ train_0001.png
│  │  ├─ train_0002.png
│  │  └─ ...
│  └─ lowlight/
│     ├─ train_0001.png
│     ├─ train_0002.png
│     └─ ...
```

这里最重要的不是目录名，而是绑定规则：

- 必须使用官方 split + 帧名进行一一映射。
- 不能依赖排序。
- 不能依赖手工维护的图像列表。

建议统一帧键定义为：

- `train_0001`
- `val_0017`
- `test_0036`

这与当前 `Blender` loader 中 `frame_name = "_".join(frame["file_path"].split("/")[-2:])` 的思路是一致的，应该继续沿用并升级为正式规范。

### 额外建议

如果未来真的需要额外预处理数据，建议全部挂在 `auxiliaries/` 下，而不是把场景目录改造成 `LITA-GS` 那种多入口结构。这样做的好处是：

- 官方数据目录始终可直接被 baseline 使用。
- 辅助模态和主数据解耦。
- 删除某个增强模块时，不会连带破坏基础数据结构。

---

## 8. 推荐的实验策略

迁移不是代码搬运，而是实验设计。建议严格按以下 ablation 顺序推进：

1. 官方 baseline
2. baseline + 更合理的低照增强
3. baseline + 曝光一致性 loss
4. baseline + 深度先验
5. baseline + 结构先验
6. baseline + 深度先验 + 结构先验
7. 最后再看是否加去噪器或其他重模块

这样做有三个好处：

- 每一步收益来源清楚。
- 一旦某步退化，可以快速回滚。
- 最后可以自然形成比赛报告与方法说明。

不建议的做法是：

- 一次性把 `LITA-GS` 三四个模块全部接入。
- 指标上涨了也不知道为什么涨。
- 指标下降了也不知道该回退哪部分。

### 推荐的实验记录字段

建议每次实验至少记录：

- 场景名
- 配置文件名
- 是否启用曝光增强
- 是否启用深度先验
- 是否启用结构先验
- 训练总步数
- checkpoint 路径
- 渲染输出路径
- 关键指标
- 主观观察结论

最后冲榜时，这份表会比散乱日志更有价值。

---

## 9. 风险清单

### 风险 1：相机约定不一致

`3DRR_codebase` 当前是围绕 Blender JSON 的相机矩阵工作，`Simple3DGS.forward()` 里还有从 OpenGL 到 OpenCV 的变换逻辑。迁移任何外部几何模块时，都必须先检查坐标系约定是否一致，否则很容易出现：

- 渲染结果错位
- 深度监督方向错误
- 先验图与视角不对齐

### 风险 2：把离线预处理误当成训练前提

`LITA-GS` 的不少能力依赖额外预处理产物，但挑战赛工程中不能让训练主链条强依赖太多脆弱前处理。正确做法是：

- 把它们定义为可选增强项。
- 缺失时系统仍能训练。

### 风险 3：评测链路被复杂模型拖垮

如果后期引入太多额外状态，可能导致：

- checkpoint 管理困难
- 推理脚本复杂化
- 提交产物不统一

所以 `eval.py` 的主职责应始终保持清晰：给定 checkpoint，稳定输出 test 渲染结果。

### 风险 4：过早重写主模型

一旦过早把低照、先验、去噪器逻辑都塞进模型主体，后续会很难判断：

- 问题来自数据
- 问题来自 loss
- 问题来自渲染
- 还是问题来自附加网络

因此主模型边界必须尽量稳定。

---

## 10. 最终建议

如果目标是参加 2026 NTIRE CVPR 3D 修复与重建挑战赛，最合理的策略不是“把 `LITA-GS` 改到能兼容官方”，而是：

- 先把 `3DRR_codebase` 打造成唯一、稳定、可复现的比赛主干。
- 再把 `LITA-GS` 中真正有价值的能力拆成若干独立模块，按阶段逐一迁移。
- 所有迁移都必须以 `3DRR_codebase/dataset` 中的官方 Blender 数据为严格参考。

一句话概括：

> 这次迁移的核心不是“复现 LITA-GS”，而是“在官方主干上重建一个适合挑战赛的低照 3DGS 系统”。

这会比直接硬改 `LITA-GS` 慢一些，但工程上更稳，实验上更可解释，也更接近最终可提交、可冲榜的版本。

---

## 11. 下一步的实际执行建议

如果按可执行性排序，我建议你接下来直接做下面三件事：

1. 先在 `3DRR_codebase` 内补一份正式的数据协议与辅助模态设计。
2. 先把 `train.py` 中的低照增强从简单 `gamma_augment` 升级成可配置模块。
3. 先做“低照闭环”的 ablation，再决定是否引入深度和结构先验。

如果按代码改动的第一周任务来拆，可以进一步细化为：

1. 重构 `blender.py`，统一样本字段和官方帧键。
2. 补一个数据检查脚本，确保未来深度图和先验图严格对齐官方帧。
3. 把 `train.py` 的增强逻辑独立出来，做成配置可控模块。
4. 增加实验记录模板，固定对比维度。

这样推进，风险最低，也最符合比赛开发节奏。
