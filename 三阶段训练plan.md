\# 3DRR Geometry-First 实验方案：重启 `stage4`，新增 `stage5b\_ft`，再接 `stage6\_shadow\_ft`



\## Summary



当前结果已经说明：



\- `stage5b -> stage6\_shadow\_ft` 这条两阶段路线有效，Development 已从 `15.17/0.55` 提升到 `16.59/0.61`

\- 但 Development 场景里，`stage5b` 本身的几何底座仍有明显错位、漂浮和脏雾

\- `BlueHawaii` 几乎没有这些问题，说明瓶颈不在 `stage6\_shadow\_ft` 本身，而在 `stage5b` 的几何质量



因此下一轮实验不再优先改新 loss 或新模块，而是把训练链路改成三段式：



\- `stage4\_tuned`: 先只用 `D\_r` 学更稳的几何

\- `stage5b\_ft`: 从 `stage4\_tuned` checkpoint 短程 fine-tune，引入 `P\_r`

\- `stage6\_shadow\_ft\_v2`: 再从 `stage5b\_ft` checkpoint 做 illumination/proxy 外观微调



本轮固定目标是验证：\*\*把 “单次训练内阶段化” 升级成 “跨 checkpoint 阶段化” 后，Development 的几何错位和漂浮是否继续下降。\*\*



\## Key Changes



\### 1. 重启并优化 `stage4`，作为统一几何底座



新增一套 `config/stage4\_tuned/\*`，不复用旧 `stage4` 目录，避免和历史结果混淆。



固定配置：



\- 初始化与 densify 直接对齐当前最稳主线：

&#x20; - `INIT\_MODE = hybrid\_anchor`

&#x20; - `INIT\_ANCHOR\_RATIO = 0.10`

&#x20; - `INIT\_ANCHOR\_MAX\_DEPTH\_GRAD = 0.08`

&#x20; - `INIT\_ANCHOR\_BRIGHTNESS\_QUANTILE = 0.7`

&#x20; - `DENSIFY\_START\_STEP = 500`

&#x20; - `DENSIFY\_STOP\_STEP = 5000`

&#x20; - `DENSIFY\_INTERVAL = 200`

&#x20; - `DENSIFY\_GRAD\_THRESH = 0.0002`

&#x20; - `OPACITY\_RESET\_INTERVAL = 3000`

\- 训练长度：

&#x20; - `TRAIN\_TOTAL\_STEP = 15000`

&#x20; - `CHECKPOINT\_STEPS = \[15000]`

\- 先验：

&#x20; - `DEPTH.ENABLED = true`

&#x20; - `DEPTH.WEIGHT = 0.02`

&#x20; - `DEPTH.START\_STEP = 5000`

&#x20; - `STRUCTURE.ENABLED = false`

\- loss 侧保持当前稳定主线：

&#x20; - 不启 `illum`

&#x20; - 不启 `recon`

&#x20; - 不改 RGB supervision



固定目的：



\- 不追求 `stage4` 单独数值最优

\- 只追求“比当前 `stage5b` 更干净的几何底座”

\- 输出只作为 `stage5b\_ft` 的 warm-start 源



\### 2. 新增 `stage5b\_ft`，从 `stage4\_tuned` 做短程结构 fine-tune



新增 `config/stage5b\_ft/\*`，完全依赖现有 `train.py` 的 `WARMSTART\_CHECKPOINT`，不新增新代码路径。



固定配置：



\- `WARMSTART\_CHECKPOINT = outputs/stage4\_tuned/{Scene}/step\_15000/step\_15000.pt`

\- `TRAIN\_TOTAL\_STEP = 5000`

\- `CHECKPOINT\_STEPS = \[2000, 3000, 5000]`

\- `VAL\_INTERVAL\_STEP = 1000`

\- 停止 densify：

&#x20; - `DENSIFY\_START\_STEP = 6000`

&#x20; - `DENSIFY\_STOP\_STEP = 6000`

\- 先验从 step 0 生效：

&#x20; - `DEPTH.ENABLED = true`

&#x20; - `DEPTH.WEIGHT = 0.02`

&#x20; - `DEPTH.START\_STEP = 0`

&#x20; - `STRUCTURE.ENABLED = true`

&#x20; - `STRUCTURE.WEIGHT = 0.01`

&#x20; - `STRUCTURE.START\_STEP = 0`

\- 学习率改成“保守几何、允许外观和属性更新”：

&#x20; - `LR\_MEANS = 8.0e-5`

&#x20; - `LR\_MEANS\_FINAL = 8.0e-7`

&#x20; - `LR\_QUATS = 5.0e-4`

&#x20; - `LR\_SCALES = 2.5e-3`

&#x20; - `LR\_OPACITIES = 2.5e-2`

&#x20; - `LR\_SH0 = 2.5e-3`

&#x20; - `LR\_SHN = 1.25e-4`

\- 不启 illumination / reconstruction



固定目的：



\- 在稳几何上补结构，而不是重新塑形几何

\- 以 `base` 清洁度优先，不以单次 `stage5b` 风格的整体亮度为主



\### 3. `stage6\_shadow\_ft` 改成只接 `stage5b\_ft`，不再直接接旧 `stage5b`



保留现有 `stage6\_shadow\_ft` 逻辑，但新一轮实验统一使用新来源：



\- `WARMSTART\_CHECKPOINT = outputs/stage5b\_ft/{Scene}/step\_{best}/step\_{best}.pt`



固定配置：



\- `TRAIN\_TOTAL\_STEP = 5000`

\- `CHECKPOINT\_STEPS = \[2000, 3000, 5000]`

\- `VAL\_INTERVAL\_STEP = 1000`

\- densify 继续关闭：

&#x20; - `DENSIFY\_START\_STEP = 6000`

&#x20; - `DENSIFY\_STOP\_STEP = 6000`

\- `RECON\_START\_STEP = 0`

\- `DEPTH.START\_STEP = 0`

\- `STRUCTURE.START\_STEP = 0`

\- proxy 和 shadow\_blend 保持当前稳定版本，不再改形式



固定目的：



\- 只验证“更干净的 `stage5b\_ft` 几何底座”是否能进一步抬升 `stage6`

\- 本轮不再继续改 illumination loss、proxy 统计或新模块



\### 4. 实验顺序固定为 “BlueHawaii 验证 + Development 难场景优先 + 全量扩展”



\#### 第一轮：先做小规模验证，不全量开跑



按这个顺序：



1\. `stage4\_tuned/BlueHawaii`

2\. `stage4\_tuned/GearWorks`

3\. `stage5b\_ft/BlueHawaii`

4\. `stage5b\_ft/GearWorks`

5\. `stage6\_shadow\_ft\_v2/BlueHawaii`

6\. `stage6\_shadow\_ft\_v2/GearWorks`



通过标准：



\- `stage4\_tuned` 的几何不能明显差于当前 `stage5b`

\- `stage5b\_ft` 的 `base` 必须比当前 `stage5b` 更少漂浮/错位

\- `stage6\_shadow\_ft\_v2` 在 BlueHawaii 上不能明显低于当前最佳 `20.97`

\- GearWorks 上不能比当前 `stage6\_shadow\_ft` 更脏



只有第一轮通过，才扩到另外 3 个 Development 场景。



\#### 第二轮：扩展到 Development 全量



按这个顺序：



1\. `stage4\_tuned/Chocolate`

2\. `stage4\_tuned/Cupcake`

3\. `stage4\_tuned/Laboratory`

4\. `stage5b\_ft/Chocolate`

5\. `stage5b\_ft/Cupcake`

6\. `stage5b\_ft/Laboratory`

7\. `stage6\_shadow\_ft\_v2/Chocolate`

8\. `stage6\_shadow\_ft\_v2/Cupcake`

9\. `stage6\_shadow\_ft\_v2/Laboratory`



\### 5. 最终提交策略固定为“按场景选 best fine-tune step”



固定不强制四个场景共享同一个 `stage6\_shadow\_ft` step。



最终从每个场景的 `2000 / 3000 / 5000` 中单独选最优：



\- `Chocolate`: 从 `stage6\_shadow\_ft\_v2` 的 `2000/3000/5000` 里选

\- `Cupcake`: 同上

\- `GearWorks`: 同上

\- `Laboratory`: 同上



BlueHawaii 只作为验证场景，不进 Development 提交。



选 step 的标准：



\- BlueHawaii：按 `PSNR/SSIM + per\_view`

\- Development：按视觉一致性优先，重点看几何伪影是否下降

\- 不允许为了更亮而选一个明显更脏、更漂浮的 step



\## Test Plan



\### 1. 几何质量验收



固定看 `base`，而不是先看 `recon`。



关键观察点：



\- `BlueHawaii`

&#x20; - 地面反射和物体边界应保持干净

\- `Chocolate`

&#x20; - 植物、花瓣、椅子边缘不能继续变成黑色毛刺

\- `Cupcake`

&#x20; - 椅子、桌边、装饰球不能新增漂浮层

\- `GearWorks`

&#x20; - 窗下、风扇、柜体和右侧暗区不能继续有大块脏雾

\- `Laboratory`

&#x20; - 线缆、支架、百叶窗不能继续糊成漂浮层



\### 2. 数值与 checkpoint 选择



固定保存并比较：



\- `stage4\_tuned`: `step\_15000`

\- `stage5b\_ft`: `step\_2000 / 3000 / 5000`

\- `stage6\_shadow\_ft\_v2`: `step\_2000 / 3000 / 5000`



BlueHawaii 必须记录：



\- `results.json`

\- `per\_view.json`

\- `val\_base\_step\*.jpg`

\- `val\_recon\_step\*.jpg`



Development 必须记录：



\- `test/\*.png`

\- `test/base/\*.png`

\- `test/recon/\*.png`



\### 3. 通过标准



本轮实验通过需满足：



\- `stage4\_tuned` 的几何底座视觉上优于旧 `stage4`，且不明显差于当前 `stage5b`

\- `stage5b\_ft` 的 `base` 比当前 `stage5b` 更少错位和漂浮

\- `stage6\_shadow\_ft\_v2` 在 BlueHawaii 上不明显低于当前最佳

\- Development 最终提交如果仍使用 `stage6\_shadow\_ft\_v2`，应优先目标：

&#x20; - `PSNR > 16.59`

&#x20; - `SSIM > 0.61`



\## Assumptions



\- 本轮不引入 denoiser/PDM、COLMAP 初始化、额外新 loss

\- 本轮不修改 `proxy\_target` 生成形式

\- 现有 `train.py` 的 `WARMSTART\_CHECKPOINT` 已可直接复用，无需新增训练入口

\- 现有 `stage6\_shadow\_ft` 的成功经验说明：跨 checkpoint 阶段化有效，因此本轮默认沿同一思路向前扩展到 `stage5b\_ft`



