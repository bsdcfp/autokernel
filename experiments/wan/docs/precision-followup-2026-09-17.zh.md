# Wan 精度跟进：残余误差定位与修订验证

日期：2026-09-17。环境、冻结参考与误差门限沿用[首轮报告](round-one-results-2026-09-16.zh.md)。

## 1. 补跑结果

第 4 张 B300（物理 index 3）空闲后恢复；守卫按原 UUID 验证，以下诊断进程前后均未检测到其他 GPU 计算进程。

- FFN long：seed 17、18、19-shifted，关闭 pattern matcher 与保留 Linear 中间输出两条路径均与 eager **逐元素一致**；显式模拟 BF16 mm→FP32 bias+GELU→BF16 与原生完整 compile 逐元素一致。结合昨日 debug/medium，共三种尺寸、九组输入完成因果对照。
- QK：保留原始 eager RMSNorm，只编译 RoPE，在 debug/medium/long、seed 17/18/19-shifted 九组输入上与完整 eager **逐元素一致**。
- 未改变默认 compile 基线。上述兼容性路径改变了优化选项或图边界，属于独立对照，不能拿它们替换旧基线后沿用加速比。

原始目录：`runs/precision-r03-ffn-long`、`runs/precision-r03-qk`，两批控制器状态均完成（QK summary 为 `complete`，FFN 单 worker 返回 0，目标 JSON 存在且九条输出比对均通过）。

## 2. QK long seed 18 的具体反例

位置为 k 的 token 28564、hidden column 605，对应 pair 604–605。Norm 跨 hidden=1536 归约，之后按 12 heads×128 旋转。

| 阶段 | eager / FP32 路径 | 候选 / 高精度诊断 |
|---|---|---|
| 输入 pair | `[-1.0390625, 2.765625]` | 同左 |
| BF16 weight pair | `[2.390625, 1.8984375]` | 同左 |
| 逐行 mean(square) | `1.0776987075805664` | FP64 `1.0776986130163473` |
| 逐行 inverse RMS | `0.9632768630981445` | FP64 `0.9632768795840814` |
| normalized BF16 pair | `[-1, 2.65625]` | 原候选 `[-1, 2.671875]` |
| weighted BF16 pair | `[-2.390625, 5.03125]` | 原候选 `[-2.390625, 5.0625]` |
| RoPE output column605 | `0.13088686764240265` | preserve-casts compile `0.11814219504594803` |

FP64 normalized 第二分量为 `2.664062620099725`；两相邻 BF16 数的中点是 `2.6640625`。直接 FP64→BF16 最近偶数舍入应取 `2.671875`。以这个高精度 norm 结果、原始 BF16 weight 舍入和原始 RoPE 重建，得到同一个 `0.11814219504594803`。

这里的 FP32 mean/inverse 是逐行探针；另已检查完整张量 eager norm 的 pair，与逐行 eager norm 一致。不能把逐行探针直接称为完整张量 reduction 内部精确指令轨迹。

最终绝对差约 `0.01274467`，容许误差约 `0.01130887`，比值 `1.1269626486178816`。归约与 inverse RMS 的微小差异跨过 BF16 舍入中点，经过权重舍入、旋转时的分量相减，形成最终超差。归一化和旋转的分界对照全部 bitwise 通过，与这条数值链路一致。

**结论**：此反例体现参考数值路径兼容性。候选在高精度诊断下取到更近的 BF16 值，也不能据此改写与 eager 对齐的评测契约。原候选仍失败；FP64 替代、只关 FMA、放宽门限均不能作为本次修复结论。高精度重建是对该点的诊断，不等于证明候选整体更准确。

## 3. AutoKernel 修订

数值反馈已冻结在 [qknorm-precision-r02.txt](../feedback/qknorm-precision-r02.txt)。修订生成仍使用已有固定任务 AutoKernel 适配器、上一版候选和测量反馈；要求保留敏感阶段的实际 eager 运算路径，并为其余部分生成新 kernel。精度、源码审查与性能验证独立进行。

修订生成完成，wall time 161.058 秒；24,506 input / 6,478 output tokens，其中 reasoning 4,943 已包含在 output 内。生成模型为已有配置 `gpt-5.6-sol` / `xhigh`，未改变模型设置。

源码共 161 行，经逐段审核：保留原始 eager RMSNorm（包括 BF16 weight 乘法），另发射 q、k 两个 Triton RoPE kernel；旋转使用显式 `mul.rn.f64`、`sub.rn.f64`、`add.rn.f64`，防止隐式 FMA 合并。无 IO、环境读取、输入修改、缓存答案或针对种子的分支。实现限定本轮公开 B=1、hidden=1536、heads=12×128、整段 grid、连续张量接口，不宣称支持任意形状。

候选为 `runs/generate-t2-qknorm-rope-r02/kernel.py`，候选映射为 `configs/candidates-r04.json`；评测器会核对生成记录的源文件 SHA256。原候选和旧结果不覆盖。

### 原有种子的独立验证

修订版三种尺寸均通过：seed 17、18、重复17、seed19 第一个输入加2，以及 CUDA Graph 新输入 seed20。检查包括 shape/dtype/device/finite/输入不变性，不能把通过当前门限等同于任意输入 bitwise 一致。

计时沿用 32 次调用捕获、50 次 CUDA Graph 重放、CUDA events p50；单位 μs。本轮 eager 和修订候选为同批次配对测量。

| S | eager | 修订候选 | eager / 候选 |
|---:|---:|---:|---:|
| 256 | 65.706 | 49.555 | 1.326× |
| 7800 | 613.760 | 461.625 | 1.330× |
| 32760 | 2612.293 | 2041.213 | 1.280× |

这些是相对 **eager** 的比值，不是相对默认 `torch.compile` 的加速比；默认 compile 仍不通过精度门限。原候选只有 debug 通过，历史测得 10.464 μs；修订约慢 4.74×，但这是跨批次诊断性对比，中长序列不能与原失败候选计算有效速度比。

### 新种子验证

`runs/qk-r02-fresh/summary.json` 为 `complete`。debug/medium/long 均通过 seed31、32、重复31、seed33 第一个输入加2，另通过 CUDA Graph 换输入 seed34。该批 fixture 未用于本次生成反馈；它仍是公开合成分布测试，不是隐藏评测集或完整模型验证。

两批候选测量共 24 次普通检查（含重复）和 6 次图重放检查通过；门限未变。新种子 long 耗时 2041.251 μs，和原种子的 2041.213 μs 接近。

候选 SHA256：`f617f48696eef6c54626437832fe9433e18c204b7e5134814adbd8e3af754e75`。

### Profiler 交叉核对

`runs/qk-r02-profiles` 的 debug 采集使用 5 次调用；nsys 采用 `cuda-sw` 全进程采集，统计只筛选 `wanbench/t2-qknorm-rope` NVTX 范围。

| 工具 | 5 次调用 kernel 总数 | 每调用 kernel 数 | 每调用累计 kernel GPU 时间 |
|---|---:|---:|---:|
| torch.profiler | 90 | 18 | 56.954 μs |
| nsys | 90 | 18 | 57.063 μs |

nsys stats 返回 0，存在真实 kernel 事件；不是只根据报告文件存在判定成功。两工具的时间接近，但这是带 profiler 的累计 kernel 时间，不替代 49.555 μs 的无 profiler Graph 计时。

原融合候选为 q/k 各一个 kernel，共2个；修订保留 eager Norm，增至18个。这个测量与短序列变慢一致。中长序列还包含多轮张量读写，不能把所有耗时变化只归因于 launch 次数。

### 本轮决策

- 修订版作为**通过当前精度契约的兼容性候选**保留，未替换历史融合版本。
- 它不是相对默认 compile 的已证明加速。默认 compile 的精度失败单独保留；FFN 的兼容性选项也没有冒充默认基线。
- 后续若继续争取融合性能，需要对齐 reference 的 reduction、rsqrt、乘法与 BF16 舍入路径，再经过独立验证；本次没有实现并验证这种完全融合修复。
- AutoKernel 能根据明确数值反馈修复兼容性，但这次通过保留 eager 计算付出了18个 kernel 的代价。因此评测必须同时报告精度、原生算子保留比例、kernel 数和延迟，不能只统计“生成成功/精度通过”。

## 4. 数据边界

目标机最终25项CPU回归全部通过（1.815秒）；本轮结束时 GPU 3 利用率0%、显存0 MiB。

本轮不使用真实模型 activation。原始 JSON、output_code、Agent 事件及候选源码目前仍位于远端；报告和自写诊断代码已在 GitHub 备份，摘要不替代原始 artifact 包。
