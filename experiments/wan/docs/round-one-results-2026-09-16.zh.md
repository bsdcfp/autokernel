# AutoKernel × Wan：B300 首轮算子实验结果

> 2026-09-17 更新：FFN 三种尺寸兼容性验证完成；QK 已定位舍入中点问题，Agent 修订通过原种子与新种子检查。修订保留 eager Norm，性能与原融合候选不同，见[跟进报告](precision-followup-2026-09-17.zh.md)。以下为首轮历史结果。
日期：2026-09-16。任务范围：六个 Wan 算子、三种序列长度、物理 GPU 3。

## 结论

- **LayerNorm + 调制、RMSNorm 有可复测收益**：中、长序列分别约 1.43–1.44×、1.25–1.32×。
- **Gated Residual 基本持平**，候选略慢。
- **RoPE 初版只在短序列明显获益**；根据性能反馈生成的第二版仍未超过中、长序列 compile 基线，且短序列退步。
- **QK Norm + RoPE 的候选在中、长序列精度失败**。FFN 修订版通过三种尺寸，但原生 compile 基线未过门限，不能给出有效加速比。
- torch.profiler 和 nsys 均获得实际 kernel 证据。完整捕获补采后，六个可通过 debug 精度检查的候选均有两种 profiler 数据；四个通过门限的 compile 基线有对应数据。

这是 **AutoKernel program.md 策略 + 固定任务生成适配器**的首轮实验，包含人工源码审核、固定评测器反馈。不是 AutoKernel 完整原生流程的无人干预成绩，也不是五框架排名或完整 Wan block 加速比。

## 1. 实验口径

- GPU：B300 SXM6 AC，物理 index 3，按 UUID 固定到逻辑 cuda:0。
- 运行时：Python 3.12.3；NVIDIA torch `2.12.0a0+0291f960b6.nv26.04.48445190`；CUDA 13.2；Triton 3.6；driver 580.159.04。
- 基线：`torch.compile(..., backend="inductor", mode="default", fullgraph=False)`，不启用 matmul TF32。
- Wan2.1 语义：hidden=1536，12 heads × 128，FFN=8960，B=1；序列长度 256 / 7800 / 32760。
- 输入：公开的合成 BF16 张量；调制和 gate 使用 FP32；RoPE 保留参考实现的 FP64 旋转语义。不是模型真实 activation。
- 精度：逐元素 `abs(error) <= 0.01 + 0.01 * abs(reference)`，另检查 shape、dtype、device、finite、输入不被修改。门限是临时 smoke 门限，未通过放宽门限换取加速比。
- 四组验证：seed 17、18、重复 17、seed 19 的第一个输入加 2。图重放另以 seed 20 替换静态输入验证。
- 计时：预热后捕获 32 次调用，计时 50 次 CUDA Graph 重放，每次除以 32，报告 p50。普通 Python 调用耗时另外保留，不混入该表；profiler 耗时不用于加速评分。
- 每个测量进程前后检查 GPU 占用。完成的本轮运行未检测到其他 GPU 3 计算进程；这不是连续监控或平台独占预约。

## 2. 配对性能数据

单位 μs；单元格为 **compile / candidate**。Norm 两项和 RoPE 第二版采用第二轮配对复测，其余采用首轮。同一单元格中的两者测量配置一致。

| 算子 | S=256 | S=7800 | S=32760 |
|---|---:|---:|---:|
| LayerNorm + 调制 | 2.543 / 2.242 | 28.857 / 20.102 | 115.835 / 80.931 |
| Gated Residual | 1.664 / 1.790 | 15.151 / 15.278 | 60.374 / 60.922 |
| RMSNorm | 1.813 / 1.645 | 10.840 / 8.195 | 51.507 / 41.274 |
| RoPE 初版 | 9.160 / 4.732 | 96.092 / 97.165 | 374.531 / 402.498 |
| RoPE 性能反馈修订版 | 9.359 / 8.306 | 96.128 / 97.531 | 374.540 / 387.866 |
| QK Norm + RoPE | 精度失败 / 10.464 | 精度失败 / 精度失败 | 精度失败 / 精度失败 |
| Linear + GELU 修订版 | 精度失败 / 10.190 | 精度失败 / 207.698 | 精度失败 / 855.654 |

LayerNorm + 调制的首轮中/长候选为 20.065 / 80.910 μs；RMSNorm 为 8.262 / 41.350 μs，和复测一致。这里的两轮进程测量不是统计置信区间，不能把同一进程的 50 次重放当成 50 次独立 Agent trial。

### 精度失败如何处理

Debug 的 QK compile 有 24 个超门限元素，最严重误差为门限的约 2.164 倍（0.458124 vs 0.490375）。FFN compile 有 4 个超门限元素，最严重约 1.026 倍（2.015625 vs 2.046875）。两者三种尺寸的 compile 均失败。

QK 候选 debug 通过，但在 medium、long 失败。这说明小尺寸通过不能代替尺度覆盖。后续排查已定位 FFN addmm 重写的舍入变化，并将 QK long seed 18 的 compile 超差从 5,951 降至 1；残余问题尚未解决，见[精度排查报告](precision-investigation-2026-09-16.zh.md)。失败项不计时、不作为加速比分母。

## 3. torch.profiler 与 nsys 的证据

Debug 每份 trace 标记 5 次被测调用。表中为每次调用的 kernel 数；失败基线记为不可用。

| 算子 | compile | candidate |
|---|---:|---:|
| LayerNorm + 调制 | 1 | 1 |
| Gated Residual | 1 | 1 |
| RMSNorm | 1 | 1 |
| RoPE | 4 | 1（两个版本均如此） |
| QK Norm + RoPE | 不可用 | 2（q、k 分别发射） |
| Linear + GELU | 不可用 | 2（原生 Linear + Triton GELU） |

补采中的两种 profiler 相互核对如下，数值为 5 次调用的 kernel 累计执行时间之和再除以 5，单位 μs；它不含 kernel 之间的 CPU 发射间隙，也不替代无 profiler 的基准计时。

| 候选 | torch.profiler | nsys |
|---|---:|---:|
| LayerNorm + 调制 | 2.534 | 2.541 |
| RoPE 第二版 | 8.461 | 8.448 |
| Linear + GELU 修订版 | 10.406 | 10.471 |

RoPE 从 4 个 kernel 降到 1 个，但大尺寸仍慢于 compile，说明减少发射次数本身不足以证明性能改善。RMSNorm 与 LayerNorm 的两方本来都已是一个 kernel，其收益不能归因于进一步减少 kernel 数。

### nsys 采集故障与修复

1. 初期报告缺失，或报告中没有 CUDA kernel 表。完整诊断显示硬件跟踪返回 `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`。
2. 使用官方支持的 `--trace=cuda-sw,nvtx,osrt` 软件跟踪，没有修改宿主机权限。
3. 两个候选的 API 范围捕获仍为空，改为完整进程捕获，再用 `cuda_gpu_kern_sum:nvtx-name` 按 `wanbench/<task>` 的 NVTX 名称筛选测量区间。
4. 成功条件为实际 CUDA kernel 实例数大于零。不能用进程返回 0、JSON 中精度 pass 或 `.nsys-rep` 文件存在来代替有效采集。

最终补采目录 `runs/final-profiles-r01/summary.json` 状态为 `complete`，torch/nsys 计数一致。累计保留 11 份有效 nsys 报告（含 RoPE 两个版本）。官方依据：[NVIDIA CUDA Trace](https://docs.nvidia.com/nsight-systems/UserGuide/#cuda-trace)。

## 4. Agent 做得怎样

### 有用的能力

- 能从固定算子语义生成可执行 Triton 候选，在 Norm 类算子上得到跨尺寸、可复测收益。
- 能利用明确反馈修正错误：T1 的 padded variance 掩码错误通过源码反例修复；FFN 的不支持 API 通过真实编译日志修复。
- 能遵守部分关键精度边界，例如 BF16 中间舍入和 FP64 RoPE；FFN 修订版三种尺寸均通过当前检查。

### 暴露的不足

- 会生成不支持的 API：初版 FFN 调用了 Triton 3.6 没有的 `tl.tanh`。
- 会犯归约掩码错误：T1 初版把 padding 对方差的贡献算进去，靠接近零均值的随机输入可能漏检。
- 不能从少数形状推断其他形状：QK 候选小尺寸通过，大尺寸失败；RoPE 小尺寸快、大尺寸慢。
- 性能反馈不保证改进：RoPE 第二版改成一个 program 处理 4 个 token，仍未超过中长序列 compile，并使 debug 性能退步。
- “生成了算子”需要说明边界：FFN 保留原生 `F.linear`，只替换 GELU，不能宣传为生成了新 GEMM。
- 本次生成阶段不自行执行测试。反馈、源码审核、GPU 调度由适配器与本次操作完成，尚不能据此评价完整 AutoKernel 的自主搜索能力。

### 生成成本

共 9 次模型调用：6 个初稿 + T1 正确性修复 + FFN API 修复 + RoPE 性能反馈修订。

- 生成 wall time 合计约 **1,042.55 秒（17.38 分钟）**。
- 输入 **214,104 tokens**；输出 **38,830 tokens**，其中 reasoning **30,416** 已包含在输出内。
- 模型配置：`gpt-5.6-sol`，`xhigh`。
- 不包含 GPU 评测、浏览器操作和环境排障时间；不据此推算账单。
- 两次新增生成日志均未出现 `command_execution` 事件，保留了生成与外部评测的区分。

## 5. 复现与产物

实验根目录：`/home/work/video_posttrain/fuping.chu/agent_projects/autokernel-wan/experiments/wan`。

| 相对路径 | 内容 |
|---|---|
| `runs/generate-*/` | 生成源码、prompt、响应、usage 与 SHA256 |
| `runs/preflight-debug-r03/` | 无其他进程干扰的首轮普通调用基线、torch traces |
| `runs/candidates-debug-r01/` | 首轮候选验证及 FFN 编译失败 |
| `runs/graph-debug-r01/` | 首轮 debug 图重放数据 |
| `runs/ffn-debug-r02/` | FFN 修订版 debug 数据 |
| `runs/graph-scale-r01/` | 六项 medium / long 数据 |
| `runs/graph-verify-r02/` | Norm 复测与 RoPE 第二版数据 |
| `runs/nsys-sw-debug-r01/` | 首次软件跟踪与空报告失败记录 |
| `runs/final-profiles-r01/` | 最终 T1 / RoPE 第二版 / FFN 的 torch 与 NVTX 筛选 nsys 数据 |

`configs/candidates-r02.json` 对应 FFN 修复后、RoPE 初版；`configs/candidates-r03.json` 对应 RoPE 第二版试验，不表示其所有尺寸都应替换原生实现。

目标环境最终 CPU 回归：**23 项通过，1.806 秒**。结束检查：GPU 3 利用率 0%，显存 27 MiB。代码改动均先推送 GitHub，再远端拉取。

**备份边界**：GitHub 保存评测器、配置和这份终端观察报告；生成源码、原始 JSON 与 trace 仍在远端，尚未回传备份。本报告表格为真实输出的四舍五入转录，不是原始结果文件的本地副本。

**后续正式评测仍缺**：完整 AutoKernel 原生闭环、真实 Wan block activation/dtype 契约、独立隔离 evaluator、冻结门限与多次 Agent trial、端到端 block 回归。首轮结果不能外推为五框架排名或真实模型端到端收益。
