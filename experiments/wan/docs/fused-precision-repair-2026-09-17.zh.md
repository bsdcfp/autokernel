# QK Norm + RoPE：恢复完整融合的精度修复

日期：2026-09-17。承接[兼容性修订](precision-followup-2026-09-17.zh.md)，目标是在原 eager 数值契约下恢复融合，消除18个 kernel 的回退代价。冻结参考和 `.01 + .01*abs(reference)` 门限不变。

## 1. 从“更精确”转为“相同运算顺序”

上轮已观察到 FP32 归约微差跨过 BF16 中点，随后被 weight 舍入和 RoPE 分量相减放大。高精度数学结果不能代替 eager 对齐契约。

本轮直接检查目标环境安装的 `torch/include/ATen/native/cuda/Reduce.cuh`：

- 100–107 行：根据归约长度、输出行数决定 block width/height。
- 480–555 行：连续输入按4元素向量加载，每线程4个独立累加器，最后按顺序合并。
- 635–676 行：CUDA warp 归约使用递减 offset。
- 1050–1140 行：连续归约向量化条件与维度映射。

对于本轮三种连续形状（hidden1536，行数均≥256），归约沿32个线程进行。lane `l` 的第 `j` 个累加器按 step0…11 依次累加 `x[l*4+j+step*128]^2`；再按 `((a0+a1)+a2)+a3` 合并，最后做32元素树形归约。先乘FP32倒数 `1/1536`，再单独加 epsilon，禁止跨这些边界的 FMA 合并。

这和“把1536个值补零到2048后直接 tl.sum”数学等价，但FP32舍入顺序不同。源码逻辑适用于当前固定shape、dtype和安装版本，不能直接外推到任意PyTorch版本或输入布局。

## 2. 独立归约探针

[source-derived reduction_probe.py](../reduction_probe.py) 是本次排查编写的诊断代码，不计为 Agent 生成的优化候选，也不用于性能评分。

三种尺寸、seed17/18/19-shifted/31/32/33-shifted、q/k两侧，共36个 side/case/seed 组合、489,792行：

| 比较量 | 与完整 eager 相比的不同元素数 |
|---|---:|
| mean(square) FP32 | 0 |
| tl.rsqrt inverse RMS | 0 |
| libdevice.rsqrt inverse RMS | 0 |
| 使用 tl.rsqrt 结果的 BF16 Norm | 0 |
| 使用 libdevice.rsqrt 结果的 BF16 Norm | 0 |

“shifted”仅对第一个输入q加2，沿用此前压力用例。探针采用 `num_warps=4, enable_fp_fusion=False`，artifact目录 `runs/qk-reduction-r01`，三worker完成，未检测到其他GPU3占用。

该对照说明原生累加顺序足以让已测试数据的归约与归一化逐位一致；并不证明任意输入都一致。最终融合 kernel 仍须经过完整输出验证。

## 3. Agent 修订及独立验收

将源码发现和实测反馈冻结到 [qknorm-fused-r03.txt](../feedback/qknorm-fused-r03.txt)，用原始融合候选作为 previous source，让 AutoKernel 固定任务适配器生成完整 Norm+weight+RoPE kernel。要求输入相关算术全部在自定义 kernel 内执行，最多q/k两次算术发射。

**归因边界**：数值问题由本次人工式源码排查和自写探针定位，Agent 接收具体算法反馈后负责集成。不能将该结果宣传为 Agent 独立发现 ATen 归约顺序。

### 生成与源码审核

新增一次有界生成，已有模型配置 `gpt-5.6-sol / xhigh`；wall time86.523秒，24,698 input / 3,976 output tokens，其中 reasoning2,033已包含在output内。

候选 `runs/generate-t2-qknorm-rope-r03/kernel.py` 共177行。逐段审核确认：按探针的四累加器顺序归约；两次显式BF16舍入；FP64 RoPE；`num_warps=4, enable_fp_fusion=False`；q/k各一发射。Torch只用于分配输出和view频率元数据；没有输入相关的Torch算术、种子特判、IO、环境读取或缓存答案。映射为 `configs/candidates-r05.json`，执行前核对生成记录中的源码SHA256。

### 原有输入上的完整输出验证

三种尺寸均通过seed17、18、重复17、seed19首输入加2，以及CUDA Graph换输入seed20；包括原来失败的long seed18。保持原门限，未改reference。通过门限不等于全部输出bitwise相同，残余极小FP64旋转舍入差异单独记录。

计时使用32次调用捕获、50次CUDA Graph重放的CUDA events p50，单位μs。同一行eager与融合候选为本轮同批对照。

| S | eager | 完整融合修订r03 | eager / r03 |
|---:|---:|---:|---:|
| 256 | 65.698 | 12.760 | 5.149× |
| 7800 | 613.819 | 241.656 | 2.540× |
| 32760 | 2612.344 | 984.280 | 2.654× |

上轮兼容性修订r02分别49.555 / 461.625 / 2041.213μs，当前融合版约快3.88 / 1.91 / 2.07倍；这是同环境和协议、不同批次的诊断性对照，不是独立trial置信区间。初始融合版仅debug精度通过、10.464μs；恢复原生归约次序后debug为12.760μs，仍有算术顺序对齐的成本。

这些比值均不是相对默认compile的加速，默认compile的失败仍保留。

### 新种子验收与误差幅度

新种子53、54、重复53、55首输入加2，以及Graph换输入56，三种尺寸全部通过。它们未出现在本轮生成反馈中，但依然是公开合成分布，不是正式隐藏测试集。

两批合计24次普通检查（含重复）与6次图重放检查全部通过。对六个候选测量文件的全部普通与Graph正确性记录汇总，最大绝对误差为 `7.275957614183426e-12`，debug两批均为0。没有改宽门限；最终输出不宣称普遍bitwise相同。

新种子long的Graph p50为984.304μs，与原种子984.280μs接近。输入不变性、finite、shape/dtype/device检查均通过。

### torch.profiler 与 nsys

两种工具均测量debug的5次调用，nsys采用 `cuda-sw` 全进程采集再按任务NVTX筛选。

| 工具 | 5次调用kernel总数 | 每次调用kernel数 | 每调用累计kernel GPU时间 |
|---|---:|---:|---:|
| torch.profiler | 10 | 2 | 13.383μs |
| nsys | 10 | 2 | 13.351μs |

两者与源码的q/k各一发射一致，消除了r02的eager Norm预处理。Profiler累计kernel时间不替代12.760μs的无profiler Graph计时。nsys统计返回0，存在10个实际kernel实例。

### 本轮决策与可复现路径

保留r03作为本轮通过验证的完整融合QK候选，r01失败与r02兼容性回退历史继续保留。关键改动是复现原生归约次序并保留浮点舍入边界，而非以FP64结果替代reference、放宽容差或回退eager。

- 原种子及配对eager：`runs/qk-r03-graph`。
- 新种子：`runs/qk-r03-fresh`。
- 原始torch/nsys报告：`runs/qk-r03-profiles`；汇总 `runs/qk-r03-profile-summary.json`。
- 候选SHA256：`6f556e248eb3764011cf68365dd95f86d3cf3a600a11e2ea86b0a85bb1f021b3`。
- 启动方式沿用 `preflight.py --tasks t2-qknorm-rope --variants candidate --cases debug medium long --candidate-map configs/candidates-r05.json --reviewed-candidates --timing cuda-graph`，新种子加 `--seed 53`，每次使用新的output目录。Profiler另跑，不能混入Graph时延评分。

这解决了当前固定Wan形状与已测输入下的融合候选精度失败；任意shape/dtype、真实模型activation、完整Wan block质量与性能仍未验证。默认compile自身的FFN/QK数值变化没有通过本次候选修复而消失，仍不得作为过关基线。

## 4. 实验口径

三个最终评测控制器（原种子、新种子、profiler）状态均为 `complete`。目标机25项回归全部通过（1.814秒），结束时GPU3利用率0%。

仍使用物理GPU3固定UUID、NVIDIA torch2.12.0a0 / CUDA13.2 / Triton3.6，合成BF16输入，固定隐藏维度与三种Wan grid。默认 `torch.compile` 精度失败独立保留；不把失败路径用作有效加速比分母，不改默认选项冒充原始基线。

自己的诊断与反馈代码先推送GitHub再部署。原始候选、JSON和trace仍在远端，报告是核对后的摘要，不能替代原始artifact备份。
