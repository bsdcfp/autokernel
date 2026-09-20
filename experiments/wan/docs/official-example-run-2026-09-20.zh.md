# AutoKernel 官方示例复现 · r01

2026-09-20。状态：**第一阶段已完成，官方示例验收未通过；尚未启动 Agent 优化。**

## 这次实际跑出了什么

在第 4 张 B300 上，原样运行四个官方命令。**54 个上游文件运行前后的 SHA256 全部一致，未修改任何 AutoKernel 代码。**

| 步骤 | 实测结果 | 对实验的影响 |
|---|---|---|
| `prepare.py` | 环境探测、tiny FP16 starter 检查通过 | 只能证明基础路径可用，不能代替完整精度检查 |
| `profile.py` | 完成紧凑 LlamaModel 分析，实际参数量 124.7M | 这是官方小模型示例，不是 7B 模型 |
| `extract.py --top 5` | 生成 5 个候选，但全部无法解析真实形状，改用默认尺寸 | 尚不能确认提取任务对应模型里的真实工作负载 |
| `bench.py` | 完整精度 **FAIL**：10 个 FP32 尺寸全部失败，边界尺寸 1537 也失败；退出码却为 **0** | 不能只看命令成功退出就判定验收通过 |

**性能数字只留作失败现场。** 原始 matmul starter 在 FP16、`M=N=K=2048` 时为 **40.69 μs**，同次原生 PyTorch 对照为 **18.41 μs**，原生报告加速比 **0.452×**。它更慢，且完整精度未通过；这也不是我们最终要求的 `torch.compile` 基线比较。

**本次不能判断离硬件极限还有多远。** 工具把 B300 的 FP16 峰值估为 153.98 TFLOPS、带宽估为 500 GB/s，因此报出计算峰值利用率 **274.2%**、带宽利用率 **123.7%**。这些比例不能作为有效硬件上限证据。本轮没有 NCU 实测，也没有新的算子或端到端加速结论。

按启动前冻结的顺序，第一阶段未通过，第二阶段不启动。**结论是原生示例在当前环境下未通过验收；还不能据此判定 Agent 的自主诊断与优化能力。** GPU 已恢复空闲，实验锁已释放。

## 结论边界

沿用的是现有 NVIDIA 环境：Python 3.12、PyTorch `2.12.0a0+0291f960b6.nv26.04.48445190`、Triton 3.6.0、NumPy 2.1.0。仓库 `.python-version` 指定 3.10，且 NumPy 要求至少 2.2.0。因此这是**现有 B300 环境的兼容性实测**，不是严格依赖复现；精度失败的根因尚未定位，不能直接归咎于框架或某项环境差异。

另有两点原始警告保留：模型转 dtype 时出现“复数转实数丢弃虚部”；profile 报出的 25.212 ms 是其 10 次采样的原生汇总值，不能直接当作单次模型端到端延迟。

## 启动前冻结的配置

- 目的：原样验证官方工具链与 Agent 工作流；不修复 review 发现的问题。
- 上游：`78435821cc3d5756ba6ee1785c397f6d8fa8c90d`，从 GitHub 已备份提交导出干净目录，运行前后核对源码哈希。
- 示例：`models/llama_7b.py` 的紧凑 `LlamaModel`，输入 `[1,512]`，float16；不是 7B 类。
- GPU：物理第 4 张卡，UUID `GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8`；启动前核对无计算进程。
- 环境：沿用已安装的 wan-autokernel 环境，`uv run` 禁止自动同步依赖。保留原 NVIDIA PyTorch 构建，不替换依赖，不更改 Python 导入规则。
- 第一阶段：原样执行 README 的 `prepare.py`、`profile.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512 --dtype float16`、`extract.py --top 5`、`bench.py`，每步最多 420 秒；记录日志和退出码。缺少前置产物则停止依赖步骤。
- 第二阶段：第一阶段验证通过后，才进入 `program.md` 原生自主循环；启动前另记预算与宿主配置。协调方不提供 review 诊断或修复提示，不修改 AutoKernel 源码。原生优化产生的候选与框架实现分别记录。
- 第三阶段：原生回接结果与外部独立验收分开；检查真实替换，不以原生成功标签直接判定能力。
- 失败处理：保留首个失败，不改脚本、门限、模型或工作流使其通过。若原生示例被阻塞，报告具体失败阶段，不冒充 Agent 已完成优化。

远端原始记录目录：`/home/work/video_posttrain/fuping.chu/agent_projects/autonomous_trials/autokernel-official-llama-r01/`，保存四步 `.log` / `.exit`、`environment.json`、`source-before.sha256`、`source-check.log`，以及 `source/workspace/` 的 profile 和提取计划。四步退出码均为 0，验收按日志内容判定。本地为核对后的结果摘要，完整原始文件仍在远端，尚未完成原始日志的异地备份。

相关记录：[代码审查](autokernel-code-review-2026-09-20.zh.md) · [评测协议](autonomous-evaluation-protocol.zh.md)。
