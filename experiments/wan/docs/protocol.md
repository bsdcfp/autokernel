# 实施边界与来源

## 参评配置

| 框架 | 原生入口（见框架版本锁） | 接入工作 |
|---|---|---|
| auto-model-optim | program.md + run_experiment.py | 替换 VAE 目标、清理历史最佳答案和记录；独立验收不得采用原项目 FP32 eager 分母 |
| KDA-Pilot | diffusion 任务合同与 launch_kda_kernel_task.sh | 去除作者机器路径依赖，保留原生任务/审查流程 |
| KDA | README 引导的 prompts / workflow | 核对实际 Agent 执行宿主与依赖，不能声称通用 SDK 已接入 |
| autokernel | program.md + bench.py | 接入多输入 Wan 子图，保留框架内搜索；独立比较原生 compile |
| AutoMegaKernel | MCP/CLI、ScheduleConfig、kernel_knobs | 先核查 Wan 导入和任务表达；配置搜索与新 kernel 生成分开计分 |

Humanize 与领域 Skills 是组件，记录到各运行配置，不增加虚假的独立框架样本数。
原生工作流比较和模型统一后的控制变量比较单列。多角色 token 共同计入预算。

## T1 精度语义

参考 Wan2.1 revision `9737cba9c1c3c4d04b33fcad41c111989865d315`：

- [WanLayerNorm 与 block](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L91)。
- normalization 在 FP32 上执行，结果先转回 x.dtype，再转 FP32 做 modulation。
- scale、shift 是已经与 block modulation 参数相加并 chunk 后的 e[1]、e[0]。
- 因此 T1 不包含 timestep embedding/modulation 参数相加；不能把这部分开销从完整 block 隐藏。
- 首版同时支持 BF16、FP32 的 x，真实调用各阶段 dtype 需用 activation 捕获确认。

## 数据与评分

自主性能诊断与决策按[自主能力评测协议 v2](autonomous-evaluation-protocol.zh.md)执行：审计 Agent 自行建立性能预期、取得硬件证据、提出可验证修改、复核指标并决定继续或停止的完整过程。单项加速不代替诊断能力；外部补采不计入自主成绩。原生与增强配置分开，先验证官方示例，再迁移任务。

没有框架实验产生前，结果字段保持 null。失败与 unsupported 分开保留。
正确性不通过的结果不进入速度排名；支持任务的性能和全部分配任务的覆盖率分别报告。
对生成 kernel 的源码、实际调用、局部速度与完整 block 收益建立独立证据链。

`smoke` 的合成 fixture、公开 seed 和固定临时容差只用于接入。
当前 runner 直接 import 可信候选，不防止进程内篡改，不执行生产级反作弊隔离。
正式 evaluator 要在模型/Agent 工作区之外运行；拒绝基线/计时篡改和测试答案读取。
当前 Python runner 也不负责终止死锁 CUDA kernel；GPU 冒烟须由外部会话设置超时与恢复策略。

## GPU API 来源

- [torch.compile 2.12](https://docs.pytorch.org/docs/2.12/generated/torch.compile.html)
- [torch.profiler 2.12](https://docs.pytorch.org/docs/2.12/profiler.html)
- [CUDA Event 2.12](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.Event.html)

实际目标是 NVIDIA 的 2.12.0a0 构建；稳定版 API 文档只是接口核对，兼容性必须在目标机器验证。
先记录工具版本，再按目标机 nsys help 核对参数。不要从版本号推断 B300 kernel 已正常执行。
