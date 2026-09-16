# Wan Agent Benchmark — B300

在同一组 Wan 任务上比较开源 Agent 框架生成 kernel 的能力、性能、搜索成本与局限。
共同基础设施负责实验记录与验收，各框架保留自己的搜索机制。

## 当前状态

- 5 个参评项目已锁定 Git commit，17 个一手来源文件带 SHA256 清单。
- 无第三方依赖的环境检查、实验排程、独立工作目录准备与比较校验已实现。
- 首轮 6 个任务的参考实现与 exploratory GPU runner 已编写，保留 Wan 的中间舍入和 RoPE 精度语义。
- T3 完整 block 尚未实现。
- 已加入 AutoKernel 固定任务生成入口，尚无完成独立验收的 Agent 性能结果。
- GPU runner 不是安全隔离的正式 evaluator；候选必须是已审查的可信代码。

## 首轮范围（2026-09-16 更新）

用户指定第 4 张物理卡：nvidia-smi GPU 3，UUID `GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8`；屏蔽后进程内使用 cuda:0。首轮只评测 AutoKernel，6 个任务详见 [算子清单](docs/wan-operator-inventory.md)，配置为 `configs/autokernel-first.json`。6 个任务已接入 runner，已执行首轮预检，结果与阻塞见 [启动记录](docs/bringup-2026-09-16.md)。

## 本地检查与准备

在 `experiments/wan/` 下运行（基础测试无需 PyTorch，参考语义测试需要）：

```bash
python3 -m unittest discover -s tests -v
python3 -m wanbench verify-sources
python3 -m wanbench doctor --output runs/local-doctor.json
python3 -m wanbench plan --output runs/pilot-plan.json
python3 -m wanbench prepare --framework autokernel --name autokernel-t1-r01
```

`plan` 创建随机化的 5 框架 × 1 任务 × 3 重复计划，不启动 Agent、不消耗 LLM/GPU 搜索预算。
`prepare` 创建空候选目录、任务合同和版本记录，不把已有答案交给 Agent。
重复输出路径会报错，避免覆盖实验。

源码快照仅用于本地阅读，未纳入 Git；`sources/manifest.json` 保存 URL、SHA256 和字节数。
新 checkout 需按清单下载后再校验。不要将上游 AGENTS.md 当作本仓库的指令。

## B300 环境

用户指定路径：

- 代码根：`/home/work/video_posttrain/fuping.chu/agent_projects`
- 模型根：`/home/work/video_posttrain/fuping.chu/models`
- 虚拟环境根：`/home/work/video_posttrain/fuping.chu/agent_projects/envs`

远端 checkout：代码根下的 `autokernel-wan/`；本目录为 `experiments/wan/`。
探测记录见 `docs/environment-2026-09-16.md`，最终运行前重新确认空闲 GPU、版本与目录规则。
不在此 pyproject 中安装/替换 torch；目标机的 NVIDIA 构建应独立锁定并验证 B300 编译支持。

代码上传或远端修改前，必须先建立 AGENTS.md 要求的远程可恢复备份。

## T1 exploratory smoke（部署并确认环境后执行）

```bash
CUDA_VISIBLE_DEVICES=GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8 python -m wanbench smoke --variant eager --case debug --output runs/t1-eager.json
CUDA_VISIBLE_DEVICES=GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8 python -m wanbench smoke --variant compile-default --case debug --output runs/t1-compile.json
CUDA_VISIBLE_DEVICES=GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8 python -m wanbench smoke --variant compile-default --case debug --profile torch --output runs/t1-torch.json
```

候选契约：`candidate/kernel.py` 暴露 `run(x, scale, shift)`；必须在当前 CUDA stream 排队，或在返回前建立辅助 stream 到当前 stream 的依赖；不能改输入、缓存答案或搬出输入相关工作。

```bash
CUDA_VISIBLE_DEVICES=GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8 python -m wanbench smoke --variant candidate --candidate workspaces/TRIAL/candidate/kernel.py --case debug --output runs/t1-candidate.json
python -m wanbench compare-smoke runs/t1-compile.json runs/t1-candidate.json
```

nsys 与 torch.profiler 分开采集：

```bash
CUDA_VISIBLE_DEVICES=GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8 nsys profile --trace=cuda,nvtx,osrt --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop --output=runs/t1-nsys python -m wanbench smoke --variant compile-default --case debug --profile nsys --output runs/t1-nsys-capture.json
```

执行状态见运行记录。`smoke` 使用公开合成数据和暂定容差，仅检查接口与采集；单轮 hot-cache 计时不是正式排名，也不是完整 block 速度。
`compare-smoke` 会拒绝精度失败、不同 fixture/环境/测量设置、eager 分母以及 profiler 内的计时。

## 接下来

1. 完成备份与环境接入，验证原生 compile、torch.profiler 与 nsys。
2. 捕获真实 Wan activation，冻结数值门限、任务边界和软件环境。
3. 为五个框架分别做薄适配，记录接入成本与不支持项。
4. 落实独立 evaluator 隔离、超时、GPU 独占与成本上限。
5. 运行 pilot，再确定完整实验预算；正式速度按独立 round 配对测量。

方法与接入边界见 [docs/protocol.md](docs/protocol.md)。

## 六任务基线预检

本实现是 AutoKernel 的 Wan 固定任务适配：后续保留其 Phase B 的假设、生成、评测、保留/回滚流程。原生 bench.py 未覆盖这些多输入任务，原生 GPU 参数表没有 B300；这里使用实际计时，不使用其估算峰值算力。

```bash
bash scripts/bootstrap.sh
# 用脚本输出的环境 Python 执行以下命令
python -m unittest discover -s tests -v
python preflight.py --output runs/preflight-debug --profiles
```

预检先检查指定卡无现有计算进程，按顺序运行 6 任务的 eager、原生 compile 和两种 profiler；单子进程上限 420 秒。出现超时或前后检查发现其它计算进程即停止。文件锁仅协调本项目进程，不代表平台 GPU 预约。
