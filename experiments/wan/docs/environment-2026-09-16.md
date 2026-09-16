# AIS B300 环境初查 · 2026-09-16

通过用户提供的 AIS WebShell 页面只读检查。隔离 Chrome helper 未找到终端；应用内浏览器的已登录页面成功连接，普通键盘与粘贴输入均已用 `pwd` 验证。未保存认证 iframe URL、token 或 cookie。

## 已直接观察

| 项目 | 结果 |
|---|---|
| GPU | 8 × NVIDIA B300 SXM6 AC |
| 每卡 memory.total | 275040 MiB（nvidia-smi 原始单位） |
| 检查时 GPU 状态 | 每卡 memory.used=0 MiB、utilization.gpu=0%；仅为检查时快照 |
| 驱动 | 580.159.04 |
| 默认 Python | 3.12.3 |
| 默认 PyTorch | 2.12.0a0+0291f960b6.nv26.04.48445190 |
| PyTorch CUDA | 13.2 |
| Triton | 3.6.0 |
| 设备 capability | (10, 3) |
| torch 编译架构列表 | sm_75, sm_80, sm_86, sm_90, sm_100, sm_120, compute_120 |
| nsys | 2026.2.1.210-262137639646v0 |
| Codex CLI | 0.153.4 |
| Claude Code | 2.1.270 |
| 其它存在的命令 | ncu、nvcc、tmux、git |

架构列表不含 sm_103 这一名字，不能据此直接判断可用或不可用；需要实际 CUDA/Inductor/Triton 冒烟验证。没有从版本号推断性能支持。

## 用户指定路径与检查

用户随后指定第 4 张物理卡。已通过 nvidia-smi 核对 index=3，UUID 为 `GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8`。后续一次检查为 4 MiB、0% utilization、无运行进程。runner 现要求用该 UUID 设置 CUDA_VISIBLE_DEVICES，并使用进程逻辑设备 0；每次正式运行前仍需重新确认占用。

- 代码根：`/home/work/video_posttrain/fuping.chu/agent_projects`，检查时为空目录。
- 虚拟环境根：`/home/work/video_posttrain/fuping.chu/agent_projects/envs`，尚未观察到已有环境。
- 模型根：`/home/work/video_posttrain/fuping.chu/models`。
- 首轮采用合成输入，不依赖模型权重。
- 从 `/` 到指定代码根的逐级 AGENTS.md 检查未输出文件；仍遵守用户在会话给出的全局部署规则。

## 未完成

已验证目标机可读取 GitHub。五个公开上游仓库已从各自 GitHub 备份拉取至代码根下的 `upstreams/`，并核对 HEAD 与 `configs/frameworks.lock.json` 全部一致；未运行安装脚本或 Agent。Codex 配置的模型名为 gpt-5.6-sol，reasoning effort 为 xhigh；未读取或输出密钥。

没有上传本地自有项目，没有创建远端虚拟环境，没有运行 Agent 搜索或本项目的 GPU runner。未验证 LLM API 调用、模型下载、框架依赖、GPU 编译、torch.profiler/nsys 实际采集。

用户已指定 GitHub fork `bsdcfp/autokernel`，实验分支 `codex/b300-wan-first-round`；必须先推送代码，再从该备份部署。
