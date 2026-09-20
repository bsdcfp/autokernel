# AutoKernel 官方示例复现 · r01

2026-09-20。状态：准备启动，尚无实验结果。

- 目的：原样验证官方工具链与 Agent 工作流；不修复 review 发现的问题。
- 上游：`78435821cc3d5756ba6ee1785c397f6d8fa8c90d`，从 GitHub 已备份提交导出干净目录，运行前后核对源码哈希。
- 示例：`models/llama_7b.py` 的紧凑 `LlamaModel`，输入 `[1,512]`，float16；不是 7B 类。
- GPU：物理第 4 张卡，UUID `GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8`；启动前核对无计算进程。
- 环境：沿用已安装的 wan-autokernel 环境，`uv run` 禁止自动同步依赖。保留原 NVIDIA PyTorch 构建，不替换依赖，不更改 Python 导入规则。
- 第一阶段：原样执行 README 的 `prepare.py`、`profile.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,512 --dtype float16`、`extract.py --top 5`、`bench.py`，每步最多 420 秒；记录日志和退出码。缺少前置产物则停止依赖步骤。
- 第二阶段：第一阶段验证通过后，才进入 `program.md` 原生自主循环；启动前另记预算与宿主配置。协调方不提供 review 诊断或修复提示，不修改 AutoKernel 源码。原生优化产生的候选与框架实现分别记录。
- 第三阶段：原生回接结果与外部独立验收分开；检查真实替换，不以原生成功标签直接判定能力。
- 失败处理：保留首个失败，不改脚本、门限、模型或工作流使其通过。若原生示例被阻塞，报告具体失败阶段，不冒充 Agent 已完成优化。

远端目录：`/home/work/video_posttrain/fuping.chu/agent_projects/autonomous_trials/autokernel-official-llama-r01/`。本记录是执行配置，不是测量结果。
