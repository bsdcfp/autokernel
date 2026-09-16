#!/usr/bin/env bash
set -euo pipefail
WAN_ENV_ROOT=/home/work/video_posttrain/fuping.chu/agent_projects/envs
WAN_ENV="$WAN_ENV_ROOT/wan-autokernel"
mkdir -p "$WAN_ENV_ROOT"
if [[ ! -x "$WAN_ENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$WAN_ENV"
fi
"$WAN_ENV/bin/python" -c 'import torch,triton; print("torch",torch.__version__,"CUDA",torch.version.cuda,"triton",triton.__version__)'
# Preserve the existing NVIDIA PyTorch/CUDA build. No uv sync or pip upgrade.
