# Wan 精度失败排查：编译器舍入边界与 QK 归约

日期：2026-09-16。状态：FFN 根因已复现；QK 大部分误差已定位，长序列单点超差仍待验证。

## 1. 结论与边界

1. **FFN compile 失败已找到可复现原因**：安装版本 Inductor 把 `addmm` 改写成 BF16 `mm` 加后续融合的 bias/GELU，改变了舍入位置。关闭 pattern matcher 或使 Linear 中间输出可见，在 debug/medium、seed 17/18/19-shifted 共六组输入上均与 eager **逐元素一致**。long 跟进尚未执行。
2. **QK compile 的主要差异来自消除中间精度转换**：保留 precision casts 后，long seed 18 的超门限元素从 5,951 降至 1；未完全通过。
3. **QK 候选还受浮点融合与归约边界影响**：禁用 FMA 的诊断消融让 medium seed 18 从 1 个超差降至 0，但 long seed 18 仍有 1 个。不能把该消融称为完整修复。
4. 所有结果继续使用原门限 `abs(error) <= .01 + .01*abs(eager)`。没有修改参考实现、候选映射或原生 compile 基线，也没有产生新的有效性能加速比。

这是固定合成输入上的数值排查，不是完整 Wan 模型质量结论。诊断代码由本次排查编写，不计作 AutoKernel 自主生成的优化算子；本轮没有新增 Agent 生成调用。

## 2. FFN 的证据链

环境与首轮相同：B300 物理 GPU 3，NVIDIA torch `2.12.0a0+0291f960b6.nv26.04.48445190`，CUDA 13.2；TF32 关闭；BF16 输入；GELU 为 tanh 近似。

### 生成代码与安装源码

完整编译图的生成代码先执行：

```python
# 示意：实际 output_code 日志保留在远端。
buf0 = empty_strided_cuda((S, 8960), ..., torch.bfloat16)
extern_kernels.mm(..., out=buf0)
# Triton 随后把 buf0 和 bias 加载为 FP32，融合 bias + GELU，写回 BF16。
```

单独编译 Linear 则保留 `extern_kernels.addmm(bias, ..., out=buf0)`。安装源码 `torch/_inductor/fx_passes/post_grad.py:1510` 起定义 `should_prefer_unfused_addmm`：GPU addmm 的所有用户为 pointwise 时，匹配 `unfuse_bias_add_to_pointwise`，将其替换为 `mm_result = x1 @ x2; return inp + mm_result`。源码规则、生成代码和对照结果相互吻合。

| 路径 | debug 17/18/19-shifted | medium 17/18/19-shifted | long 跟进 |
|---|---|---|---|
| 原生完整 compile | 精度失败 | 精度失败 | 初批诊断失败 |
| `emulate_precision_casts=True` | 仍失败 | 仍失败 | 初批诊断仍失败 |
| `pattern_matcher=False` | 全部与 eager 逐元素一致 | 全部与 eager 逐元素一致 | 未运行 |
| 返回 GELU 和 Linear 中间结果，再取 GELU | 全部与 eager 逐元素一致 | 全部与 eager 逐元素一致 | 未运行 |
| 显式模拟 BF16 mm→FP32 bias+GELU→BF16 | 全部与原生完整 compile 逐元素一致 | 全部与原生完整 compile 逐元素一致 | 未运行 |

中等序列诊断进程结束时检测到其他 GPU 任务；该行仅用于数值诊断，不用于性能结论。

### 一个可读的数值反例

Debug、seed 17、row 43 / column 736：

| 量 | 值 |
|---|---:|
| FP64 dot（不含 bias） | 2.0855004711193033 |
| BF16 bias | -0.0147705078125 |
| BF16 mm | 2.078125 |
| eager Linear BF16 | 2.078125 |
| eager GELU | 2.046875 |
| 完整 compile GELU | 2.015625 |
| 显式模拟 GELU | 2.015625 |
| 误差 / 当前容许误差 | 1.0256410256410255 |

虽然此位置 mm 与 eager Linear 恰好舍入到相同值，编译图仍在已舍入 mm 后加上负 bias，再计算 GELU，因此最终不同。`emulate_precision_casts` 没有阻止这个 addmm 图重写。

**修复候选的含义**：关闭全部 pattern matcher 范围较大；使 Linear 中间结果可见改变输出图；分开编译 Linear/GELU 改变融合边界。这些是兼容性路径，必须另命名、补齐尺度验证并重新计时，不能替换原生默认 compile 后沿用旧加速比。

## 3. QK 的对照数据

下表为超门限元素数，q/k 合计；0 表示通过当前门限，不表示 bitwise 相等。

| case / seed | 原生 compile | 保留 casts | 原候选 | 原候选禁用 FMA 消融 |
|---|---:|---:|---:|---:|
| debug / 17 | 24 | 0 | 0 | 0 |
| debug / 18 | 15 | 0 | 0 | 0 |
| medium / 17 | 1194 | 0 | 0 | 0 |
| medium / 18 | 1161 | 0 | 1 | 0 |
| medium / 19-shifted | 1225 | 0 | 0 | 0 |
| long / 17 | 5085 | 0 | 0 | 0 |
| long / 18 | 5951 | 1 | 1 | 1 |
| long / 19-shifted | 5899 | 0 | 0 | 0 |

重复 seed 17 得到相同结果。medium/long 的首轮候选失败由 seed 18 触发，不能把 seed 17 的通过误读为执行不确定性。

安装版 `torch/_inductor/config.py` 的 `emulate_precision_casts` 默认 False，注释说明低精度融合中可能移除 downcast→upcast 对。开启后大部分差异消失，支持舍入边界被消除这一解释。候选源码则已显式保留 norm→BF16、weight→BF16 和 FP64 RoPE。

### 已观察到的舍入敏感性；尚非残余超差的最终因果证明

初批 stage probe 的 long、seed 17、q row 1119 / col 32：

- FP64 归一化值：`-0.09448242182143599`。
- 相邻 BF16 值：`-0.0947265625` 和 `-0.09423828125`；中点为 `-0.094482421875`。
- 高精度数值离中点约 `5.36e-11`；候选为后一值，eager 为前一值。
- 初版诊断中的 `norm64.bfloat16()` 也返回前一值。该转换不能当作“直接 FP64 舍入 oracle”；中间若先转 FP32，会落在中点上并触发 ties-to-even。

后续诊断已加入独立的 binary64→BF16 最近偶数舍入，并测试中点及左右相邻 binary64 值，避免 oracle 自己二次舍入。FP64 归约本身仍是数值诊断参考，不是数学精确证明。

**未决问题**：上述是 seed 17 的 norm 差异，不是 long seed 18 最后一个 RoPE 超差坐标。需要提取后者的受影响 pair，比较 eager 全张量归约、逐行 FP32、FP64 归约、直接 BF16 舍入及旋转输出，才能确认是否由归约微差跨 BF16 中点后被旋转抵消放大。现阶段仅为待验证假设，不能声称候选精度问题解决。

## 4. 执行状态与续跑

- `runs/precision-r01/summary.json`：complete，六个诊断 worker 完成；包含原生/保留 casts、候选/禁用 FMA、分阶段探针与 output_code。
- `runs/precision-r02/summary.json`：`stopped-inspect-required`；FFN debug/medium 返回 0、JSON 存在。medium 结束时检测到 GPU 3 上 ComfyUI Python PID 328464，守卫停止，未终止该任务。
- QK 的 eager-norm + compiled-RoPE 分界对照已实现，但尚未运行；该路径也必须与完整融合候选区分。
- 新增 `--tasks` 以只补跑缺失任务；原始 artifact 不覆盖。

第 4 张卡释放后，在激活实验虚拟环境、工作目录 `experiments/wan` 下顺序运行（控制器会重新验证占用和 UUID）：

```sh
python precision_campaign.py --output runs/precision-r03-ffn-long --followup --tasks wan-linear-gelu --cases long
python precision_campaign.py --output runs/precision-r03-qk --followup --tasks t2-qknorm-rope
```

若输出目录已存在，使用新的 run ID；不覆盖历史实验。补跑后的下一步是将具体失败反馈交给 AutoKernel 生成修订，再做独立精度验证与配对计时。不能凭禁用 FMA 或放宽容差宣布修复。

本地回归共 25 项，19 项通过、6 项因本地无 torch 跳过；新增两项直接 BF16 舍入测试通过。目标机此前的 23 项回归已通过；本次新增两项舍入测试部署后均通过。部署后再次查询，第 4 张卡仍有 ComfyUI PID 328464，未启动补跑。

## 5. 证据位置与可恢复性

代码：[precision_diagnose.py](../precision_diagnose.py)、[precision_followup.py](../precision_followup.py)、[precision_campaign.py](../precision_campaign.py)。基准与候选来源见[首轮报告](round-one-results-2026-09-16.zh.md)。

所有自己的诊断代码先推送 GitHub，再部署。**原始 JSON、生成代码日志、远端候选源码仍在远端，尚未回传备份**；本报告是从终端核对的摘要，不能替代原始结果包。远端根为 `/home/work/video_posttrain/fuping.chu/agent_projects/autokernel-wan/experiments/wan`。
