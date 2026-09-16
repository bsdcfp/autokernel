# Wan block 算子清单与 AutoKernel 首轮任务

范围：官方 Wan2.1 T2V-1.3B 单个 Transformer block，qk_norm=True、cross_attn_norm=True。
源码 revision 为 `9737cba9c1c3c4d04b33fcad41c111989865d315`。

## 源码语义调用计数

| 类别 | 次数 | 位置 |
|---|---:|---|
| Linear（含 bias） | 10 | self-attention 的 Q/K/V/O 四次；cross-attention 四次；FFN 两次 |
| LayerNorm | 3 | self-attention、cross-attention、FFN 之前；cross-attention 前带 affine |
| RMSNorm | 4 | self-attention Q/K、cross-attention Q/K |
| 3D RoPE | 2 | self-attention 的 Q 和 K |
| Attention 核心调用 | 2 | 一次 self-attention、一次 text cross-attention |
| GELU（tanh 近似） | 1 | FFN 两层 Linear 之间 |
| Modulation（scale/shift） | 2 | self-attention 和 FFN 输入 |
| Gated residual（x + y * gate） | 2 | self-attention 和 FFN 输出 |
| 普通 residual add | 1 | cross-attention 输出 |
| 合计 | 27 | 按本表定义的主要语义调用；不是 kernel launch 数 |

此外还有一次 `self.modulation + e` 条件参数加法、chunk/view/reshape、类型转换、RoPE 的频率构造和 padding 处理。这些操作也可能产生 GPU 开销。若把条件参数加法作为单独语义调用，则为 28 处；该数字不能当作穷尽所有 ATen 操作的数量。
Attention 内部还可展开为矩阵乘、softmax 等，编译后又可能融合。因此实际 PyTorch 算子数和 GPU kernel 次数必须由 trace 给出；其它 Wan 版本或 I2V 路径需重新统计。

## 首轮 AutoKernel：六个互补任务（用户已确认）

| 次序 | 任务 ID | 任务 | 主要考察 |
|---|---|---|---|
| 1 | t1-norm-modulation | LayerNorm + modulation | reduction 与逐元素融合，保留中间精度 |
| 2 | wan-gated-residual | gated residual | 简单融合，检查能否超过已有编译器优化 |
| 3 | wan-rmsnorm | Q/K RMSNorm | reduction、dtype 与 affine 语义 |
| 4 | wan-rope3d | Q/K 的 3D RoPE | 三维坐标、精度与布局 |
| 5 | t2-qknorm-rope | QK RMSNorm + 3D RoPE | 跨算子融合、中间张量消除 |
| 6 | wan-linear-gelu | FFN Linear + bias + GELU | GEMM 与 epilogue 融合，区别于前面的带宽型任务 |

2026-09-16 用户确认以上六项作为首轮范围，使用 AutoKernel 与物理 GPU 3；此确认不表示实验已运行。

共用任务 shape 与数据来源，但各任务使用真实自身的输入契约。例如 cross-attention K 的长度来自文本，不应伪造为全部视频序列长度；FFN 中间维为 8960。
Q/K RMSNorm 在 1536 维上执行，然后才 reshape 为 12 × 128，不能拿 Llama 的按 head 归一化实现直接替换。FFN 激活是 GELU(tanh)，不是 SwiGLU。

Self-attention、cross-attention 核心及整个 FFN/完整 block 留到后续阶段。正式框架比较仍须回到完整 block 验证收益。

首轮配置 `configs/autokernel-first.json`，只分配 AutoKernel；5 框架原 pilot 配置保留。三次重复和 60 分钟/次目前是预算草案，计划不自动启动搜索。T1 有 exploratory runner，其余任务目前只有规格，不能将准备完成误报为实验完成。

## GPU 分配

用户指定“第 4 张卡”，按一基序号映射到 nvidia-smi index=3。执行前核对 UUID 和占用，用 UUID 绑定 `CUDA_VISIBLE_DEVICES`，进程内仅见逻辑 cuda:0。不要把物理 index=3 和屏蔽后的逻辑 index 混用。索引/UUID 核对记录保存在运行配置中。

## 来源

- [官方模型源码](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py)
- [1.3B 配置](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/configs/wan_t2v_1_3B.py)
