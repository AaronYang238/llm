# 术语表 · 符号表 · 缩写速查

> 全书术语集中查阅处。每条给**一句话解释 + 去哪一章看透**。被某个词劝退时翻这里，先拿到"钩子"，再去对应阶段补细节。
>
> 用法：按主题分组（主题本身就是索引）。`等宽` 是代码标识符 / `config.json` 字段。

## 一、符号约定（贯穿全书）

公式与代码里反复出现的记号，统一约定如下（首次定义见 [阶段 0 §0.1](chapters/00-先修与硬件基础.md)）：

| 符号 | 含义 | 对应 `config.json` 字段 |
|---|---|---|
| `B` | batch size（并发请求数 / 批大小） | — |
| `S` | 序列长度（token 数） | `max_position_embeddings`（上限） |
| `D` | hidden size（token 向量维度） | `hidden_size` |
| `H` | Q head 数 | `num_attention_heads` |
| `H_kv` | KV head 数（MHA: `=H`；GQA: `<H`；MQA: `=1`） | `num_key_value_heads` |
| `d` | 单 head 维度（通常 `D/H`） | `head_dim` |
| `d_ff` | FFN 中间层维度（LLaMA 约 `2.67×D`） | `intermediate_size` |
| `L` | 层数 | `num_hidden_layers` |
| `V` | 词表大小 | `vocab_size` |
| `g` | GQA 分组数（`H/H_kv`） | — |
| `N` | 并行/通信里的设备数（rank 数） | — |

**两个最常用的口算公式**：

- KV cache 字节 ≈ `2 × L × 2 × B × S × H_kv × d × dtype_bytes`（前 2=K/V，后 2=BF16 字节数）。详见 [阶段 1 §1.2.2](chapters/01-Transformer与单卡推理.md)。
- 参数显存 ≈ `参数量 × dtype_bytes`（BF16=2 字节，故 7B → 14 GB）。详见 [前置篇 P.1.1](chapters/0-上车准备.md)。

---

## 二、术语速查（按主题）

### 基础与硬件（阶段 0）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **SM** | Streaming Multiprocessor | GPU 的"核"，H100 SXM 有 132 个 | 阶段 0 §0.2.1 |
| **Tensor Core** | — | 矩阵乘专用单元，跑 BF16/FP8/INT GEMM，算力随精度位宽翻倍 | 阶段 0 §0.2.1 |
| **SMEM** | Shared Memory | 每 SM 内的高速 scratchpad（H100 可配 228 KB），attention tiling 的硬约束 | 阶段 0 §0.2.1 |
| **HBM** | High Bandwidth Memory | GPU 主存（H100 80 GB / 3.35 TB/s），decode 的带宽瓶颈在它 | 阶段 0 §0.2.2 |
| **TMA** | Tensor Memory Accelerator | Hopper 起的异步 DMA 引擎（SMEM↔HBM），FA3 关键 | 阶段 0 §0.2.1 |
| **Roofline** | — | 判定算子 compute-bound 还是 memory-bound 的图形模型；拐点 ≈ 峰值算力/带宽 | 阶段 0 §0.2.2 |
| **FLOPs / FLOPS** | 运算次数 / 每秒运算次数 | FLOP**s**=活有多少（计数）；FLOP**S**=干得多快（算力速率，如 989 TFLOPS）。耗时 ≈ FLOPs ÷ FLOPS | 阶段 0 §0.2.2 |
| **算术强度** | Arithmetic Intensity | FLOPs（次数）/ 从 HBM 搬的字节数；决定算子在 Roofline 哪一侧 | 阶段 0 §0.2.2 |
| **GEMM** | General Matrix Multiply | 通用矩阵×矩阵乘（`[M,K]×[K,N]`，约 `2MKN` FLOPs）；prefill / 大 batch 的主算子，M 大→复用高→compute-bound | 阶段 0 §0.2.2 |
| **GEMV** | General Matrix-Vector multiply | 矩阵×向量乘，即 GEMM 中 `M=1` 的退化；decode（B=1）每步只算 1 token、读全权重→memory-bound，拼 batch 即升回 GEMM | 阶段 0 §0.2.2 |
| **NVLink / NVSwitch** | — | GPU 间高速点对点链路 / 其 crossbar；节点内 TP/EP 的物理底座 | 阶段 0 §0.2.3 |
| **IB** | InfiniBand | 主流 RDMA 网络（NDR 400 Gb/s）；跨节点通信主力 | 阶段 0 §0.2.3 |
| **NUMA** | Non-Uniform Memory Access | CPU socket 间访存非对称，跨 socket 走 QPI 会掉速 | 阶段 0 §0.5 |
| **FP8 (E4M3/E5M2)** | — | 8bit 浮点，Hopper 起有 Tensor Core；前向用 E4M3、梯度用 E5M2 | 阶段 0 §0.2.4 |
| **CUDA Graph** | — | 把一串 kernel launch 录下来重放，消除 launch 开销 | 阶段 0 §0.3 |

### Transformer 结构（阶段 1）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **token** | — | 文字被切成的最小处理单位（≠ 字）；长度/速度/显存都按它算 | 前置篇 P.1.2 |
| **embedding** | 词向量 | token 编号 → 模型能算的"语义坐标"向量 | 前置篇 P.1.2 |
| **MHA / MQA / GQA / MLA** | 各类 Attention | 标准多头 / 共享 1 个 KV / 分组共享 KV / 低秩压缩 KV | 阶段 1 §1.2.2 |
| **RMSNorm** | Root Mean Square Norm | LayerNorm 的简化版（省均值），现代 LLM 主流归一化 | 阶段 1 §1.2.1 |
| **RoPE** | Rotary Position Embedding | 旋转位置编码，把相对位置编进 attention 内积 | 阶段 1 §1.2.3 |
| **SwiGLU / GeGLU** | 门控 FFN | `SiLU/GELU(W_gate·x) ⊙ (W_up·x)`，比经典 FFN 多一个门控 | 阶段 1 §1.2.1 |
| **MoE** | Mixture of Experts | 把 FFN 拆成多个专家，每 token 只激活 top-k 个 | 阶段 1 §1.2.4、阶段 9 §9.6 |
| **prefill / decode** | 推理两阶段 | 一次读完 prompt（吃算力）/ 逐 token 自回归生成（吃带宽） | 前置篇 P.1.4、阶段 1 §1.5 |
| **KV cache** | — | 历史 token 的 K/V 缓存，避免 decode 重算；显存第一矛盾 | 前置篇 P.1.5、阶段 5 |
| **采样** | greedy/top-k/top-p/temperature… | 从 logits 概率里挑下一个 token 的策略 | 阶段 1 §1.2.5 |

### 并行策略（阶段 2）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **DP** | Data Parallelism | 每卡一份完整模型，batch 切 N 份；梯度 AllReduce | 阶段 2 §2.2.1 |
| **TP** | Tensor Parallelism | 把每层权重切到多卡（Megatron 列/行并行）；`TP ≤ H_kv` | 阶段 2 §2.2.2 |
| **PP** | Pipeline Parallelism | 按层切成 stage 流水（1F1B / DualPipe）；有 bubble | 阶段 2 §2.2.4 |
| **EP** | Expert Parallelism | MoE 专家分散到多卡，dispatch/combine 走 All-to-All | 阶段 2 §2.2.5 |
| **CP** | Context Parallelism | 序列切到多卡（Ring/Ulysses），长序列用 | 阶段 2 §2.2.6 |
| **SP** | Sequence Parallelism | 与 TP 配套，在 LN/Dropout 处切序列省激活显存 | 阶段 2 §2.2.3 |
| **ZeRO 1/2/3** | Zero Redundancy Optimizer | 沿 DP 维分片优化器状态/梯度/参数（DeepSpeed） | 阶段 2 §2.2.8 |
| **FSDP / FSDP2** | Fully Sharded Data Parallel | ZeRO-3 的 PyTorch 原生实现；`FULL_SHARD` vs `HYBRID_SHARD` | 阶段 2 §2.2.8、阶段 7 §7.3 |
| **DeviceMesh / DTensor** | — | 把 rank 组织成多维网格 / 描述张量怎么切的分布式张量 | 阶段 7 §7.3 |

### 集合通信（阶段 3）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **AllReduce / AllGather / ReduceScatter / All-to-All** | 集合通信原语 | TP/DP 用 AllReduce、EP 用 All-to-All… | 阶段 2 §2.1、阶段 3 §3.2 |
| **NCCL** | NVIDIA Collective Comm Lib | GPU 集合通信标准库；Ring/Tree/NVLS 算法 | 阶段 3 §3.2/3.3 |
| **busbw / algbw** | bus / algorithm bandwidth | 调优只看 busbw（反映链路用满没），algbw 会误导 | 阶段 3 §3.2.2 |
| **SHARP / NVLS** | — | 在 switch 内做 reduction，省一轮跨网/跨卡往返 | 阶段 3 §3.3.3 |
| **IBGDA** | IB GPUDirect Async | GPU kernel 内直接发 RDMA 请求，CPU 退出热路径 | 阶段 3 §3.3.3 |
| **NVSHMEM** | — | 单边通信（put/get）原语，DeepEP 的底座 | 阶段 3 §3.4 |
| **DeepEP** | — | DeepSeek 的 MoE All-to-All 专用 kernel（low-latency ~8μs） | 阶段 3 §3.4.3 |
| **PD 分离的 KV 传输** | Mooncake / NIXL / LMCache | prefill 的 KV 经 RDMA 传到 decode 节点 | 阶段 3 §3.5、阶段 5 §5.6 |

### 算子 / Kernel（阶段 4）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **FlashAttention (v1/2/3)** | — | IO-aware tiling + online softmax，不 materialize `[S,S]` 矩阵 | 阶段 4 §4.2 |
| **online softmax** | — | 分块增量算 softmax 的数值稳定算法，FA 的算法基石 | 阶段 4 §4.2.2 |
| **PagedAttention** | — | KV 分页（block + block table），消除碎片；vLLM 招牌 | 阶段 4 §4.3 |
| **FlashInfer** | — | 推理专用 attention 库：变长 batch、decode 专用、CUDA Graph 友好 | 阶段 4 §4.4 |
| **FlashMLA** | — | DeepSeek MLA 在 Hopper 上的专用 kernel | 阶段 4 §4.5 |
| **矩阵吸收** | Matrix Absorption | MLA 把升维矩阵吸收进 Q/O 投影，直接在 latent 维算 | 阶段 4 §4.5.2 |
| **Triton** | — | Python 写 GPU kernel 的 DSL，适合融合周边小算子 | 阶段 4 §4.6 |
| **CUTLASS / cuBLASLt** | — | GEMM 模板库 / 运行时库；epilogue 融合、FP8 GEMM | 阶段 4 §4.7 |
| **async-TP / flux** | — | 把 TP 的 AllReduce 与 GEMM 重叠（通信藏进计算） | 阶段 4 §4.8 |

### KV cache 与调度（阶段 5）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **block / page** | KV block | 固定大小的 KV 分配单元（典型 16 token） | 阶段 5 §5.2 |
| **continuous batching** | 连续批处理 | 调度粒度降到"每 step"，完成就退、空位即补（Orca） | 阶段 5 §5.3 |
| **chunked prefill** | — | 把长 prefill 切块、与 decode 混跑，稳住 TPOT（SARATHI） | 阶段 5 §5.4 |
| **prefix cache / RadixAttention** | 前缀缓存 | 相同前缀的 KV 跨请求复用（vLLM hash / SGLang 基数树） | 阶段 5 §5.5 |
| **PD 分离** | Prefill/Decode Disaggregation | prefill 与 decode 跑在不同 GPU，各用最优配置 | 阶段 5 §5.6 |
| **preemption** | 抢占 | 显存不足时把某请求 KV 换出（swap）或丢弃（recompute） | 阶段 5 §5.3.3 |
| **KV offload** | — | 把冷 KV 搬到 CPU/NVMe/远端（LMCache） | 阶段 5 §5.8 |

### 推理引擎 / 服务（阶段 6、10）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **vLLM / SGLang** | — | 两大开源推理引擎：通用最广 / 前缀复用 + DeepSeek 最优 | 阶段 6 §6.3/6.4 |
| **DP attention** | — | MLA 的 KV 太小不宜 TP 切，attention 改走 DP（每卡完整 KV） | 阶段 6 §6.4.3、阶段 9 §9.8 |
| **OpenAI 兼容 API** | — | `/v1/chat/completions` 等事实标准接口 | 阶段 10 §10.2 |
| **structured output** | 结构化输出 | softmax 前 mask logits，约束生成合法 JSON/正则（xgrammar） | 阶段 10 §10.3 |
| **multi-LoRA serving** | 多 LoRA | 一个基座 + 多个 LoRA 同时服务，分段 kernel 处理异构 batch | 阶段 10 §10.4 |
| **AI Gateway** | — | 引擎前的网关：路由/限流/鉴权/failover（LiteLLM 等） | 阶段 10 §10.6 |

### 量化与加速（阶段 8）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **PTQ** | Post-Training Quantization | 训练后量化（不重训）：GPTQ / AWQ / SmoothQuant / HQQ | 阶段 8 §8.3 |
| **outlier** | 离群值 | 激活里少数远大于均值的元素，量化掉点的头号原因 | 阶段 8 §8.2.3 |
| **WxAy** | — | 权重 x bit、激活 y bit（W4A16 / W8A8 / W4A8） | 阶段 8 §8.5 |
| **Marlin / Machete** | — | vLLM 的高性能 W4A16 GEMM kernel（反量化融进 GEMM） | 阶段 8 §8.5.2 |
| **KIVI** | — | KV 量化：Key 走 per-channel、Value 走 per-token | 阶段 8 §8.6、阶段 5 §5.7 |
| **投机解码** | Speculative Decoding | draft 猜 k 个、target 并行验证；无损，加速比≈接受率 | 阶段 8 §8.7 |
| **EAGLE / Medusa / MTP** | — | 投机解码的 draft 来源：特征级 / 多头 / DeepSeek 原生多 token | 阶段 8 §8.7、阶段 9 §9.7 |
| **稀疏化** | Sparsity | MoE 路由 / 激活稀疏 / 稀疏 attention（NSA、MoBA） | 阶段 8 §8.8 |
| **蒸馏 / KD** | Knowledge Distillation | 训练小 student 模仿大 teacher 的软分布（暗知识）；换更小模型、可再量化（R1-Distill） | 阶段 8 §8.9 |

### 长上下文与 DeepSeek 体系（阶段 9）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **位置外推** | PI / NTK-aware / YaRN / LongRoPE | 让短训练泛化到长推理（YaRN 是主流） | 阶段 9 §9.3 |
| **Ring / Ulysses Attention** | — | 序列并行 attention：K/V 环轮转 / 沿 head 维 All-to-All | 阶段 9 §9.4 |
| **MLA** | Multi-head Latent Attention | 低秩压缩 KV（~1/30）；长上下文 × 大 MoE 的交汇点 | 阶段 9 §9.5、阶段 1 §1.2.2 |
| **DeepSeekMoE** | — | 细粒度专家(256) + 共享专家(1) + loss-free balance 三创新 | 阶段 9 §9.6 |
| **loss-free balance** | — | 用可学习偏置动态平衡 MoE 负载，不加辅助 loss | 阶段 9 §9.6.3 |
| **DualPipe** | — | 双向流水把 bubble 压到近零，并把 MoE 通信藏进气泡 | 阶段 9 §9.7.1 |
| **expert offload** | — | Hot expert 留 GPU、Cold 放 CPU/NVMe，省显存 | 阶段 9 §9.8.3 |

### 性能指标与工具（阶段 5、11）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **TTFT** | Time To First Token | 首 token 延迟，由 prefill 决定 | 阶段 5 §5.1、阶段 10 §10.7 |
| **TPOT / ITL** | Time Per Output Token | 每输出 token 间隔，由 decode 决定 | 阶段 5 §5.1 |
| **goodput** | — | 满足 SLO 前提下的有效吞吐（不是裸 throughput） | 阶段 5 §5.1 |
| **compute / memory / overhead-bound** | 算力 / 带宽 / 开销受限 | 三类瓶颈，决定优化方向（提算力 / 减访存 / 减 launch）；判错方向优化全白做 | 阶段 0 §0.2.2（定义）、阶段 11 §11.2（判读） |
| **torch.profiler / nsys / ncu** | — | 找热点 / 看时序 / 挖单 kernel，由粗到细 | 阶段 11 §11.1 |
| **flight recorder** | — | `TORCH_NCCL_TRACE_BUFFER_SIZE`，定位多卡 hang（collective mismatch） | 阶段 11 §11.6.3 |

### 模型架构（阶段 12）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **六旋钮** | — | attention/FFN/位置/归一化/词表/长上下文，读 config 的框架 | 阶段 12 §12.1 |
| **tie embedding** | 词表共享 | embedding 与 `lm_head` 共享权重（Qwen/Gemma tie，LLaMA 不 tie） | 阶段 12 §12.3 |
| **QK-norm** | — | 对 Q/K 各做一次 RMSNorm，稳大规模训练（Qwen3 起普及） | 阶段 12 §12.3 |
| **sliding window** | SWA | 每 token 只看最近 W 个，KV 上限封顶（Mistral） | 阶段 12 §12.4 |
| **M-RoPE** | 多模态 RoPE | 把位置编码扩到时间/高/宽三维（Qwen-VL） | 阶段 12 §12.7 |

### 集群编排与调度（阶段 13）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **gang scheduling** | 成组调度 | 多卡作业全或无地一起调度，避免部分占卡死锁 | 阶段 13 §13.2.2 |
| **Volcano / Kueue** | — | 给 K8s 补 gang 调度 / 队列 / 配额的两套主流插件 | 阶段 13 §13.3.2 |
| **Slurm** | — | HPC 批调度器，gang / 拓扑 / 高性能网络原生，偏训练 | 阶段 13 §13.4 |
| **Ray** | — | 应用层分布式 Python（task/actor + placement group），偏 RLHF | 阶段 13 §13.5 |
| **placement group** | — | Ray 原子预留一组资源 bundle，表达 gang + 放置策略 | 阶段 13 §13.5.1 |
| **MIG / MPS / time-slicing** | 卡共享三档 | 隔离强度递减：硬件分区 / 进程空间共享 / 纯分时 | 阶段 13 §13.6 |
| **拓扑感知** | topology-aware | 把同作业的卡排进同一 NVLink 域 / rail，避免跨机拖慢 | 阶段 13 §13.7.3 |
| **HPA / KEDA** | — | K8s 弹性伸缩；KEDA 支持按队列长度 / QPS 等自定义指标 | 阶段 13 §13.7.1 |
| **scale-to-zero** | 缩到零 | 无流量缩到 0 副本省卡，代价是权重加载的冷启动延迟 | 阶段 13 §13.7.1 |
| **抢占 / 潮汐复用** | preemption | 高优先级作业踢掉低优先级（被踢者存盘退让），昼推理夜训练 | 阶段 13 §13.7.2 |

### 训练容错与数据/权重管线（阶段 14）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **MTBF** | Mean Time Between Failures | 平均无故障时间；作业级 ≈ 单卡 MTBF / N，千卡作业几小时一次 | 阶段 14 §14.2.1 |
| **可靠性回路** | — | checkpoint + 检测 + 弹性重启 + 从 ckpt 续跑 | 阶段 14 §14.2 |
| **collective mismatch** | — | 某 rank 少/多调一次集合通信致全员 hang，最常见卡死根因 | 阶段 14 §14.3.2 |
| **straggler** | 拖后腿 rank | 某 rank 异常慢、不报错，同步训练全员等它 | 阶段 14 §14.3.3 |
| **SDC** | Silent Data Corruption | 硬件算错却不报错，最隐蔽，靠主动一致性校验揪出 | 阶段 14 §14.3.5 |
| **torch elastic / rendezvous** | — | `torchrun` 弹性启动；节点变动后各 worker 重新"点名组队"续跑 | 阶段 14 §14.4 |
| **异步 / 分布式 checkpoint** | async / DCP | 快照到 CPU + 后台落盘（勤存少丢）；各 rank 存自己分片、支持 resharding | 阶段 14 §14.5 |
| **流式数据集** | WebDataset / StreamingDataset | 分片存对象存储、边下边训、可断点续读，不全量落盘 | 阶段 14 §14.6 |
| **weight streaming** | — | 逐张量从存储直灌 GPU、下载与拷贝重叠，把冷启动压到秒级 | 阶段 14 §14.7 |

### Post-training / RLHF（阶段 15）

| 术语 | 全称 / 中文 | 一句话 | 出处 |
|---|---|---|---|
| **RLHF** | 基于人类反馈的强化学习 | 用奖励信号把模型往人类偏好推；PPO 要四模型 + rollout 回路 | 阶段 15 §15.2 |
| **PPO / GRPO / DPO** | — | 对齐算法三档：PPO 四模型最重、GRPO 省 critic、DPO 去 rollout 最简 | 阶段 15 §15.3.1 |
| **rollout** | 生成相位 | actor 对 prompt 批量生成回答（一次推理，memory-bound） | 阶段 15 §15.3.3 |
| **actor / critic / reward / reference** | — | RLHF 四模型：策略 / 价值 / 奖励 / KL 锚 | 阶段 15 §15.2 |
| **colocate / disaggregate** | 共置 / 分离 | 多模型共卡分时（省卡）vs 分卡常驻（高吞吐） | 阶段 15 §15.3.2 |
| **两相位** | rollout↔train | 生成（memory-bound）↔ 反向（compute-bound）交替，复用是吞吐命门 | 阶段 15 §15.3.3 |
| **权重同步** | weight sync | 训练新 actor 权重灌回 rollout 引擎；colocate 时几乎免费 | 阶段 15 §15.3.4 |
| **KL penalty** | KL 惩罚 | 限制策略别偏离 reference，RLHF 的安全带，防 reward hacking | 阶段 15 §15.4.3 |
| **reward hacking** | 奖励钻空 | 钻奖励模型漏洞而非真变好；信号是 reward 升而 KL 爆 | 阶段 15 §15.6 |
| **veRL / OpenRLHF** | — | 主流 RLHF 框架，建在 Ray + vLLM rollout + 训练后端之上 | 阶段 15 §15.3.5 |

---

> 没找到某个词？它大概率在对应阶段的 "N.1 核心概念与术语" 里有更细的中英对照。本表只收录跨章高频、需要集中查阅的核心术语。
