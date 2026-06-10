# 大模型系统工程教材（LLM Systems for Engineers）

> 一套面向工程师的中文大模型系统教材：从 GPU 硬件、Transformer 结构出发，逐层向上打通
> **并行策略 → 集合通信 → 算子内核 → KV 调度 → 推理引擎 → 训练框架 → 量化加速 → 长上下文 MoE → 生产服务 → 性能调优 → 模型架构** 的完整知识链。
> 不是论文综述，也不是 API 文档——**原理讲透 + 图解直观 + 可跑代码 + 工程调优**。

## 这本书是什么

- **读者画像**：有 PyTorch 基础、用过 vLLM / HuggingFace 但没系统读过源码，想从"会用"进阶到"会改、会调、会讲"的工程师。
- **形态**：13 章正文（阶段 0–12）+ 23 张矢量插图（SVG），可单章查阅、也可串读，能直接拼成一本电子书。
- **风格约束**：图先于文；代码必跑、必标依赖版本；任何吞吐 / 延迟数字必标硬件（如 "H100 80GB, NVLink"）；剖析源码必给文件路径与类名。

## 怎么读

- **线性精读**：按阶段 0 → 12 顺序读，每章建立在前面之上，用大量交叉引用织成一张网。
- **按需查阅**：带着真实问题（OOM、TTFT 不达标、吞吐打不上去、8 卡只跑出 3 卡速度）直接跳到对应阶段——每章开头都有"一句话定位"。
- **章节类型**：A 类（基础原理，0/1/4/5）、B 类（并列对比 + 选型矩阵，2/6/7/8/12）、C 类（综合案例，9 以 DeepSeek-V3 为主线）、D 类（工具 cookbook，11）、A+D 混合（10）。
- 全部插图在 `svg/`，章节用相对路径 `../svg/NN-topic.svg` 引用。

## 目录结构

```
.
├── chapters/   # 14 章正文（前置篇 + 阶段 0–12），命名 NN-kebab-slug.md
├── svg/        # 24 张矢量插图，编号与章节对应
├── examples/   # 书里代码的"可跑版"——clone 即跑，不用从 markdown 手抄
└── README.md   # 本文件：总目录 + 完成状态 + 各章导航
```

> **想动手？** 看 [`examples/`](examples/)——前置篇的第一个推理、阶段 1 的手写 LLaMA、阶段 2 的 rank 拓扑模拟器、阶段 4 的 Triton kernel 等都在那里，其中 `02_rank_sim.py` 和 `P_first_run.py` **不挑硬件、立刻能跑**。

## 进度总览

| 阶段 | 主题 | 章节 | 状态 |
|---|---|---|---|
| **前置** | **上车准备：基础概念 + 环境搭建**（写给小白，老手可跳过） | `0-onboarding` | ✓ |
| 0 | 先修与硬件基础 | `00-prereq-hardware` | ✓ |
| 1 | Transformer 与单卡推理 | `01-transformer-basics` | ✓ |
| 2 | 并行策略系统化 | `02-parallelism` | ✓ |
| 3 | 集合通信与通信库 | `03-collective-comm` | ✓ |
| 4 | 核心算子与高性能 Kernel | `04-kernels` | ✓ |
| 5 | KV Cache、调度器与显存 | `05-kv-cache-scheduler` | ✓ |
| 6 | 推理引擎深读 | `06-inference-engines` | ✓ |
| 7 | 训练框架深读 | `07-training-frameworks` | ✓ |
| 8 | 量化、蒸馏与加速 | `08-quantization` | ✓ |
| 9 | 长上下文与 MoE 专题 | `09-long-context-moe` | ✓ |
| 10 | 生产化服务与多模态 | `10-serving-multimodal` | ✓ |
| 11 | 性能分析与调优工具 | `11-profiling` | ✓ |
| 12 | 代表模型架构选读 | `12-model-architectures` | ✓ |

**正文 12 个阶段（13 章 + 23 图）全部定稿。** Capstone 动手项目（P1–P7）与推荐阅读清单持续推进。

下面按阶段列出每章的**对应文件、主图、覆盖的知识点**（带小节定位，可当详细目录查）。

---

## 前置篇｜上车准备：基础概念 + 环境搭建 ✓

> 写给真·小白（已会 vLLM/HuggingFace 的可跳过）：[chapters/0-onboarding.md](chapters/0-onboarding.md)，主图 [svg/24-llm-request-lifecycle.svg](svg/24-llm-request-lifecycle.svg)（一次对话的生命周期）。

- [x] **30 分钟深度学习地基**：模型/参数、token/embedding、训练 vs 推理、**prefill/decode**、KV cache、一张"大图"（P.1）
- [x] **Python / 数学最小集**：会看张量形状 + 知道"主要计算是矩阵乘" + "模型按概率选词"就够（P.2）
- [x] **环境搭建**：CPU/GPU/云/Colab 怎么选、装 PyTorch + transformers、验证 `torch.cuda.is_available()`（P.3）
- [x] **跑通第一个推理**：~25 行 transformers 脚本（CPU 可跑）+ vLLM 起服务（GPU）（P.4）
- [x] **小白最短主线**：路径 A（最快会用）/ B（完整精读）/ C（带问题查）+ FAQ（P.5–P.6）

---

## 阶段 0｜先修与硬件基础 ✓

> 已就位：[chapters/00-prereq-hardware.md](chapters/00-prereq-hardware.md)，主图 [svg/08-gpu-memory-topology.svg](svg/08-gpu-memory-topology.svg)。

- [x] **GPU 体系结构**：SM / Warp / Tensor Core / SMEM / HBM；H100 / H200 / B200 / MI300X 关键参数对比
- [x] **GPU 内存层级与带宽**：寄存器 → SMEM → L2 → HBM 带宽与延迟、Roofline 模型
- [x] **多卡互联拓扑**：NVLink / NVSwitch / NVL72、PCIe Gen5、IB NDR/XDR、RoCEv2、NVLink-C2C
- [x] **数值精度**：FP32 / TF32 / BF16 / FP16 / FP8 (E4M3, E5M2) / INT8 / INT4，溢出与缩放
- [x] **CUDA 基础**：kernel launch、stream、event、graph capture；`cudaMemcpyAsync` 与 P2P
- [x] **NUMA / PCIe 拓扑**：`nvidia-smi topo -m`、`numactl`、`hwloc-ls` 解读

---

## 阶段 1｜Transformer 与单卡推理基础 ✓

> 已就位：[chapters/01-transformer-basics.md](chapters/01-transformer-basics.md)，主图 [svg/09-transformer-block.svg](svg/09-transformer-block.svg)。

- [x] **核心模块**：Embedding、RMSNorm/LayerNorm、Linear、Residual
- [x] **Attention 家族**：MHA → MQA → GQA → MLA（DeepSeek）→ Differential / NSA
- [x] **位置编码**：Sinusoidal、ALiBi、RoPE、YaRN、NTK-aware、LongRoPE
- [x] **FFN 家族**：标准 FFN、SwiGLU、GeGLU；MoE 的 Expert FFN
- [x] **MoE 路由**：Top-K、Switch、Expert Choice、Loss-Free Balance（DeepSeek-V3）（仅列名，工程细节留阶段 9）
- [x] **采样与解码**：greedy / top-k / top-p / temperature / min-p / typical / DRY
- [x] **手撕一遍**：用 PyTorch 实现一个 ~150 行的 LLaMA forward（含 RoPE + GQA + SwiGLU）

---

## 阶段 2｜并行策略系统化

> 已就位 ✓：[chapters/02-parallelism.md](chapters/02-parallelism.md) 2.0–2.4 节覆盖 DP/TP/SP/PP/EP/CP 六种并行与多维编排；2.2.8 节深入 ZeRO 1/2/3、FSDP1/FSDP2、`HYBRID_SHARD`；2.4.4 节 3D 实战含可跑 rank 拓扑模拟器 + Megatron 启动配置（真机指标待硬件校准）。

- [x] **DP（数据并行）**：`DDP` 梯度桶、`gradient_as_bucket_view`；与 ZeRO 的关系（02 §2.2.1）
- [x] **ZeRO 1/2/3 与 FSDP**：参数/梯度/优化器状态分片；`FULL_SHARD` vs `HYBRID_SHARD`；`torch.distributed.fsdp` 与 FSDP2 (`fully_shard`) 的差异（02 §2.2.8）
- [x] **TP（张量并行）**：Megatron 列并行 + 行并行的配对；`g`/`f` 算子的 AllReduce 位置（02 §2.2.2，参见 `svg/02-tp-forward.svg`）
- [x] **SP（序列并行）**：与 TP 配合，省 LN/Dropout 的激活显存（02 §2.2.3）
- [x] **PP（流水并行）**：GPipe、1F1B、Interleaved 1F1B、Zero Bubble、DualPipe（DeepSeek）（02 §2.2.4，参见 `svg/03-pp-1f1b-schedule.svg`）
- [x] **EP（专家并行）**：All-to-All 通信模式、Dispatch/Combine、Token Drop（02 §2.2.5，参见 `svg/04-ep-moe-all2all.svg`）
- [x] **CP（上下文/序列并行）**：Ring Attention、Striped Attention、Ulysses（02 §2.2.6，参见 `svg/05-cp-ring-attention.svg`）
- [x] **多维并行编排**：TP×PP×DP×EP×CP 组合，rank 拓扑映射（02 §2.4，参见 `svg/07-multi-dim-parallel-topology.svg`）
- [x] **3D parallelism 实战**：在 2 节点 ×8 GPU 上跑通 Megatron-LM 训练 LLaMA-7B（02 §2.4.4，含可跑 rank 拓扑模拟器 + 真实启动配置 + 模拟指标；真机实测待硬件就绪校准）

---

## 阶段 3｜集合通信与高性能通信库 ✓

> 已就位：[chapters/03-collective-comm.md](chapters/03-collective-comm.md)，主图 [svg/10-nccl-algos.svg](svg/10-nccl-algos.svg)（四种 AllReduce 算法对比）+ [svg/11-nccl-busbw-curve.svg](svg/11-nccl-busbw-curve.svg)（busbw 随消息大小的典型曲线）。原语层基础参见 [chapters/02-parallelism.md](chapters/02-parallelism.md) §2.1。

- [x] **七大原语**：Broadcast / Reduce / AllReduce / AllGather / ReduceScatter / All-to-All / Send-Recv（02 §2.1.2，参见 `svg/01-collective-primitives.svg`）
- [x] **NCCL 算法**：Ring / Tree / NVLS / Double Binary Tree；`NCCL_ALGO`、`NCCL_PROTO`（02 §2.1.3 + 03 §3.2 busbw 推导与自动选择规则）
- [x] **NCCL 调优**：`NCCL_DEBUG=INFO`、`NCCL_IB_HCA`、`NCCL_P2P_LEVEL`、SHARP、PXN、IBGDA（03 §3.3）
- [x] **NVSHMEM**：单边通信模型；与传统 send/recv 的差异（03 §3.4.1–3.4.2）
- [x] **DeepEP**：MoE All-to-All 内核（dispatch/combine、low-latency vs normal）（03 §3.4.3）
- [x] **PD 分离的 KV 传输**：Mooncake、NIXL、LMCache、`NCCL_RDMA_RW` 的 KV 拷贝路径（03 §3.5，通信侧；KV cache 管理与调度详见阶段 5）
- [x] **微基准**：用 `nccl-tests` 跑 allreduce / alltoall 的 busbw 曲线，画一张随消息大小的图（03 §3.7 + `svg/11`）

---

## 阶段 4｜核心算子与高性能 Kernel ✓

> 已就位：[chapters/04-kernels.md](chapters/04-kernels.md)，主图 [svg/12-flashattn-tiling.svg](svg/12-flashattn-tiling.svg)（FlashAttention tiling + online softmax）+ [svg/13-paged-attention.svg](svg/13-paged-attention.svg)（PagedAttention block table 与 prefix sharing）。

- [x] **Attention kernel 演进**：FlashAttention v1 → v2 → v3 → FlashAttention-3 FP8 / Hopper TMA（04 §4.2，主图 `svg/12-flashattn-tiling.svg`）
- [x] **PagedAttention**：block table、KV 分页布局（04 §4.3，主图 `svg/13-paged-attention.svg`）
- [x] **FlashInfer**：变长 batch、append/decode 专用 kernel、CUDA Graph 友好接口（04 §4.4）
- [x] **FlashMLA**：DeepSeek MLA 在 Hopper 上的低秩 KV cache 加速（04 §4.5）
- [x] **Triton 基础**：写一个 RMSNorm + Fused Residual 的 Triton kernel（04 §4.6，含完整 ~80 行可跑代码 + benchmark）
- [x] **CUTLASS / cuBLASLt**：GEMM epilogue 融合、FP8 GEMM（04 §4.7）
- [x] **All-Reduce 融合**：Norm + AllReduce、Allgather + GEMM（`flux`、`async-TP`）（04 §4.8）

---

## 阶段 5｜KV Cache、调度器与显存管理 ✓

> 已就位：[chapters/05-kv-cache-scheduler.md](chapters/05-kv-cache-scheduler.md)，主图 [svg/14-kv-layout.svg](svg/14-kv-layout.svg)（连续 vs 分页布局）+ [svg/15-continuous-batching.svg](svg/15-continuous-batching.svg)（static vs continuous batching）+ [svg/16-radix-tree-prefix.svg](svg/16-radix-tree-prefix.svg)（RadixAttention 前缀树）。

- [x] **KV Cache 布局**：layer-major vs token-major、`[B, H, S, D]` vs paged block（05 §5.2，主图 `svg/14-kv-layout.svg`）
- [x] **Continuous Batching**：Orca 论文 → vLLM 的 scheduler（05 §5.3，主图 `svg/15-continuous-batching.svg`）
- [x] **Chunked Prefill**：长 prompt 的分块、prefill/decode 混跑（05 §5.4）
- [x] **Prefix Cache / RadixAttention**：SGLang 的前缀树共享（05 §5.5，主图 `svg/16-radix-tree-prefix.svg`）
- [x] **PD 分离（Disaggregated Prefill/Decode）**：vLLM v1、Mooncake、DistServe（05 §5.6，调度侧；KV 传输通信侧见 03 §3.5）
- [x] **KV 量化**：FP8/INT8/INT4 KV、per-channel vs per-token scale（05 §5.7；量化算法 GPTQ/AWQ 等见阶段 8）
- [x] **KV offload**：CPU / NVMe / 远端节点；LMCache（05 §5.8）

---

## 阶段 6｜推理引擎深读 ✓

> 已就位：[chapters/06-inference-engines.md](chapters/06-inference-engines.md)，主图 [svg/17-inference-engine-arch.svg](svg/17-inference-engine-arch.svg)（推理引擎通用骨架 loop + 六条差异轴）。选 vLLM/SGLang 深读源码，其它对照阅读。

- [x] **vLLM**：（06 §6.3）
  - [x] 架构：`LLMEngine` / `Scheduler` / `BlockManager` / `ModelRunner` / `Worker`（06 §6.3.1）
  - [x] v1 重构：单进程 + executor、TP/PP/EP 配置（06 §6.3.2）
  - [x] 自定义模型接入：`registry`、`linear`、`column_parallel_linear`（06 §6.3.3）
  - [x] Speculative decoding、structured output（xgrammar/outlines）（06 §6.3.4，算法细节见阶段 8/10）
- [x] **SGLang**：（06 §6.4）
  - [x] `Runtime` / `TokenizerManager` / `Scheduler` / `TpModelWorker`（06 §6.4.1）
  - [x] RadixAttention 与 prefix sharing 的实现（06 §6.4.2，复用 `svg/16`）
  - [x] DeepSeek-V3 全套优化（DP attention、EP、MTP）（06 §6.4.3）
- [x] **TensorRT-LLM**：plugin、in-flight batching、FP8 recipe（06 §6.5）
- [x] **LMDeploy / Turbomind**：与 vLLM 在 kernel 上的差异（06 §6.6）
- [x] **llama.cpp / MLX / Ollama**：本地推理路径，量化生态（06 §6.7）
- [x] **对比矩阵**：吞吐 / 首 token 延迟 / 长 context / MoE 支持 / PD 分离支持 / 多模态（06 §6.8）

---

## 阶段 7｜训练框架深读 ✓

> 已就位：[chapters/07-training-frameworks.md](chapters/07-training-frameworks.md)，主图 [svg/18-training-framework-arch.svg](svg/18-training-framework-arch.svg)（training step loop + DeviceMesh 多维并行 + 八条差异轴）。深读 Megatron-Core/DeepSpeed，PyTorch 原生底座先讲，其它对照阅读。

- [x] **PyTorch 分布式**：`torch.distributed`、`DTensor`、`DeviceMesh`、`pipeline_parallel`（07 §7.3，主图 `svg/18-training-framework-arch.svg`）
- [x] **FSDP2 (`fully_shard`)**：与 FSDP1 的 API/性能差异（已由 02 §2.2.8.2 完整覆盖；本阶段聚焦 PyTorch distributed 整体、DTensor、与 TP/PP/CP 的组合而不再单独讲 FSDP2 自身）
- [x] **Megatron-LM / Megatron-Core**：TP+PP+DP+EP+CP 的官方实现（07 §7.4）
- [x] **DeepSpeed**：ZeRO-3、ZeRO-Infinity、MoE、Ulysses（07 §7.5）
- [x] **TorchTitan**：原生 PyTorch 写的训练范例（07 §7.6）
- [x] **Colossal-AI / Pax (JAX) / MaxText**：作为对照阅读（07 §7.7）
- [x] **Checkpoint 格式**：`safetensors`、分布式 ckpt、`torch.distributed.checkpoint`（07 §7.8）

---

## 阶段 8｜量化、蒸馏与加速 ✓

> 已就位：[chapters/08-quantization.md](chapters/08-quantization.md)，主图 [svg/19-quantization-landscape.svg](svg/19-quantization-landscape.svg)（三类加速手段 + scale 粒度谱 + outlier）+ [svg/20-speculative-decoding.svg](svg/20-speculative-decoding.svg)（投机解码时序）。三类正交手段：量化(§8.3–8.6) / 投机解码(§8.7) / 稀疏化(§8.8)。

- [x] **PTQ**：GPTQ、AWQ、SmoothQuant、HQQ（08 §8.3）
- [x] **FP8 训练 / 推理**：per-tensor / per-token / per-block / Hopper FP8 recipe（08 §8.4；硬件见 0 §0.2.4，GEMM kernel 见 4 §4.7.4）
- [x] **W4A16 / W8A8 / W4A8**：vLLM `quantization` 后端、Marlin、Machete kernel（08 §8.5）
- [x] **KV 量化**：KIVI、FlashInfer KV INT8（08 §8.6 算法侧；显存视角见 5 §5.7）
- [x] **Speculative Decoding**：draft model、Medusa、EAGLE、Lookahead、Self-Speculative（08 §8.7，主图 `svg/20-speculative-decoding.svg`；引擎位置见 6 §6.3.4）
- [x] **稀疏化**：MoE 路由稀疏、Activation Sparsity、稀疏 Attention（NSA、MoBA）（08 §8.8；MoE 工程见 2 §2.2.5/3 §3.4，稀疏 attn 长序列见阶段 9）

---

## 阶段 9｜长上下文与 MoE 专题 ✓

> 已就位（类型 C 案例章）：[chapters/09-long-context-moe.md](chapters/09-long-context-moe.md)，主图 [svg/21-deepseek-v3-fullstack.svg](svg/21-deepseek-v3-fullstack.svg)（DeepSeek-V3 全栈技术地图）+ 复用 [svg/05](svg/05-cp-ring-attention.svg)/[svg/06](svg/06-deepseek-v3-topology.svg)。以 DeepSeek-V3 为核心案例串起长上下文 + 大 MoE 两条线。

- [x] **位置外推**：YaRN、LongRoPE、Position Interpolation（09 §9.3；RoPE 基础见 1 §1.2.3）
- [x] **长上下文 attention**：Ring / Striped / Ulysses / DistFlashAttn（09 §9.4，复用 `svg/05`；CP 基础见 2 §2.2.6）
- [x] **DeepSeek 体系**：MLA + DeepSeekMoE + DualPipe + Multi-Token Prediction（09 §9.5–9.7，全栈地图 `svg/21-deepseek-v3-fullstack.svg` + 部署拓扑 `svg/06-deepseek-v3-topology.svg`）
- [x] **MoE 推理**：expert offload、Hot/Cold expert、推理时的 EP+DP+TP 组合（09 §9.8，复用 `svg/06`）
- [x] **Mixture-of-Depths / 早退**：选读（09 §9.11，作为"深度维度稀疏"补充）

---

## 阶段 10｜生产化服务与多模态 ✓

> 已就位（A + D 混合类型）：[chapters/10-serving-multimodal.md](chapters/10-serving-multimodal.md)，主图 [svg/22-multimodal-token-path.svg](svg/22-multimodal-token-path.svg)（图像→token 路径）。前半 A 类（API/结构化输出/多 LoRA/多模态 §10.2–10.5），后半 D 类 cookbook（网关/可观测性 §10.6–10.7）。

- [x] **OpenAI 兼容 API**：`/v1/chat/completions`、`/v1/responses`、tool use、JSON schema（10 §10.2）
- [x] **结构化输出**：xgrammar、outlines、lm-format-enforcer（10 §10.3；logits mask 原理见 1 §1.2.5，引擎位置见 6 §6.3.4）
- [x] **多 LoRA serving**：vLLM `--enable-lora`、SGLang `--lora-paths`（10 §10.4）
- [x] **多模态推理**：Qwen-VL、LLaVA、InternVL 的图像/视频 token 路径（10 §10.5，主图 `svg/22-multimodal-token-path.svg`）
- [x] **路由与网关**：Envoy AI Gateway、LiteLLM、vLLM Production Stack（10 §10.6，D 类 cookbook）
- [x] **可观测性**：metrics、tracing、SLO（TTFT、TPOT、E2E）（10 §10.7，D 类 cookbook；指标定义见 5 §5.1）

---

## 阶段 11｜性能分析与调优工具（持续）✓

> 已就位（纯 D 类 cookbook）：[chapters/11-profiling.md](chapters/11-profiling.md)，全章"任务→命令→输出→判读→下一步"。把前面各阶段的调优散点收拢成系统方法论：nvidia-smi → Roofline → profiler/nsys → ncu/NCCL trace → 阶段旋钮的五步下钻。

- [x] **Profiler**：`torch.profiler`、Nsight Systems (`nsys`)、Nsight Compute (`ncu`)（11 §11.3/11.4/11.5）
- [x] **通信 trace**：`NCCL_DEBUG=TRACE`、`TORCH_NCCL_TRACE_BUFFER_SIZE`（11 §11.6；NCCL 调优旋钮见 3 §3.3）
- [x] **Roofline 分析**：算子是 compute-bound 还是 memory-bound？（11 §11.2；Roofline 基础见 0 §0.2.2）
- [x] **常见瓶颈定位**：HBM 带宽 / NVLink 带宽 / PCIe / IB 拥塞 / kernel launch overhead（11 §11.8，端到端排障实战 + 瓶颈速查表）
- [x] **CUDA Graph**：减少 launch overhead，注意与变长 batch 的兼容（11 §11.7；基础见 0 §0.3，引擎集成见 4 §4.4.3/6 §6.3.2）

---

## 阶段 12｜代表模型架构（选读，对照源码）✓

> 已就位（类型 B 横向对比章）：[chapters/12-model-architectures.md](chapters/12-model-architectures.md)，主图 [svg/23-llm-architecture-knobs.svg](svg/23-llm-architecture-knobs.svg)（标准骨架 + 六个旋钮）。§12.1 立"六旋钮 + config 速读"框架，§12.2–12.7 每个模型按这个框架填取值，§12.8 横向对比矩阵 + §12.9 排错心法收尾。**全书最后一个内容章节，至此 12 个阶段全部就位。**

- [x] **LLaMA 1/2/3/4**：GQA、Tokenizer、MoE（L4 Scout/Maverick）（12 §12.2，主图 `svg/23-llm-architecture-knobs.svg`）
- [x] **Qwen 2 / 2.5 / 3**：tied embedding、长上下文、MoE（Qwen3-MoE）（12 §12.3）
- [x] **Mixtral / Mistral**：sliding window、稀疏 MoE（12 §12.4）
- [x] **DeepSeek V2 / V3 / R1**：MLA、DeepSeekMoE、MTP、推理优化（12 §12.5；架构原理详见阶段 9）
- [x] **GLM / Yi / Gemma / Phi**：架构差异点（12 §12.6）
- [x] **多模态**：Qwen-VL、InternVL、LLaVA-NeXT、视频模型的 token 化（12 §12.7；token 路径机制见 10 §10.5）

---

## Capstone｜动手项目（贯穿全程，建议至少完成 3 个）

- [ ] **P1**：从 0 写一个 ~500 行的 mini-vLLM（KV 分页 + continuous batching + greedy）
- [ ] **P2**：在 2×8 H100 上跑通 Llama-3-70B 的 TP=8、PP=2 推理，画通信时序图
- [ ] **P3**：把一个 7B 模型用 GPTQ-4bit + Marlin kernel 部署到 vLLM，对比 FP16 的吞吐/精度
- [ ] **P4**：DeepSeek-V3 上跑 EP + DP attention，画一张 dispatch/combine 通信图
- [ ] **P5**：实现 PD 分离 demo：1×prefill 节点 + 2×decode 节点，KV 走 RDMA
- [ ] **P6**：写一个 FlashAttention 风格的 Triton kernel 并和官方版本做 perf 对比
- [ ] **P7**：复现一篇推理优化论文（如 Medusa / EAGLE / Chunked Prefill）

---

## 推荐阅读清单（持续追加）

- [ ] Megatron-LM、ZeRO、FSDP、GPipe、PipeDream、1F1B、Zero-Bubble 原论文
- [ ] FlashAttention v1/v2/v3、PagedAttention（vLLM）、RadixAttention（SGLang）
- [ ] DeepSpeed-MoE、Switch Transformer、GShard、DeepSeek-V3 技术报告
- [ ] Orca（continuous batching）、SARATHI（chunked prefill）、DistServe（PD 分离）
- [ ] Ring Attention、Striped Attention、Ulysses
- [ ] NVIDIA《GPU Performance Background》《CUDA C++ Best Practices》
- [ ] 官方源码导读：vLLM `docs/source`、SGLang `docs/backend`、Megatron `docs/`
