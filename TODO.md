# 大模型框架系统学习 TODO

> 目标：从 Transformer 基础结构开始，逐层向上打通 **并行策略 → 集合通信 → 分布式框架 → 推理引擎 → 算子内核 → 量化加速 → 生产化部署** 的完整知识链。

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

> 已就位（教材写作部分）：[chapters/02-parallelism.md](chapters/02-parallelism.md) 2.0–2.4 节覆盖 DP/TP/SP/PP/EP/CP 六种并行与多维编排；2.2.8 节深入 ZeRO 1/2/3、FSDP1/FSDP2、`HYBRID_SHARD`。剩余 "3D parallelism 实战" 是硬件 hands-on 任务（需 2 节点 H100），归入待办，**不阻塞章节定稿**。

- [x] **DP（数据并行）**：`DDP` 梯度桶、`gradient_as_bucket_view`；与 ZeRO 的关系（02 §2.2.1）
- [x] **ZeRO 1/2/3 与 FSDP**：参数/梯度/优化器状态分片；`FULL_SHARD` vs `HYBRID_SHARD`；`torch.distributed.fsdp` 与 FSDP2 (`fully_shard`) 的差异（02 §2.2.8）
- [x] **TP（张量并行）**：Megatron 列并行 + 行并行的配对；`g`/`f` 算子的 AllReduce 位置（02 §2.2.2，参见 `svg/02-tp-forward.svg`）
- [x] **SP（序列并行）**：与 TP 配合，省 LN/Dropout 的激活显存（02 §2.2.3）
- [x] **PP（流水并行）**：GPipe、1F1B、Interleaved 1F1B、Zero Bubble、DualPipe（DeepSeek）（02 §2.2.4，参见 `svg/03-pp-1f1b-schedule.svg`）
- [x] **EP（专家并行）**：All-to-All 通信模式、Dispatch/Combine、Token Drop（02 §2.2.5，参见 `svg/04-ep-moe-all2all.svg`）
- [x] **CP（上下文/序列并行）**：Ring Attention、Striped Attention、Ulysses（02 §2.2.6，参见 `svg/05-cp-ring-attention.svg`）
- [x] **多维并行编排**：TP×PP×DP×EP×CP 组合，rank 拓扑映射（02 §2.4，参见 `svg/07-multi-dim-parallel-topology.svg`）
- [ ] **3D parallelism 实战**：在 2 节点 ×8 GPU 上跑通 Megatron-LM 训练 LLaMA-7B（hands-on，待硬件就绪；与 capstone P2 推理实战互补）

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

## 阶段 5｜KV Cache、调度器与显存管理

- [ ] **KV Cache 布局**：layer-major vs token-major、`[B, H, S, D]` vs paged block
- [ ] **Continuous Batching**：Orca 论文 → vLLM 的 scheduler
- [ ] **Chunked Prefill**：长 prompt 的分块、prefill/decode 混跑
- [ ] **Prefix Cache / RadixAttention**：SGLang 的前缀树共享
- [ ] **PD 分离（Disaggregated Prefill/Decode）**：vLLM v1、Mooncake、DistServe
- [ ] **KV 量化**：FP8/INT8/INT4 KV、per-channel vs per-token scale
- [ ] **KV offload**：CPU / NVMe / 远端节点；LMCache

---

## 阶段 6｜推理引擎深读

> 选两家深读源码，其它对照阅读。

- [ ] **vLLM**：
  - [ ] 架构：`LLMEngine` / `Scheduler` / `BlockManager` / `ModelRunner` / `Worker`
  - [ ] v1 重构：单进程 + executor、TP/PP/EP 配置
  - [ ] 自定义模型接入：`registry`、`linear`、`column_parallel_linear`
  - [ ] Speculative decoding、structured output（xgrammar/outlines）
- [ ] **SGLang**：
  - [ ] `Runtime` / `TokenizerManager` / `Scheduler` / `TpModelWorker`
  - [ ] RadixAttention 与 prefix sharing 的实现
  - [ ] DeepSeek-V3 全套优化（DP attention、EP、MTP）
- [ ] **TensorRT-LLM**：plugin、in-flight batching、FP8 recipe
- [ ] **LMDeploy / Turbomind**：与 vLLM 在 kernel 上的差异
- [ ] **llama.cpp / MLX / Ollama**：本地推理路径，量化生态
- [ ] **对比矩阵**：吞吐 / 首 token 延迟 / 长 context / MoE 支持 / PD 分离支持 / 多模态

---

## 阶段 7｜训练框架深读

- [ ] **PyTorch 分布式**：`torch.distributed`、`DTensor`、`DeviceMesh`、`pipeline_parallel`
- [ ] **FSDP2 (`fully_shard`)**：与 FSDP1 的 API/性能差异
- [ ] **Megatron-LM / Megatron-Core**：TP+PP+DP+EP+CP 的官方实现
- [ ] **DeepSpeed**：ZeRO-3、ZeRO-Infinity、MoE、Ulysses
- [ ] **TorchTitan**：原生 PyTorch 写的训练范例
- [ ] **Colossal-AI / Pax (JAX) / MaxText**：作为对照阅读
- [ ] **Checkpoint 格式**：`safetensors`、分布式 ckpt、`torch.distributed.checkpoint`

---

## 阶段 8｜量化、蒸馏与加速

- [ ] **PTQ**：GPTQ、AWQ、SmoothQuant、HQQ
- [ ] **FP8 训练 / 推理**：per-tensor / per-token / per-block / Hopper FP8 recipe
- [ ] **W4A16 / W8A8 / W4A8**：vLLM `quantization` 后端、Marlin、Machete kernel
- [ ] **KV 量化**：KIVI、FlashInfer KV INT8
- [ ] **Speculative Decoding**：draft model、Medusa、EAGLE、Lookahead、Self-Speculative
- [ ] **稀疏化**：MoE 路由稀疏、Activation Sparsity、稀疏 Attention（NSA、MoBA）

---

## 阶段 9｜长上下文与 MoE 专题

- [ ] **位置外推**：YaRN、LongRoPE、Position Interpolation
- [ ] **长上下文 attention**：Ring / Striped / Ulysses / DistFlashAttn
- [ ] **DeepSeek 体系**：MLA + DeepSeekMoE + DualPipe + Multi-Token Prediction（参见 `svg/06-deepseek-v3-topology.svg`）
- [ ] **MoE 推理**：expert offload、Hot/Cold expert、推理时的 EP+DP+TP 组合
- [ ] **Mixture-of-Depths / 早退**：选读

---

## 阶段 10｜生产化服务与多模态

- [ ] **OpenAI 兼容 API**：`/v1/chat/completions`、`/v1/responses`、tool use、JSON schema
- [ ] **结构化输出**：xgrammar、outlines、lm-format-enforcer
- [ ] **多 LoRA serving**：vLLM `--enable-lora`、SGLang `--lora-paths`
- [ ] **多模态推理**：Qwen-VL、LLaVA、InternVL 的图像/视频 token 路径
- [ ] **路由与网关**：Envoy AI Gateway、LiteLLM、vLLM Production Stack
- [ ] **可观测性**：metrics、tracing、SLO（TTFT、TPOT、E2E）

---

## 阶段 11｜性能分析与调优工具（持续）

- [ ] **Profiler**：`torch.profiler`、Nsight Systems (`nsys`)、Nsight Compute (`ncu`)
- [ ] **通信 trace**：`NCCL_DEBUG=TRACE`、`TORCH_NCCL_TRACE_BUFFER_SIZE`
- [ ] **Roofline 分析**：算子是 compute-bound 还是 memory-bound？
- [ ] **常见瓶颈定位**：HBM 带宽 / NVLink 带宽 / PCIe / IB 拥塞 / kernel launch overhead
- [ ] **CUDA Graph**：减少 launch overhead，注意与变长 batch 的兼容

---

## 阶段 12｜代表模型架构（选读，对照源码）

- [ ] **LLaMA 1/2/3/4**：GQA、Tokenizer、MoE（L4 Scout/Maverick）
- [ ] **Qwen 2 / 2.5 / 3**：tied embedding、长上下文、MoE（Qwen3-MoE）
- [ ] **Mixtral / Mistral**：sliding window、稀疏 MoE
- [ ] **DeepSeek V2 / V3 / R1**：MLA、DeepSeekMoE、MTP、推理优化
- [ ] **GLM / Yi / Gemma / Phi**：架构差异点
- [ ] **多模态**：Qwen-VL、InternVL、LLaVA-NeXT、视频模型的 token 化

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
