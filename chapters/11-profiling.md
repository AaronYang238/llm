# 阶段 11｜性能分析与调优工具 ✓

> 一句话定位：把前面所有阶段散落的"怎么测、怎么判、怎么调"收拢成一套系统的性能分析方法论。本章是纯 cookbook（D 类）——按"任务 → 命令 → 输出片段 → 怎么判读 → 下一步"组织，拿来当工具书查。遇到一个性能问题，照着对应任务的配方走，从现象定位到根因。

## 目录

- [11.0 为什么需要这一层](#110-为什么需要这一层)
- [11.1 工具速览](#111-工具速览)
- [11.2 任务：判定 compute-bound 还是 memory-bound（Roofline）](#112-任务判定-compute-bound-还是-memory-boundroofline)
- [11.3 任务：用 torch.profiler 定位热点](#113-任务用-torchprofiler-定位热点)
- [11.4 任务：用 nsys 看时间线](#114-任务用-nsys-看时间线)
- [11.5 任务：用 ncu 深挖单个 kernel](#115-任务用-ncu-深挖单个-kernel)
- [11.6 任务：定位通信瓶颈（NCCL trace）](#116-任务定位通信瓶颈nccl-trace)
- [11.7 任务：判断该不该上 CUDA Graph](#117-任务判断该不该上-cuda-graph)
- [11.8 端到端排障实战](#118-端到端排障实战)
- [11.9 常见坑与 FAQ](#119-常见坑与-faq)
- [11.10 延伸阅读](#1110-延伸阅读)

---

## 11.0 为什么需要这一层

前面十个阶段，每章都讲了"这里可能慢、那里可以调"——阶段 0 的 Roofline、阶段 3 的 NCCL 调优、阶段 5 的调度指标、阶段 10 的可观测性。但这些是**散点**。真实排障时你面对的是一个黑盒现象——"吞吐只有预期一半""TTFT 突然翻倍""8 卡只跑出 3 卡的速度"——**你得有一套系统的方法，从现象一步步定位到根因**。这就是本章。

性能分析的第一原则：**先测量，再优化（measure, don't guess）**。新手最常见的错误是凭直觉改代码——"我觉得是 attention 慢"，改了半天发现瓶颈在数据加载。**任何优化前，先用 profiler 看清楚时间花在哪。** 本章的全部工具，都是为了把"我觉得"变成"我看到"。

第二原则：**分层定位**。一个 LLM 性能问题可能出在任何一层：

```
应用层   →  调度 / batch 配置不对（阶段 5）
框架层   →  Python 开销、kernel launch overhead（阶段 0/4）
kernel 层 →  某个 kernel 写得差、没用 Tensor Core（阶段 4）
通信层   →  NCCL 走错算法、跨节点带宽不足（阶段 3）
硬件层   →  HBM 带宽、NVLink、PCIe、掉卡（阶段 0）
```

不同层用不同工具：torch.profiler 看框架/kernel、nsys 看时间线和通信、ncu 深挖单 kernel、NCCL trace 看通信、nvidia-smi 看硬件。**本章教的就是"哪层用哪个工具、怎么读它的输出"**。

第三原则：**Roofline 是总纲**。回阶段 0 §0.2.2——任何算子要么 compute-bound、要么 memory-bound、要么 overhead-bound。**先判定算子在哪一类，才知道往哪个方向优化**：

- compute-bound → 提 Tensor Core 利用率、换更快 kernel；
- memory-bound → 减少访存、融合算子、加 batch；
- overhead-bound → 减少 kernel launch（CUDA Graph）、减 Python 开销。

往错误的方向优化是浪费——给 memory-bound 算子换更快的计算 kernel，没用。

读完之后你应当能：

1. 面对一个性能问题，知道**第一步该跑哪个工具**；
2. 读懂 torch.profiler / nsys / ncu 的关键输出，定位热点；
3. 判定一个算子是 compute / memory / overhead 哪类 bound；
4. 定位多卡通信瓶颈（NCCL 走没走对算法、带宽够不够）；
5. 判断该不该上 CUDA Graph，以及为什么。

---

## 11.1 工具速览

性能分析工具按"作用层次"和"粒度"分。先建立一张"什么问题用什么工具"的速查表（D 类章节的导航）：

| 工具 | 看什么层 | 粒度 | 典型命令 |
|---|---|---|---|
| **`nvidia-smi`** | 硬件 | 实时快照 | `nvidia-smi`、`nvidia-smi dmon` |
| **`nvidia-smi topo -m`** | 硬件拓扑 | 静态 | 看 NVLink/PCIe 连接（阶段 0 §0.5） |
| **`torch.profiler`** | 框架 + kernel | 算子级 | Python API，导出 trace |
| **Nsight Systems (`nsys`)** | 系统时间线 | kernel + 通信 + CPU | `nsys profile python x.py` |
| **Nsight Compute (`ncu`)** | 单个 kernel | 指令/访存级 | `ncu --set full -k kernel_name` |
| **NCCL trace** | 通信 | collective 级 | `NCCL_DEBUG=TRACE`（阶段 3 §3.3.1） |
| **`nccl-tests`** | 通信微基准 | busbw 曲线 | `all_reduce_perf`（阶段 3 §3.10） |
| **Prometheus metrics** | 服务 | 聚合指标 | 引擎 `/metrics`（阶段 10 §10.7） |

选工具的决策逻辑：

```
性能问题
├─ 先看硬件正常吗？        → nvidia-smi（利用率/显存/掉卡/降频）
├─ 时间花在哪个大阶段？     → torch.profiler 或 nsys（找热点）
│   ├─ 某个 kernel 慢？     → ncu 深挖（为什么慢）
│   └─ 多卡通信慢？         → nsys timeline + NCCL trace
├─ 算子是哪类 bound？       → Roofline 判定（§11.2）
└─ 服务级延迟？            → Prometheus metrics（阶段 10 §10.7）
```

三个工具的分工要记牢（最常用）：

- **torch.profiler**：**Python 友好、看 PyTorch 算子**。第一个上，看"时间花在哪些算子"。门槛低，集成进训练/推理脚本即可。
- **nsys（Nsight Systems）**：**系统级时间线**。看 kernel、通信、CPU 的**时序关系**——尤其"GPU 在不在等 CPU""通信和计算有没有重叠"。定位 overhead-bound 和通信问题的主力。
- **ncu（Nsight Compute）**：**单 kernel 显微镜**。锁定某个 kernel 后，看它的访存效率、Tensor Core 利用率、occupancy——回答"这个 kernel 为什么慢"。最细，也最慢（会重放 kernel）。

由粗到细：**torch.profiler 找热点 → nsys 看时序 → ncu 挖单点**。不要一上来就 ncu（太细，淹没在细节里），先用粗工具缩小范围。

> 本章心智模型：**性能分析是"由现象到根因的下钻过程"——用对工具、读对输出、往对方向。** nvidia-smi 看硬件正不正常、profiler/nsys 找时间花哪了、ncu 挖单 kernel、NCCL trace 看通信、Roofline 定方向。前面十章讲"机制是什么"，本章讲"机制出问题时怎么查"。每个任务（§11.2–11.7）都是一个可照做的配方。

---

## 11.2 任务：判定 compute-bound 还是 memory-bound（Roofline）

**场景**：你有一个慢的算子/阶段，想知道往哪个方向优化——是该提算力（compute-bound）、减访存（memory-bound），还是减 launch（overhead-bound）。**这是所有优化的第一步**（回 §11.0 第三原则、阶段 0 §0.2.2）。判错方向，后面全白做。

### 11.2.1 三类 bound 的判定标准

回阶段 0 §0.2.2 的 Roofline：算子性能上限 = `min(峰值算力, 算术强度 × 带宽)`。算术强度 = FLOPs / 从 HBM 搬的字节数。三类：

| 类型 | 特征 | LLM 里的典型 | 优化方向 |
|---|---|---|---|
| **compute-bound** | 算术强度 > ridge point，算力打满 | prefill 大 GEMM、大 batch decode | 提 Tensor Core 利用率、FP8、更快 kernel |
| **memory-bound** | 算术强度 < ridge point，带宽打满 | 小 batch decode、RMSNorm、KV 读 | 融合算子、加 batch、量化减字节 |
| **overhead-bound** | GPU 大量空闲、卡在 CPU/launch | 小 batch、kernel 碎、动态 shape | CUDA Graph、减 Python 开销 |

H100 BF16 的 ridge point ≈ 295 FLOP/Byte（阶段 0 §0.5）。但实战中**不用手算算术强度**——用工具直接看症状更快。

### 11.2.2 命令：先看 GPU 利用率定大方向

最快的第一刀——`nvidia-smi dmon` 实时看利用率：

```bash
nvidia-smi dmon -s um -d 1     # u=利用率 m=显存，每秒刷新
```

输出（节选）：

```
# gpu   sm   mem    mclk   pclk
# Idx    %     %     MHz    MHz
    0    95    78    2619   1980      <- SM 95%：算力忙
    0    34    92    2619   1980      <- SM 34% 但 mem 高：memory-bound 嫌疑
    0     8     3    2619   1980      <- 都很低：overhead-bound 嫌疑（GPU 在等）
```

**怎么判读**：

- **`sm` 高（>80%）** → 算力忙 → 大概率 **compute-bound**；
- **`sm` 中等、`mem` 控制器忙** → 在搬数据 → **memory-bound** 嫌疑；
- **`sm` 和 `mem` 都低** → GPU 闲着等 → **overhead-bound**（卡在 CPU / launch / 同步）。

这只是粗判，`sm%` 不直接等于 Tensor Core 利用率（CUDA core 跑也算 sm 忙）。要精确，下一步上 ncu（§11.5）。

### 11.2.3 命令：torch.profiler 看 GPU 空闲比例

overhead-bound 最容易被忽略——GPU 利用率低但你以为是算得慢。torch.profiler 直接给"GPU 有多少时间在干活 vs 空闲"：

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model.generate(...)        # 跑一段真实负载
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**怎么判读**：

- 看 **CUDA 总时间 / 墙钟时间**——比值低（如 < 60%）说明 GPU 大量空闲 → **overhead-bound**；
- 看 top 算子——如果是一堆小 kernel（norm、element-wise）占大头而非 GEMM → 融合机会（memory-bound）；
- 如果 GEMM/attention 占绝大多数且 GPU 忙 → compute-bound，优化方向是 kernel 本身。

### 11.2.4 判读决策表

把三个信号综合：

| `sm%` | GPU 忙/闲 | top 算子 | 结论 | 下一步 |
|---|---|---|---|---|
| 高 | 忙 | GEMM/attention | compute-bound | ncu 看 Tensor Core 利用率（§11.5）；考虑 FP8（§8.4） |
| 中 | 忙 | norm/element-wise 多 | memory-bound | 融合算子（§4.6）、加 batch（§5.3） |
| 低 | 闲 | kernel 碎、间隙大 | overhead-bound | nsys 看 launch 间隙（§11.4）→ CUDA Graph（§11.7） |

### 11.2.5 LLM 的经验先验

不用每次都测——LLM 的两个阶段有强先验（回阶段 1 §1.5）：

- **prefill**：序列长、GEMM 大 → **compute-bound**（除非 batch 极小）；
- **decode**：
  - **小 batch** → **memory-bound**（GEMV，读权重带宽是瓶颈）甚至 **overhead-bound**（kernel 碎）；
  - **大 batch** → 被 continuous batching 推成 **compute-bound**（§5.3.2）。

所以"decode 慢"的默认怀疑顺序：先查 batch 够不够大（memory→compute 的关键）、再查 launch overhead（CUDA Graph 开没开），最后才怀疑 kernel 本身。

> 判读纪律：**先 `nvidia-smi dmon` 定大方向（10 秒），再 torch.profiler 看空闲比例和热点（1 分钟），方向明确了再用 ncu 精确确认。** 由粗到细，别一上来精测。判定 bound 类型是优化的"指南针"——指对了方向，后面每章的优化手段才用得对地方。

---

## 11.3 任务：用 torch.profiler 定位热点

**场景**：方向定了（§11.2），现在要知道**时间具体花在哪些算子**。torch.profiler 是第一个该上的工具——Python 友好、集成简单、直接给算子级耗时排名。

### 11.3.1 命令：最小集成

```python
from torch.profiler import profile, ProfilerActivity, schedule

# 用 schedule 跳过 warmup，只测稳定阶段（避免首次 launch/编译污染数据）
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=3),   # 跳过 3 步，测 3 步
    on_trace_ready=lambda p: p.export_chrome_trace("trace.json"),
    record_shapes=True,                               # 记录张量 shape
    with_stack=True,                                  # 记录调用栈（定位到代码行）
) as prof:
    for _ in range(6):                                # 至少 wait+warmup+active 步
        model.generate(...)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

关键参数：`schedule` 跳过 warmup（首次有 launch/autotune 开销，污染数据）；`record_shapes` 让你看到算子的输入 shape（判断 GEMM 大小）；`with_stack` 定位到 Python 代码行。

### 11.3.2 输出：算子耗时排名

```
-----------------------------  ------------  ------------  ------------
                          Name    Self CUDA    CUDA total    # of Calls
-----------------------------  ------------  ------------  ------------
              ampere_gemm_...        45.2ms        45.2ms           320
       flash_attn_fwd_kernel        18.7ms        18.7ms            80
         elementwise_kernel         12.3ms        12.3ms          640   <- 小算子多
                 rms_norm...         8.1ms         8.1ms           160
            cudaLaunchKernel         0.0ms        31.5ms          1840   <- launch 开销大
-----------------------------  ------------  ------------  ------------
Self CPU time total: 52ms
Self CUDA time total: 89ms
```

**怎么判读**：

- **`Self CUDA`** = 算子自己在 GPU 上的时间，**按它排序找热点**；
- **`# of Calls`** 多 + 单次时间小（如 `elementwise_kernel` 640 次）→ **小算子碎、融合机会**（回 §4.6 Triton 融合）；
- **`cudaLaunchKernel` 的 CUDA total 大**（这里 31.5ms / 1840 次）→ **launch overhead 重** → overhead-bound 信号 → CUDA Graph（§11.7）；
- **CPU total 接近或超过 CUDA total** → CPU 是瓶颈，GPU 在等。

### 11.3.3 命令：导出 trace 看可视化时间线

table 是聚合的，看不到时序。导出的 `trace.json` 用 Chrome `chrome://tracing` 或 Perfetto（`ui.perfetto.dev`）打开，看**可视化时间线**：

```python
# 上面 on_trace_ready 已导出 trace.json
# 浏览器打开 chrome://tracing，Load trace.json
```

**怎么判读时间线**：

- **kernel 之间有大量空白条带** → GPU 在等 CPU launch → overhead-bound（§11.4 nsys 看得更清）；
- **CPU 行很忙、GPU 行有间隙** → CPU 喂不上数据 → 优化 Python 端、数据加载；
- **kernel 排得密、几乎无间隙** → GPU 喂饱了，瓶颈在 kernel 本身 → ncu 深挖（§11.5）。

### 11.3.4 任务：找融合机会

LLM 里常见的浪费是**小算子太多**——每个 RMSNorm、residual add、激活都是独立 kernel，各自往返 HBM（memory-bound）+ 各自 launch（overhead）。profiler 里的信号：

- 一堆 `elementwise` / `norm` / `add` 类 kernel，调用次数多、单次小、加起来占比可观。

**下一步**：把这些融合成一个 kernel（回阶段 4 §4.6 的 Triton fused RMSNorm + residual）。或者用 `torch.compile` 自动融合：

```python
model = torch.compile(model)   # inductor 自动把 element-wise 融合 + 生成 Triton kernel
```

profiler 再跑一遍对比——融合后小 kernel 数量应明显下降，HBM 流量减少。

### 11.3.5 适用边界

torch.profiler 的强项和短板：

| 强项 | 短板 |
|---|---|
| Python 集成最简单 | 看不到 NCCL 通信细节（用 nsys §11.4） |
| 算子级耗时 + 调用栈 | 看不到单 kernel 内部（用 ncu §11.5） |
| 能导出可视化 trace | 多卡场景信息有限 |

所以 torch.profiler 是**入口**——找到热点和大方向后，通信问题转 nsys、单 kernel 转 ncu。

> 判读纪律：**torch.profiler 回答"时间花在哪些算子" + "GPU 忙不忙"。** 按 `Self CUDA` 找热点、按 `# of Calls` 找碎 kernel、按 `cudaLaunchKernel` 占比找 overhead。它是最低成本的第一刀，但看不到通信和 kernel 内部——那是 §11.4 nsys 和 §11.5 ncu 的活。

---

## 11.4 任务：用 nsys 看时间线

**场景**：torch.profiler（§11.3）告诉你"有 overhead""通信可能慢"，但看不清**时序关系**——GPU 到底在等什么、通信和计算有没有重叠。nsys（Nsight Systems）是系统级时间线的主力，看的就是"谁在什么时候干什么"。

### 11.4.1 命令：抓一段 trace

```bash
nsys profile -o report \
  --trace=cuda,nvtx,osrt,nccl \      # 抓 CUDA kernel + NVTX 标注 + OS + NCCL 通信
  --gpu-metrics-device=all \         # 顺带抓 GPU 硬件指标
  python infer.py
```

关键：`--trace=...,nccl` 把 **NCCL 通信也抓进时间线**（这是 torch.profiler 看不到的，多卡排障的关键）。生成 `report.nsys-rep`，用 Nsight Systems GUI 打开。

只测一段（避免 trace 太大）：

```bash
# 在代码里圈定范围
import torch.cuda.nvtx as nvtx
nvtx.range_push("decode_steps"); model.generate(...); nvtx.range_pop()
# nsys 加 --capture-range=nvtx 只抓这段
```

### 11.4.2 怎么读时间线：四个观察点

nsys GUI 是多行时间线——CUDA kernel 一行、NCCL 一行、CPU 线程几行。重点看四个：

**① kernel 之间的间隙（gap）**

```
GPU: [gemm][gap][attn][gap][norm][gap]...     <- 每个 gap 是 GPU 空闲
```

- 间隙密、加起来占比大 → **overhead-bound**，GPU 在等 CPU launch 下一个 kernel；
- 对应到 CPU 行：会看到 CPU 在忙着 `cuLaunchKernel`。
- **下一步**：CUDA Graph 把这些 launch 录下来重放，消除间隙（§11.7）。

**② 通信和计算有没有重叠**

```
不好（串行）：  GPU: [GEMM........][AllReduce....][GEMM........]
好（重叠）：    GPU: [GEMM........][GEMM........]
               NCCL:      [AllReduce....]          <- 和计算并行
```

- AllReduce/All-to-All 占着时间线、计算停着等 → **通信没重叠**，是瓶颈；
- **下一步**：async-TP / 计算通信 overlap（阶段 4 §4.8）、或查 NCCL 配置（§11.6）。

**③ CPU 是不是瓶颈**

```
CPU: [busy.................]      <- CPU 一直忙
GPU: [k][gap....][k][gap....]     <- GPU 等 CPU
```

- GPU 频繁等 CPU → Python 开销、数据预处理慢 → 优化 CPU 端（多进程 dataloader、减 Python 逻辑）。

**④ 通信本身慢不慢**

- NCCL 行的某个 AllReduce 持续时间异常长 → 通信带宽问题 → §11.6 NCCL trace 深挖。

### 11.4.3 命令：不开 GUI 的快速统计

不想开 GUI，nsys 也能出文字统计：

```bash
nsys stats report.nsys-rep
```

输出（节选）：

```
** CUDA GPU Kernel Summary **
 Time(%)  Total Time  Instances           Name
   38.2%     45.2ms        320    ampere_gemm_...
   15.8%     18.7ms         80    flash_attn_fwd
   ...
** CUDA GPU MemOps / NCCL Summary **
   12.1%     14.3ms         80    ncclAllReduce       <- 通信占 12%
```

**怎么判读**：通信占比（这里 12% AllReduce）——如果通信占比很高（> 20%）且没和计算重叠，是优化重点。kernel summary 和 torch.profiler 的 table 类似，但 nsys 多了 NCCL 和内存操作。

### 11.4.4 典型判读案例

| 时间线现象 | 结论 | 下一步 |
|---|---|---|
| kernel 间隙密、CPU 忙 launch | overhead-bound | CUDA Graph（§11.7） |
| NCCL 占时间线、计算停等 | 通信没重叠 | async-TP（§4.8）/ NCCL 调优（§11.6） |
| CPU 行满、GPU 等 | CPU 瓶颈 | 多进程 dataloader、减 Python |
| 某 AllReduce 异常长 | 通信带宽问题 | NCCL trace（§11.6）、查拓扑（§0.5） |
| kernel 密排无间隙 | GPU 喂饱、瓶颈在 kernel | ncu 深挖（§11.5） |

> 判读纪律：**nsys 回答"时序关系"——GPU 在等谁、通信有没有藏进计算、CPU 是不是瓶颈。** 它是 torch.profiler（看算子耗时）和 ncu（看单 kernel）之间的关键一环，**也是多卡通信问题的主力工具**（唯一能在一张时间线上同时看到计算和 NCCL 的）。看到间隙想 overhead、看到通信不重叠想 overlap、看到 CPU 满想 Python 开销。

---

## 11.5 任务：用 ncu 深挖单个 kernel

**场景**：nsys（§11.4）告诉你"kernel 密排无间隙，瓶颈在 kernel 本身"——现在要回答**这个具体 kernel 为什么慢**。ncu（Nsight Compute）是单 kernel 显微镜，看到指令、访存、Tensor Core 利用率的细节。它最细，也最慢（会重放 kernel 多次采集指标），所以**只在锁定了具体 kernel 后才用**。

### 11.5.1 命令：锁定一个 kernel 测

ncu 默认会测每个 kernel（极慢）。务必**用 `-k` 过滤 + `-c` 限次数**：

```bash
ncu -k "regex:gemm|flash" \      # 只测名字含 gemm/flash 的 kernel
    -c 10 \                      # 每个最多测 10 次
    --set full \                 # 采集完整指标集
    -o ncu_report \              # 导出报告
    python infer.py
```

`--set full` 采集所有指标（慢但全）；快速看用 `--set basic`。生成 `ncu_report.ncu-rep`，用 Nsight Compute GUI 打开，或 `ncu --import` 看文字。

### 11.5.2 怎么读：先看三个顶层指标

ncu 报告信息量巨大，先看 **Speed of Light（SOL）** 区的三个数——它直接告诉你 kernel 卡在算力还是访存：

```
Section: GPU Speed Of Light Throughput
  Compute (SM) Throughput  [%]        38.5      <- 算力利用率
  Memory Throughput        [%]        91.2      <- 带宽利用率
  DRAM Throughput          [%]        91.2
```

**怎么判读**（这就是 Roofline 的精确版，§11.2）：

- **Compute 高、Memory 低** → **compute-bound**，算力打满 → 已接近最优，或换更高算力精度（FP8）；
- **Memory 高、Compute 低**（如上例 91% vs 38%）→ **memory-bound**，带宽打满 → 减少访存、融合、提复用；
- **两个都低** → kernel 没喂饱（occupancy 低 / launch 配置差）→ 看下一项。

### 11.5.3 关键诊断指标

按问题查对应指标：

**① Tensor Core 用没用上（compute-bound 时关键）**

```
  Tensor (TC) Throughput   [%]        72.3      <- Tensor Core 利用率
```

- LLM 的 GEMM/attention 应该跑在 Tensor Core 上。如果这个数很低（< 20%）但 Compute 高 → **算力浪费在 CUDA core**（没走 Tensor Core，或精度没对齐 MMA 要求，回阶段 4 §4.1）→ 检查 dtype、kernel 实现。

**② occupancy 够不够**

```
  Achieved Occupancy       [%]        45.0
  Theoretical Occupancy    [%]       100.0
```

- achieved 远低于 theoretical → SM 没填满 → 寄存器/SMEM 用太多限制了并发 block，或 grid 太小（batch 小）→ 调 block 大小 / launch 配置。

**③ 访存效率（memory-bound 时关键）**

```
  Memory Throughput        [%]        91.2
  L2 Hit Rate              [%]        45.0
  DRAM Throughput          [%]        91.2      <- 已接近 HBM 上限
```

- DRAM 已打满 91% → 真的 memory-bound，唯一出路是**减少从 HBM 搬的字节**（融合、量化、加 batch 提复用）；
- 如果 DRAM 没满但 Memory% 高 → 访存模式差（非合并访存、bank conflict，回阶段 0 §0.3）→ 改访存 pattern。

### 11.5.4 判读决策表

| SOL 现象 | TC% / occupancy | 结论 | 下一步 |
|---|---|---|---|
| Compute 高、Mem 低、TC 高 | TC > 70% | compute-bound，接近最优 | 考虑 FP8（§8.4）换更高算力 |
| Compute 高、TC 低 | TC < 20% | 算力没走 Tensor Core | 查 dtype / MMA 对齐（§4.1） |
| Mem 高、DRAM 满 | — | memory-bound | 融合 / 量化 / 加 batch |
| 两个都低、occupancy 低 | occ < 50% | 没喂饱 | 调 launch 配置 / 加 batch |
| Mem 高、DRAM 没满 | L2 命中低 | 访存模式差 | 改 coalesced access（§0.3） |

### 11.5.5 什么时候不用 ncu

ncu 是重武器，**大多数 LLM 性能问题不需要它**：

- 用现成 kernel（FlashAttention、cuBLAS、vLLM 的 kernel）→ 它们已经被 NVIDIA/社区调到接近最优，ncu 挖了也改不动；
- 只有**自己写 kernel**（Triton/CUDA，阶段 4 §4.6）时，ncu 才是必需的——它告诉你自己的 kernel 离硬件上限还差多少、卡在哪。

所以 ncu 主要服务两类人：**写自定义 kernel 的**、**怀疑现成 kernel 在自己 shape 上没跑好的**。日常调优（batch、调度、量化、CUDA Graph）用不到 ncu，用 nvidia-smi + torch.profiler + nsys 就够。

> 判读纪律：**ncu 回答"单个 kernel 为什么慢"——先看 SOL 三个数定 compute/memory bound（Roofline 精确版），再按方向查 Tensor Core 利用率 / occupancy / 访存效率。** 它是三件套里最细的，只在"锁定了 kernel 且能改它"时用。现成 kernel 别浪费时间 ncu——那是 kernel 作者的活。

---

## 11.6 任务：定位通信瓶颈（NCCL trace）

**场景**：多卡训练/推理慢，nsys（§11.4）显示通信占比高或不重叠。现在要深挖**通信到底卡在哪**——是走错算法、带宽不足、还是某张卡 hang 住。本节承接阶段 3 §3.3（NCCL 调优）、§3.7（nccl-tests），讲两类通信排障：**慢**和**卡死**。

### 11.6.1 先分清：慢 vs 卡死

两类完全不同的问题，先判断是哪一类：

| 现象 | 类型 | 主工具 |
|---|---|---|
| 能跑完但吞吐低 | **通信慢** | `NCCL_DEBUG=INFO` + nccl-tests 对照 |
| 训练突然卡住不动、超时报错 | **通信卡死（hang）** | `TORCH_NCCL_TRACE_BUFFER_SIZE` flight recorder |

### 11.6.2 任务：通信慢——先看走没走对算法

第一步，`NCCL_DEBUG=INFO` 看 NCCL 实际用了什么（回阶段 3 §3.3.1）：

```bash
NCCL_DEBUG=INFO python train.py 2>&1 | grep -E "via|Channel|NVLS|Connected"
```

**怎么判读**（回 §3.3.1 的字段表）：

- `via P2P/IPC` → 节点内走 NVLink，对；
- `via SHM` → 节点内走了 host 共享内存，**P2P 没启**，慢 → 查 ACS / `nvidia-peermem`（§3.3.4）；
- `via NET/IB/0/GDRDMA` → 跨节点走 IB + GDR，对；缺 `GDRDMA` → GDR 没启；
- algo 是 `Ring` 但多节点 → 可能该用 `Tree`/SHARP（§3.2.4）。

**下一步**：对照 nccl-tests 的理论 busbw（§3.7）——实测如果远低于 nccl-tests，是配置问题（按 §3.3.4 症状→旋钮表查）；如果接近 nccl-tests，是物理上限，得改并行策略减少通信。

### 11.6.3 任务：通信卡死——flight recorder

最难排的是 **hang**——训练卡住、几分钟后 NCCL watchdog 超时报错，但不知道哪张卡、哪个 collective 卡的。PyTorch 的 **NCCL flight recorder** 专治这个：

```bash
# 启动时开启（环境变量）
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000      # 记录最近 2000 个 collective
export TORCH_NCCL_DUMP_ON_TIMEOUT=1           # 超时自动 dump
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_trace   # dump 路径
```

hang 发生、watchdog 超时后，每个 rank 会 dump 自己最近的 collective 记录。分析 dump：

```python
# 加载各 rank 的 dump，找"卡在哪个 collective"
from torch.distributed.flight_recorder import ...   # 或用官方分析脚本
# 关键看：哪些 rank 完成了 collective N，哪些卡在 collective N
```

**怎么判读**：

- **所有 rank 都卡在同一个 collective** → 正常的集合通信在等某个慢 rank → 看那个 rank 是不是在做别的（数据加载慢、OOM 边缘）；
- **部分 rank 完成了、部分没到** → **mismatch**：有 rank 少调了一次 collective（常见于条件分支里 if 不对称、某 rank 提前 return），导致其它 rank 永远等不到 → 查代码里的 collective 调用是否所有 rank 对称;
- **某 rank 根本没记录** → 那张卡可能挂了 / 掉卡（`nvidia-smi` 确认）。

flight recorder 是多卡 hang 排障的**唯一系统手段**——否则只能盲猜。

### 11.6.4 任务：定位是哪条物理链路慢

确认是带宽问题后，定位是哪条链路（回阶段 0 §0.5、§11.1）：

```bash
nvidia-smi topo -m            # 看逻辑拓扑：哪些卡 NVLink、哪些跨 PCIe/SYS
nvidia-smi nvlink -s          # 看 NVLink 实时状态/带宽
```

**怎么判读**：

- `topo -m` 里两张常通信的卡显示 `SYS`（跨 socket）→ 拓扑映射错了，TP/EP 跨了 socket（回阶段 2 §2.4.3、阶段 7 §7.2.2）→ 改 rank 映射；
- IB 链路慢 → `ibstat` / `ib_write_bw` 测纯 IB（回阶段 3 §3.5.4）排除物理层。

### 11.6.5 通信排障决策表

| 现象 | 根因 | 下一步 |
|---|---|---|
| `via SHM` 而非 P2P | P2P 没启 | 关 ACS、装 `nvidia-peermem`（§3.3.4） |
| 缺 `GDRDMA` | GDR 没启 | `nvidia-peermem`、查 `dmesg`（§3.3.4） |
| busbw 远低于 nccl-tests | 配置问题 | §3.3.4 症状→旋钮表逐项查 |
| 多节点用 Ring | 没用 Tree/SHARP | `NCCL_ALGO`、`NCCL_COLLNET_ENABLE`（§3.3） |
| hang，所有 rank 卡同一 collective | 某 rank 慢 | 查那个 rank（dataloader/OOM 边缘） |
| hang，rank 完成数不一致 | collective mismatch | 查代码：所有 rank 对称调用 collective |
| `topo -m` 显示 SYS | 拓扑映射错 | 改 rank↔GPU 映射（§2.4.3） |

> 判读纪律：**通信排障先分"慢"还是"卡死"。** 慢 → `NCCL_DEBUG=INFO` 看走没走对路 + nccl-tests 对照理论值，按 §3.3.4 调旋钮；卡死 → `TORCH_NCCL_TRACE_BUFFER_SIZE` flight recorder 定位是哪个 collective、哪些 rank（尤其抓 collective mismatch 这类对称性 bug）。这是 nsys（§11.4 看到通信问题）之后的深挖，最终落到阶段 3 的具体旋钮。

---

## 11.7 任务：判断该不该上 CUDA Graph

**场景**：profiler（§11.3）/nsys（§11.4）显示 overhead-bound——kernel 间隙密、CPU 忙 launch、GPU 在等。CUDA Graph 是治这个的主药。但它不是万能、有约束。本节讲**怎么判断该不该上、上了怎么验证**。承接阶段 0 §0.3、阶段 4 §4.4。

### 11.7.1 先确认：真的是 overhead-bound 吗

CUDA Graph 只治 **launch overhead**——把一串 kernel launch 录下来一次性重放，消除每个 launch 的 CPU 开销（回阶段 0 §0.3）。所以先确认问题真的是它能治的：

判据（综合 §11.2–11.4）：

- `nvidia-smi dmon`：sm% 和 mem% 都低（GPU 闲）；
- torch.profiler：`cudaLaunchKernel` 占 CUDA total 比例高、小 kernel 多；
- nsys 时间线：kernel 间隙密、CPU 行忙着 launch。

**典型受益场景**：**小 batch decode**——70B 模型 decode 一个 token 要数百次 kernel launch，每次 ~5–10 μs，加起来 2–4 ms（回阶段 0 §0.3、阶段 4 §4.4）。这部分纯属 CPU 开销，CUDA Graph 能砍掉大半。

**不受益场景**：

- compute-bound（大 batch prefill）→ GPU 本来就忙，launch 开销占比小，CUDA Graph 收益微乎其微；
- kernel 本身慢 → 那是 kernel 问题（ncu §11.5），Graph 不解决。

### 11.7.2 CUDA Graph 的硬约束

CUDA Graph 录制的是**固定的 launch 序列 + 固定的内存地址**，所以有硬约束（回阶段 0 §0.6、阶段 4 §4.4.3）：

| 约束 | 含义 | LLM 里的影响 |
|---|---|---|
| **shape 必须固定** | 录制时的张量 shape，重放时必须一样 | 变长 batch 不能直接录 |
| **不能有 CPU 同步** | capture region 内不能 `.item()`、`.cpu()` 等 | 采样逻辑要小心 |
| **不能动态分配显存** | 不能 `cudaMalloc` | 内存要预分配 |
| **控制流固定** | 不能有数据依赖的 if 分支 | 投机解码等动态逻辑要特殊处理 |

这就是为什么推理引擎做了大量工作让 CUDA Graph 兼容变长 batch——**FlashInfer 的 static-shape 元数据**（阶段 4 §4.4.3）、**vLLM 的 piecewise CUDA graph**（按 batch size 分段录制，阶段 6 §6.3.2）。它们的本质都是**把"变长"转成"几个固定 shape 的集合"**，每个 shape 录一个 graph。

### 11.7.3 任务：在引擎里开启与验证

大多数人不手写 CUDA Graph，而是用引擎的开关。vLLM：

```bash
# vLLM 默认开启 CUDA Graph（非 eager 模式）
vllm serve model              # 默认 enforce_eager=False，即开 graph
vllm serve model --enforce-eager   # 关闭 graph（调试用）
```

**验证收益**：开/关各跑一次，对比：

```bash
# 关 graph 测 baseline
vllm bench ... --enforce-eager
# 开 graph 测
vllm bench ...
```

**怎么判读**：

- 小 batch decode 的 **TPOT 明显下降**（launch overhead 被消除）→ Graph 生效、有收益；
- 吞吐基本不变 → 本来就 compute-bound，Graph 没用上（正常，大 batch 不靠它）；
- 开了反而报错 / 变慢 → shape 太多导致 graph 重录、或显存不够存多个 graph → 调 graph 的 batch size 列表。

### 11.7.4 手写场景（自定义循环）

如果不用引擎、自己写推理循环，手写 CUDA Graph（回阶段 0 §0.3）：

```python
# 1) warmup（让 cudnn/cublas 选好算法）
for _ in range(3):
    out = model(static_input)
torch.cuda.synchronize()

# 2) capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_input)   # 录制，输入输出用固定 buffer

# 3) replay：填新数据到 static_input，replay
static_input.copy_(new_data)
g.replay()                             # 重放，零 launch 开销
result = static_out.clone()
```

关键：**输入输出是固定 buffer**——replay 前把新数据 `copy_` 进去，replay 后从固定输出 buffer 取。变长就为每个 shape 录一个 graph（piecewise）。

### 11.7.5 判读决策表

| profiler 现象 | 该上 Graph？ | 说明 |
|---|---|---|
| 小 batch decode、launch 占比高、GPU 闲 | **该上** | 典型受益场景，TPOT 大降 |
| 大 batch prefill、GPU 忙 | 不必 | compute-bound，收益小 |
| kernel 本身慢（ncu 确认） | 不该（治不了） | 用 ncu 优化 kernel |
| 变长 batch | 该上但要 piecewise | 引擎已处理（vLLM/FlashInfer） |
| 有动态控制流（投机解码） | 谨慎 | 需特殊处理，引擎有专门支持 |

> 判读纪律：**CUDA Graph 专治 launch overhead——先用 §11.2–11.4 确认真的 overhead-bound（GPU 闲、launch 占比高），再上。** 它的硬约束是"固定 shape + 无 CPU 同步 + 无动态分配"，所以变长场景靠引擎的 piecewise graph / static 元数据兼容。日常用引擎开关（vLLM 默认开），开关对比验证 TPOT 收益。这是 overhead-bound 的标准解药，但对 compute-bound 无用——别指望它治所有慢。

---

## 11.8 端到端排障实战

前面六个任务（§11.2–11.7）是单点工具。本节把它们串成**完整的排障流程**——给三个真实场景，演示从现象到根因的完整下钻路径。这也顺带把 TODO 关心的"常见瓶颈定位（HBM / NVLink / PCIe / IB / launch）"串了一遍。

### 11.8.1 标准排障流程

任何 LLM 性能问题，按这个顺序下钻（由粗到细、由外到内）：

```
① nvidia-smi          硬件正常吗？利用率/显存/掉卡/降频
        ↓
② Roofline 粗判        compute / memory / overhead 哪类？（§11.2）
        ↓
③ torch.profiler       时间花在哪些算子？GPU 忙不忙？（§11.3）
        ↓
④ nsys                 时序关系：等谁、通信重不重叠？（§11.4）
        ↓
⑤ 分叉：
   ├─ 单 kernel 慢  →  ncu（§11.5）
   ├─ 通信慢/卡死  →  NCCL trace（§11.6）
   └─ launch 多    →  CUDA Graph（§11.7）
        ↓
⑥ 落到对应阶段的旋钮    调度（5）/ kernel（4）/ 通信（3）/ 量化（8）
```

**核心纪律**：**每一步都是为了缩小范围**——别跳步（直接 ncu 会淹没在细节）、别凭猜（不 profile 就改）。

### 11.8.2 案例一：单卡 decode 吞吐只有预期一半

**现象**：单卡跑 7B，decode 吞吐远低于预期。

**下钻**：

1. `nvidia-smi dmon`：sm% 8%、mem% 3% —— **GPU 大量空闲** → overhead-bound 嫌疑（§11.2.2）；
2. torch.profiler：`cudaLaunchKernel` 占 CUDA total 的 40%、一堆小 kernel —— 确认 **overhead-bound**（§11.3.2）；
3. nsys：kernel 间隙密、CPU 忙 launch —— 实锤（§11.4.2 观察点①）；
4. **根因**：batch 太小（单请求）+ 没开 CUDA Graph，launch overhead 主导；
5. **下一步**：开 continuous batching 把 batch 拉大（§5.3）+ 开 CUDA Graph（§11.7）。

**教训**：decode 慢先查 batch 和 CUDA Graph，别一上来怀疑 kernel。

### 11.8.3 案例二：8 卡只跑出 3 卡的速度

**现象**：8 卡 TP 训练，吞吐远不到 8 卡线性扩展。

**下钻**：

1. `nvidia-smi dmon`：8 卡 sm% 都中等、不满 —— 不是单卡算力问题；
2. nsys（`--trace=nccl`）：时间线上 **AllReduce 占大块、计算停着等** —— 通信没重叠（§11.4.2 观察点②）；
3. `NCCL_DEBUG=INFO`：看到 `via SHM` 而非 `via P2P/IPC` —— **P2P 没启，走了 host 内存**（§11.6.2）；
4. `nvidia-smi topo -m`：确认卡间是 NVLink，但 P2P 被 ACS 挡了；
5. **根因**：ACS 没关，NCCL 退化到 SHM，节点内通信带宽腰斩；
6. **下一步**：关 ACS、装 `nvidia-peermem`（§3.3.4），重测 nccl-tests 确认 busbw 恢复（§3.7）。

**教训**：多卡不扩展先看通信走没走对路（NCCL\_DEBUG），对照 nccl-tests 理论值。

### 11.8.4 案例三：长 prompt 一来，TTFT 全线抖动

**现象**：平时 TTFT 正常，偶尔某些请求 TTFT p99 暴涨。

**下钻**：

1. Prometheus metrics（§10.7）：TTFT p99 尖刺，和长 prompt 请求时间吻合；
2. 引擎日志 / trace：尖刺时刻有一个超长 prompt 在 prefill —— **长 prefill 阻塞了后续请求**（§5.4.1）；
3. **根因**：没开 chunked prefill，长 prompt 的 prefill 独占 GPU，后面的请求 TTFT 被拖；
4. **下一步**：开 chunked prefill、调 `max_num_batched_tokens`（§5.4.3），让长 prefill 切块和 decode 混跑。

**教训**：TTFT 抖动（而非整体高）多半是长 prompt 干扰，看 chunked prefill。

### 11.8.5 常见瓶颈速查

把"哪个硬件链路是瓶颈"的判定收成一张表（回阶段 0 §0.2.2 的带宽层级）：

| 瓶颈 | 怎么发现 | 典型原因 | 下一步 |
|---|---|---|---|
| **HBM 带宽** | ncu DRAM throughput 满（§11.5.3） | memory-bound 算子 | 融合、量化、加 batch |
| **NVLink** | nsys 通信不重叠 + topo 是 NV | 节点内通信量大 | async-TP（§4.8）、减 TP 通信 |
| **PCIe** | topo 显示 PIX/PXB、P2P 慢 | 没走 NVLink / offload 频繁 | 关 ACS、查拓扑（§3.3.4） |
| **IB（跨节点）** | nsys 跨节点 AllReduce 慢 | 带宽不足 / 没 GDR | 多 QP、GDR、SHARP（§3.3） |
| **kernel launch** | profiler launch 占比高、GPU 闲 | 小 batch、kernel 碎 | CUDA Graph（§11.7） |
| **CPU** | nsys CPU 行满、GPU 等 | Python 开销、dataloader | 多进程、减 Python |

> 端到端纪律：**nvidia-smi（硬件正常吗）→ Roofline（哪类 bound）→ profiler/nsys（时间花哪、等谁）→ ncu/NCCL trace（单点深挖）→ 阶段旋钮（落地修）。** 五步下钻，每步缩小范围，最终落到前面某一章的具体优化手段。性能分析不是玄学——它是一条有纪律的下钻路径，本章的工具是路上的每一级台阶。

---

## 11.9 常见坑与 FAQ

1. **不 profile 就优化**：凭直觉改代码，改错地方。**永远先测量**（§11.0 第一原则）——profiler 五分钟省你五小时瞎改。
2. **profiler 没跳 warmup**：首次 launch / autotune / cudnn 选算法的开销污染数据。用 `schedule(wait,warmup,active)` 跳过（§11.3.1）。
3. **一上来就 ncu**：太细、太慢、淹没在细节。先 torch.profiler / nsys 缩小范围，锁定 kernel 再 ncu（§11.1、§11.5.5）。
4. **看平均延迟**：平均正常不代表没问题，p99 暴涨才是用户痛点。盯分位数（§10.7.2、§11.8.4）。
5. **给 memory-bound 算子换更快计算 kernel**：方向错。memory-bound 要减访存，不是提算力（§11.2）。
6. **`sm%` 高就以为算力打满**：sm% 包括 CUDA core，不等于 Tensor Core 利用率。要看 ncu 的 TC throughput（§11.5.3）。
7. **CUDA Graph 治所有慢**：它只治 launch overhead。compute-bound、kernel 慢都治不了（§11.7.1）。
8. **多卡 hang 盲猜**：用 flight recorder（`TORCH_NCCL_TRACE_BUFFER_SIZE`）定位，尤其抓 collective mismatch（§11.6.3）。
9. **nsys trace 太大打不开**：用 NVTX 圈定范围 + `--capture-range` 只抓关键段（§11.4.1）。
10. **现成 kernel 还想 ncu 优化**：FlashAttention/cuBLAS 已近最优，ncu 挖了也改不动。ncu 是给自定义 kernel 用的（§11.5.5）。

---

## 11.10 延伸阅读

- **PyTorch Profiler 官方教程 + `torch.profiler` 文档** — §11.3 的权威参考，含 TensorBoard 插件可视化。
- **NVIDIA Nsight Systems User Guide** — nsys 的完整手册，§11.4 的深入，重点看 timeline 和 NCCL trace 部分。
- **NVIDIA Nsight Compute 文档（Kernel Profiling Guide）** — ncu 的指标释义，§11.5 的 SOL / occupancy / 访存指标怎么读。
- **NVIDIA《GPU Performance Background User's Guide》** — Roofline 和 bound 类型判定的官方教程，§11.2 的理论基础（阶段 0 也引过）。
- **Horace He《Making Deep Learning Go Brrrr From First Principles》** — compute / memory / overhead 三类 bound 的分类法，本章的方法论源头。
- **PyTorch NCCL Flight Recorder 文档 / 博客** — §11.6.3 多卡 hang 排障的官方手段。
- **vLLM / SGLang 的 benchmark 与 profiling 文档** — 把本章工具用到真实引擎上的实操指引。
- **`nccl-tests` README** — 通信微基准的标尺（阶段 3 §3.10 也引），§11.6 对照理论 busbw 的依据。

---
