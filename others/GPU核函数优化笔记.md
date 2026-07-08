# GPU / NPU 核函数优化笔记

> 核心视角:几乎所有 folklore 都能从三个第一性事实推出来。
>
> 1. **GPU/NPU 是吞吐机器,不是延迟机器** —— 不靠大缓存兜底,靠海量在途并发。
> 2. **访存按"事务"计费,不按"字节"计费** —— 一次搬运固定粒度的段。
> 3. **SIMT:32 个线程锁步执行** —— 发散即串行。
>
> 记住:下文所有定量参数(事务粒度、饱和所需并发数、原子/fence 相对开销)都是 NVIDIA 硬件上的数值。原理可迁移到自研 NPU,但每条的"档位"必须按自家硬件的串行化规则与链路特性重新实测标定。

---

## 一、事实一:访存按"事务"计费,不按"字节"计费

内存控制器一次搬运固定大小的段(NVIDIA L2 事务 32B,DRAM burst 更大)。由此推出一族优化。

| 优化 | 原理 | 代价 / 注意 |
|------|------|-------------|
| **合并访存 (coalescing)** | 同 warp 32 线程访问连续地址,硬件合并成最少事务。跨步访问或 AoS 布局让每个事务只有几字节有用,有效带宽除以浪费比例。 | 数据结构倾向 **SoA** 而非 AoS。 |
| **向量化访存** | 用 128-bit `ld.global.v4` / `st.global.v4`(`float4` / NCCL pack)一条指令搬 16B,指令数降为 1/4,访存队列压力同步下降。 | 强制 16B 对齐;头尾未对齐部分需单独标量路径(即 bcastDeep 里 `partial` 分支的理由之一)。 |
| **避免 bank conflict** | shared memory 32 个 bank,同 warp 两线程落到同一 bank 不同地址则串行化。 | 经典解法:二维数组加一列 **padding** 错开映射。 |

**关联代码点**:bcastDeep 里的 `tmp[u]` 就是 16B pack 向量化访存。

---

## 二、事实二:延迟靠"在途并发"掩盖,不是被消除

全局访存延迟数百周期。打满带宽的定量条件是 **Little's law**:

```
在途字节数 ≥ 带宽 × 延迟
```

凑够在途请求有两条路:

- **TLP(线程级并行)**:多驻留 warp,即 occupancy。
- **ILP(指令级并行)**:单线程内多条独立 load 背靠背发射。

### 由此推出的常识

1. **occupancy 是手段不是目标**
   Volkov 经典结论:更低 occupancy + 更高 ILP 反而更快 —— 寄存器里的操作数比 shared memory 里的更近。

2. **软件流水 / 双缓冲 (double buffering)**
   把"下一轮 load"提前到"本轮 store/compute"之前发射,让访存与计算/写出重叠。

   ```
   朴素串行:  load0 → store0 → load1 → store1     (load 期间链路空转)
   双缓冲:    load0
              store0  ‖  load1
              store1  ‖  load2   ...              (store 流量遮住 load 延迟)
   ```

   bcastDeep 主循环即教科书写法:循环末尾预取下一轮 `tmp`,循环开头做本轮远端 store,load 返回延迟被 store 流量完全遮住。

   **代价**:寄存器压力近似翻倍。`UnrollPacks × UnrollPeers` 每加一档都吃寄存器,溢出到 local memory 就得不偿失。展开系数要与 `__launch_bounds__` 一起调。

3. **grid-stride loop**
   块数按 SM 数算(而非按数据量),每线程循环处理多个元素。既摊薄启动开销,又天然适配任意规模。

---

## 三、事实三:32 线程锁步,发散即串行

warp 内分支两侧都会被执行(各自谓词化),所以条件尽量做成 **warp-uniform** —— 按 `warpId` 而非 `threadIdx` 分支,让整个 warp 走同一侧。

> bcastDeep 里 `partial`、`dr` 都是 warp 内一致值,发散为零,是刻意设计。

### 同源常识

- **warp 内交换用 `__shfl_sync`**,不走 shared memory(省一次往返 + 一次同步)。
- **warp-aggregated atomics**:多线程对同一地址原子操作时,先做 warp 内聚合,把 32 次原子合成 1 次。
- **能用 `__syncwarp` 就不用 `__syncthreads`**。

### 指令层面

- **消除整数除法/取模**:GPU 上 `/` 和 `%` 是几十条指令的软件序列。循环内 `% nRanks` 改写成比较加减 —— `if (++r == nRanks) r = 0` 就是这条 folklore 的体现。
- **编译期常数模板化**:把 `nRanks` 作为模板参数,让编译器做强度削减(strength reduction),消掉除法。
- **`__restrict__`**:消除别名分析障碍。
- **FMA / 快速数学开关**:按精度需求开启。

---

## 四、通信 kernel 特有优化(与对称内存 / NPU 通信库最相关)

### 1. push 优于 pull
- **远端 store** 是 fire-and-forget,发出即可继续。
- **远端 load** 必须等数据跨链路返回才能推进,暴露完整一个 RTT。
- 对称内存 kernel 几乎全是"写者主动推"(bcastDeep 亦然)。
- **例外**:pull 能省一跳的场合(如 reduce 时本地聚合);某些互连 read 通道更空闲时。FLUX 实测 PCIe 与 NVLink 上两者偏好相反 —— **按互连实测**。

### 2. 打满链路不需要很多计算单元
- NVLink 上 **8–16 个 SM** 即可饱和带宽,再多只有边际收益。
- 推论:通信 kernel 用尽量少的 block,把 SM 留给可重叠的计算。
- **persistent kernel**:常驻不退出,靠 flag 驱动下一轮,省掉每次 launch 的几微秒 —— 对小消息延迟敏感场景是决定性的。

### 3. 同步用最小充分的内存序强度
- 数据写完发 flag:老写法 `__threadfence_system()` + 普通 store;现代写法带 scope 的 `st.release` / `ld.acquire`(或 multimem 等价物),避免全域 fence 把在途写序列化。
- flag 轮询侧配 **backoff**(`nanosleep`),防止自旋流量挤占链路。
- **数据与 flag 布局**:分开放避免伪共享;或反过来把 flag 嵌进数据尾部,换取一次事务同时完成"数据 + 通知"。取舍点在消息大小。
- **跨链路原子 RMW**(fetch-add 等)延迟是纯 store 的数倍。能用**单写者协议**(每 rank 只写自己的 slot)就绝不用远端原子。

### 4. 按消息大小分档选算法

| 消息大小 | 界 | 算法 | 特点 |
|----------|-----|------|------|
| 小 | 延迟界 | **one-shot** | 每人直接写所有人,α 项最小,流量 N 倍冗余 |
| 中 | —— | **two-shot** | reduce-scatter + allgather,流量最优,两次同步 |
| 大 | 带宽界 | **ring / pipeline** | 摊平带宽 |

> NCCL 对 bcast 分出 Deep / Enhanced 多个 kernel、按 size 路由,即此条的工程化。

### 5. 移位 / 旋转调度(shift / rotation schedule)

**问题**:若所有源(所有 rank,或同一 rank 内所有 warp)按固定顺序 `0, 1, 2, …` 写 peer,则 t=0 时 N 份流量同时汇聚到交换机通往 GPU 0 的**同一个出端口**(incast)。crossbar 再无阻塞,单出端口带宽固定,N 路排队串行化,有效带宽退化为 **1/N**;下一时刻大家又一起挤 GPU 1,逐个"轮流堵车"。

**手法**:每个源从"自己的下一个 rank"开始、模 N 循环写 —— 即代码中的 `r = (rank + dr) mod nRanks`,第 i 步通信模式变成循环置换 `σᵢ(k) = (k + i) mod N`。

**为什么在 switch 场景显著**:置换流量是 crossbar 最理想的输入 —— 每步每个入端口/出端口恰好承载一路流,无任何端口收敛,N 条链路同时跑满。

- 图论视角:等价于把完全二部图 `K_{N,N}` 的边做 **1-因子分解**(1-factorization),N−1 个循环置换不重不漏覆盖所有"源→目的"对。这也是 pairwise exchange 恰好需要 **p−1 步**、且每步无竞争的原因。

**代码细节(bcastDeep)**:

| 细节 | 作用 |
|------|------|
| `dr = inPlace ? 1 : 0` | in-place 时自己缓冲区已有数据,从 rank+1 起跳过自己;否则连自己也写一份 |
| warp 级同规则推进 + `UnrollPeers` 展开 | 使发往 N 个目的的 store 流**在时间上交织**,把端口冲突从粗粒度(整块)打散到细粒度(每 `UnrollPacks×WARP_SIZE` 个 pack),让 NVSwitch 仲裁器持续保持所有出端口忙碌 |
| `partial` 双层循环 | 主体按 UnrollPeers 满展开、余数单独收尾,保证主循环指令流水静态可调度 |

**局限(同源于原理)**:

- 假设底层是(近似)无阻塞 crossbar,置换流量才无冲突。若是 **torus / mesh 或多级收敛 fat-tree**,单纯 `(rank+i) mod N` 可能在中间链路撞车,需换成**拓扑感知的置换序列**。
- 对写延迟远大于带宽瓶颈的**极小消息**,旋转收益被同步开销掩盖 —— 也是分 kernel 按 size 路由的动机之一。

**出处 / prior art**:该调度是并行计算经典手法,公开文献可追溯到 1990s。

- 论文:Bruck et al. 1997(IEEE TPDS,基于 index rotation);Thakur/Rabenseifner/Gropp 的 MPICH 集合通信优化(2003 会议 / 2005 IJHPCA),明确写出"第 i 步向 `rank+i` 发、从 `rank−i` 收"。
- 专利:富士通 US 10,484,264(fat-tree 中 `(k+i) % N` 的 shift pattern)、US 5,826,033 / US 10,430,375 / EP 2302525(torus all-to-all 相位划分)、IBM US 9,251,118(n 维 torus/mesh 调度)。
- 结论:**思想层面属现有技术,自研使用新颖性风险低**;若要自己申请专利,单纯 rank 旋转难过新颖性,但"旋转调度 + NPU 特有的跨 die PBLink / SISO 同步机制的组合"仍可能有可专利空间。

---

## 六、迁移到自研 NPU 的提醒

- 上述**原理**普遍可迁移,但**每条的"档位"要按自家硬件重新实测标定**。
- 差异最大的三处:**指令排序规则**、**同步原语强度**、**链路特性**。
- 照抄 CUDA folklore 的**具体数字**是这类移植中最常见的坑。
