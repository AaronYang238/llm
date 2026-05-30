# 阶段 1｜Transformer 与单卡推理基础

> 一句话定位：把现代 decoder-only LLM 拆到模块级——Embedding、RMSNorm、GQA Attention、SwiGLU FFN、RoPE、采样——并用 ~150 行 PyTorch 完整复刻一遍 LLaMA forward，作为 Capstone P1（mini-vLLM）的起点。

## 目录

- [1.0 为什么需要这一层](#10-为什么需要这一层)
- [1.1 核心概念与术语](#11-核心概念与术语)
- [1.2 原理详解](#12-原理详解)
- [1.3 关键实现剖析：vLLM LlamaModel](#13-关键实现剖析vllm-llamamodel)
- [1.4 最小可运行示例：~150 行 LLaMA forward](#14-最小可运行示例150-行-llama-forward)
- [1.5 性能与显存估算](#15-性能与显存估算)
- [1.6 常见坑与 FAQ](#16-常见坑与-faq)
- [1.7 延伸阅读](#17-延伸阅读)

---

## 1.0 为什么需要这一层

你要写 mini-vLLM、读 vLLM / SGLang 源码、判断 KV cache 该用 GQA 还是 MLA——这些都需要先在脑子里有一张精确的 Transformer block 拓扑：每一层进什么 shape、出什么 shape、显存占多少、算量分布在哪。

读完这一章你应当能：

- 默写一个 LLaMA decoder block 的前向，含每一步 shape
- 给定 model config（`hidden_size`、`num_heads`、`num_kv_heads`、`d_ff`）口算单层参数量、单层算量、KV cache 大小
- 解释 MHA → GQA → MLA 的演进为什么是为推理服务的
- 选对采样参数：`temperature` / `top_p` / `min_p` 的相互作用

---

## 1.1 核心概念与术语

| 缩写 | 全称 | 一句话 |
|---|---|---|
| D / hidden | hidden_size | token embedding 维度 |
| H | num_heads | Q head 数 |
| H_kv | num_kv_heads | KV head 数（MHA 时 H_kv=H；GQA 时 H_kv<H） |
| d | head_dim | 单 head 维度，通常 D / H |
| d_ff | ffn_dim | FFN 中间层维度，LLaMA 约 2.67×D |
| S | seq_len | 序列长度 |
| B | batch | batch size |
| V | vocab | 词表大小 |
| MHA | Multi-Head Attention | 标准多头 |
| MQA | Multi-Query Attention | 所有 Q head 共享 1 个 KV head（H_kv=1） |
| GQA | Grouped-Query Attention | g 个 Q head 共享一个 KV head |
| MLA | Multi-head Latent Attention | DeepSeek，KV 压缩到低秩潜空间 |
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| YaRN | Yet another RoPE extensioN | 长上下文外推 |
| RMSNorm | Root Mean Square Norm | LayerNorm 简化版，省去均值 |
| SwiGLU | Swish-Gated Linear Unit | 门控 FFN：SiLU(W_gate x) ⊙ (W_up x) |
| LM head | Language Modeling Head | 最后一层 Linear，hidden → vocab |
| KV cache | Key/Value Cache | 历史 K/V 缓存，让 decode 复杂度从 O(S²) 降到 O(S) |

---

## 1.2 原理详解

### 1.2.1 LLaMA 风格 Decoder Block

![LLaMA-style Decoder Block (GQA + SwiGLU + RoPE)](../svg/09-transformer-block.svg)

整个 block 的张量流（pre-norm，LLaMA 起沿用至今）：

```
x         = [B, S, D]
h         = x + Attention(RMSNorm(x))
y         = h + FFN(RMSNorm(h))
```

Attention 子块（GQA）展开：

```
xn    = RMSNorm(x)              [B, S, D]
Q     = xn @ W_Q                [B, S, H,    d]
K     = xn @ W_K                [B, S, H_kv, d]
V     = xn @ W_V                [B, S, H_kv, d]
Q, K  = RoPE(Q, pos), RoPE(K, pos)
K, V  = append(KV_cache, K), append(KV_cache, V)   # decode 时复用历史
out   = softmax(Q · K^T / √d) · V                   # K/V 在 head 维上广播 H/H_kv 倍
out   = out @ W_O               [B, S, D]
```

FFN 子块（SwiGLU）：

```
h = RMSNorm(...)                [B, S, D]
y = (SiLU(h @ W_gate) ⊙ (h @ W_up)) @ W_down        [B, S, D]
```

注意 SwiGLU 比经典 FFN 多了 `W_gate`，参数量从 `2 × D × d_ff` 增到 `3 × D × d_ff`。LLaMA 一般取 `d_ff ≈ (2/3) × 4D ≈ 2.67D`，让总参数量与原始 FFN 持平。

### 1.2.2 Attention 家族演进

| 类型 | H_kv | KV cache 体积 | 代表模型 | 推理收益 |
|---|---|---|---|---|
| MHA | H | 100% | GPT-3、LLaMA-1 | baseline |
| MQA | 1 | 1/H | PaLM、Falcon | KV 极致省，质量略降 |
| GQA | H/g（g=4 或 8 常见） | g/H | LLaMA-2/3、Qwen2/3、Mistral | KV 缩 g 倍，质量基本无损 |
| MLA | 低秩潜变量（典型 512）+ 解耦 RoPE | ~1/16（DeepSeek-V3） | DeepSeek V2/V3/R1 | 长 context 极致省，训练较复杂 |

KV cache 体积公式（FP16，L 层）：

$$\text{KV bytes} = 2 \times L \times 2 \times B \times S \times H_{kv} \times d$$

前 `2` 是 K+V，后 `2` 是 FP16 字节数。

**例**：LLaMA-3-70B（L=80，D=8192，H=64，H_kv=8，d=128），B=1，S=8192：

```
KV = 2 × 80 × 2 × 1 × 8192 × 8 × 128  ≈  2.68 GB
```

若按 MHA（H_kv=64），就是 **21.5 GB**，单卡都装不下大 batch。GQA 是 LLaMA-2 之后能做长 context + 大 batch 的关键工程改动。

### 1.2.3 位置编码

LLM 主流是 **RoPE**：把每个 head 的 d 维向量两两配对，按位置 m 旋转角度 m·θ_i，θ_i = base^(−2i/d)，base 通常取 10000。

$$\text{RoPE}(x, m)_{2i,\,2i+1} =
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

优势：相对位置信息直接编码进 attention 内积，外推更友好。

**长上下文外推**——训练 S=4K 想推理时跑 128K，主要靠：

| 方法 | 思路 | 备注 |
|---|---|---|
| PI（Position Interpolation） | 把 m 缩成 m·(L_train/L_infer)，全维度统一压缩 | 简单粗暴，质量下降明显 |
| NTK-aware | 调 base，让高频维度少压、低频多压 | 社区常见 |
| YaRN | 分频段不同策略，质量目前最好的开源方案 | LLaMA-3.1 / Qwen2.5 都用 |
| LongRoPE | 每维独立搜索缩放系数 | Phi-3-128K 等 |

### 1.2.4 MoE 路由（仅 MoE 模型）

仅列名词，工程细节留到阶段 9：

- **Top-K**：每 token 取分数最高的 K 个 expert（Mixtral、Qwen3-MoE 是 Top-2 / Top-8）
- **Switch**：K=1，最简单
- **Expert Choice**：反过来由 expert 选 token，天然均衡
- **Loss-Free Balance**：DeepSeek-V3 用，靠偏置项动态平衡负载，无需辅助 loss

### 1.2.5 采样与解码

logits 出来后的常见管线：

```
logits = lm_head(h_last)                       # [B, V]
logits = logits / temperature
logits = top_k_filter(logits, k=40)
logits = top_p_filter(logits, p=0.9)
logits = min_p_filter(logits, p=0.05)
probs  = softmax(logits)
next   = multinomial(probs)
```

| 参数 | 作用 | 直觉 |
|---|---|---|
| temperature | logits / T | T<1 更确定，T>1 更随机；T→0 即 greedy |
| top-k | 只保留概率最高的 k 个 | 简单截断 |
| top-p (nucleus) | 累计概率达 p 的最小集合 | 自适应长度 |
| min-p | 保留 P > p · max(P) 的 token | 对长尾更鲁棒，社区新派 |
| typical | 抑制偏离典型熵的 token | 防奇怪输出 |
| DRY / no-repeat-ngram | 抑制重复 | 多轮 RP 场景常用 |

结构化输出（JSON Schema、正则、CFG）是在 softmax **之前**对 logits mask 掉所有不合法 token——xgrammar / outlines / lm-format-enforcer 的工程实现详见阶段 10。

---

## 1.3 关键实现剖析：vLLM LlamaModel

vLLM 的 LLaMA 实现在 `vllm/model_executor/models/llama.py`。骨架和我们手写版完全一致，但所有 Linear 被替换成 **TP-aware 版本**（`QKVParallelLinear`、`MergedColumnParallelLinear`、`RowParallelLinear`），attention 走 `vllm.attention.Attention`（内部分派到 FlashAttn / PagedAttention / FlashInfer）。

简化骨架：

```python
class LlamaAttention(nn.Module):
    def __init__(self, config):
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=config.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
        )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * config.head_dim,
            output_size=config.hidden_size,
        )
        self.rotary_emb = get_rope(...)
        self.attn = Attention(num_heads_local, head_size, scale, num_kv_heads_local)

    def forward(self, positions, hidden_states, kv_cache, attn_metadata):
        qkv, _   = self.qkv_proj(hidden_states)
        q, k, v  = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k     = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v, kv_cache, attn_metadata)
        out, _   = self.o_proj(attn_out)
        return out
```

值得记下的几个工程决策：

1. **QKV 合并成一个 Linear**：少一次 launch、复用 input activation。是几乎所有推理引擎的标配。
2. **`positions` 是显式参数**：vLLM 的 attention metadata 把 prefill / decode 的位置统一编码，不依赖 RNN 风格的隐含 state，方便 continuous batching。
3. **`kv_cache` 是 PagedAttention 的 block table**：这里只看到接口，物理布局详见阶段 5。
4. **TP rank 切分发生在 `QKVParallelLinear` 内部**：每张卡只持有自己负责的 head 子集；GQA 时 KV head 不能再切（已经只有 H_kv 个），所以 TP ≤ H_kv 是硬约束——详见阶段 2。

读 vLLM 时建议这条路径：`model_executor/models/llama.py` → `attention/layer.py` → `attention/backends/flash_attn.py`。

---

## 1.4 最小可运行示例：~150 行 LLaMA forward

下面这段代码是一个**单卡、纯 PyTorch、无外部依赖**的 LLaMA forward。形状全部标注，可直接 `python llama_mini.py` 跑通。

```python
# llama_mini.py — 单卡 prefill 演示（不含 KV cache 管理，专注架构）
import torch, torch.nn as nn, torch.nn.functional as F

class Config:
    vocab_size = 32000
    hidden_size = 1024
    n_layers = 4
    n_heads = 16
    n_kv_heads = 4            # GQA：16 Q / 4 KV → group=4
    head_dim = 64             # = hidden_size // n_heads
    d_ff = 2752               # ≈ 2.67 × hidden_size
    rope_base = 10000.0
    max_seq = 2048

cfg = Config()

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        # rms 计算在 FP32 做，避免长序列尾部数值漂移
        v = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * v).to(x.dtype) * self.w

def build_rope_cache(seq, d, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))     # [d/2]
    pos = torch.arange(seq).float()
    freqs = torch.outer(pos, inv_freq)                                  # [seq, d/2]
    return torch.cos(freqs), torch.sin(freqs)                           # 各 [seq, d/2]

def apply_rope(x, cos, sin):
    # x: [B, S, H, d]； cos/sin: [S, d/2]
    x1, x2 = x[..., ::2], x[..., 1::2]                                  # [B,S,H,d/2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.wq = nn.Linear(c.hidden_size, c.n_heads    * c.head_dim, bias=False)
        self.wk = nn.Linear(c.hidden_size, c.n_kv_heads * c.head_dim, bias=False)
        self.wv = nn.Linear(c.hidden_size, c.n_kv_heads * c.head_dim, bias=False)
        self.wo = nn.Linear(c.n_heads    * c.head_dim, c.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        c = self.c
        q = self.wq(x).view(B, S, c.n_heads,    c.head_dim)
        k = self.wk(x).view(B, S, c.n_kv_heads, c.head_dim)
        v = self.wv(x).view(B, S, c.n_kv_heads, c.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # GQA：把 KV 在 head 维度复制 group 次
        g = c.n_heads // c.n_kv_heads
        k = k.repeat_interleave(g, dim=2)                              # [B,S,H,d]
        v = v.repeat_interleave(g, dim=2)
        # [B, H, S, d]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # PyTorch ≥ 2.0 会自动选 FlashAttn / mem-efficient / math 后端
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B,H,S,d]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)

class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w_gate = nn.Linear(c.hidden_size, c.d_ff,       bias=False)
        self.w_up   = nn.Linear(c.hidden_size, c.d_ff,       bias=False)
        self.w_down = nn.Linear(c.d_ff,        c.hidden_size, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.hidden_size)
        self.attn      = Attention(c)
        self.ffn_norm  = RMSNorm(c.hidden_size)
        self.ffn       = SwiGLU(c)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class MiniLlama(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed   = nn.Embedding(c.vocab_size, c.hidden_size)
        self.blocks  = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm    = RMSNorm(c.hidden_size)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        cos, sin = build_rope_cache(c.max_seq, c.head_dim, c.rope_base)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
    def forward(self, ids):                                            # ids: [B, S]
        B, S = ids.shape
        x = self.embed(ids)                                            # [B, S, D]
        cos, sin = self.cos[:S], self.sin[:S]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.lm_head(self.norm(x))                              # [B, S, V]

if __name__ == '__main__':
    torch.manual_seed(0)
    model = MiniLlama(cfg).cuda().bfloat16()
    ids   = torch.randint(0, cfg.vocab_size, (2, 64), device='cuda')
    logits = model(ids)
    print('logits.shape =', tuple(logits.shape))
    print('param count  =', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
```

H100 SXM 上预期输出（< 1 秒）：

```
logits.shape = (2, 64, 32000)
param count  = 116.5 M
```

把 config 改成 `n_layers=32, hidden_size=4096, n_heads=32, n_kv_heads=8, d_ff=11008`，就是 LLaMA-2-7B 的骨架；外接 `safetensors` 加载真权重就能跑出真实输出。Capstone P1 的 mini-vLLM 在此基础上加 **KV cache 管理 + 分页 + continuous batching scheduler** 即成。

---

## 1.5 性能与显存估算

记住三组拍脑袋公式（B=batch，S=seq，L=层数）：

**单层参数量**（GQA + SwiGLU，不含 norm）：

$$P_{\text{layer}} = \underbrace{D \cdot (H + 2H_{kv}) \cdot d}_{QKV} + \underbrace{H \cdot d \cdot D}_{W_O} + \underbrace{3 \cdot D \cdot d_{ff}}_{SwiGLU}$$

LLaMA-2-7B 校验：32 × (12.6M + 16.8M + 135M) + embedding 131M ≈ **6.7 B** ✓。

**单层 FLOPs**（forward，prefill 阶段，忽略 norm / softmax）：

$$F_{\text{layer}} \approx 2 B S \cdot P_{\text{layer}} + \underbrace{4 B H S^2 d}_{\text{attention } QK^T,\ AV}$$

第一项 ≈ "2 × 参数量 × token 数"，第二项是 attention 的 $O(S^2)$ 部分；当 S 较大（≥ 4K）时第二项主导。

**KV cache 体积**（FP16）：见 1.2.2 公式。

**Roofline 速判**：

| 阶段 | 主导算子 | 计算特征 | 优化方向 |
|---|---|---|---|
| Prefill（S 大、batch 任意） | Linear GEMM、attention $QK^\top$ | compute-bound | 提 Tensor Core 利用率、FlashAttention |
| Decode（S=1、batch 小） | Linear GEMV、KV cache 读 | **memory-bound** | continuous batching 把 GEMV 升 GEMM |
| Decode（S=1、batch 大） | Linear GEMM、PagedAttention | compute-bound | 提调度器吞吐、CUDA Graph |

这就是为什么所有推理引擎都拼命做 **continuous batching**——把多条 decode 拼成一个大 batch，把 memory-bound 的 GEMV 升级成 compute-bound 的 GEMM。

---

## 1.6 常见坑与 FAQ

1. **RoPE 实现差异**：HuggingFace 走"复数 stack"路径（两个 half 拼接），llama.cpp 走"两两交错"路径（even/odd 交错）。权重不能跨实现直接用，rope cache 也要重排。
2. **GQA repeat 写错**：必须 `repeat_interleave(g, dim=2)`，不能用 `repeat(1,1,g,1)`——后者会把 head 顺序打乱，等价于让 attention 学了错误的 head 映射。
3. **`scaled_dot_product_attention` 默认 backend**：PyTorch ≥ 2.0 会在 FlashAttn / mem-efficient / math 之间自动选。生产代码要 `with sdpa_kernel(SDPBackend.FLASH_ATTENTION):` 显式锁定，避免长序列回退到 math 后端跑爆显存。
4. **`bfloat16` 下 RMSNorm 数值**：rms 计算在 FP32 做完再 cast 回 BF16，否则长序列尾部精度漂移、推理质量肉眼可见地下降。
5. **`lm_head` 是否与 embedding tie**：LLaMA 不 tie，Qwen2 tie，Gemma tie。加载权重时认错会导致输出乱码或 vocab size 对不上。
6. **decode 时整段 KV 重算**：典型新手错误是没维护 KV cache，每步重新 forward 完整前缀，复杂度从 O(S) 升到 O(S²)。
7. **`temperature=0`**：不要除 0；T=0 时直接走 argmax。
8. **TP 切分 + GQA**：TP 度必须 ≤ `num_kv_heads`，否则 KV head 无法均分。LLaMA-3-8B 的 `num_kv_heads=8`，TP=8 是上限。
9. **`apply_rope` 在 FP16/BF16 下**：cos/sin 建议保持 FP32 缓存，乘的时候临时 cast，避免长序列高频项被吃掉精度。

---

## 1.7 延伸阅读

- **LLaMA / LLaMA-2 / LLaMA-3 技术报告** — GQA、RoPE base 调整、词表演进的工程动机。
- **《GQA: Training Generalized Multi-Query Transformer》(Ainslie 2023)** — 看 MHA → MQA → GQA 的实证 trade-off。
- **《RoFormer: Enhanced Transformer with Rotary Position Embedding》** — RoPE 原论文。
- **《YaRN: Efficient Context Window Extension》** — 长上下文外推的标准工程参考。
- **DeepSeek-V2 报告 MLA 章节** — 看 KV cache 还能怎么压。
- **HuggingFace `transformers/models/llama/modeling_llama.py` 与 vLLM `model_executor/models/llama.py`** — 同一架构两种风格的工程实现对照，是阶段 6 推理引擎深读的入口。
