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
