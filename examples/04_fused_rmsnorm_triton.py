import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_rmsnorm_kernel(
    x_ptr, residual_ptr, out_ptr, weight_ptr,
    n_tokens, hidden, eps,
    BLOCK_SIZE: tl.constexpr,   # 编译期常量，下一 power-of-2 ≥ hidden
):
    # 每个 program 处理一行 token
    token_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden

    row_offset = token_idx * hidden

    # 1) 读 x、residual，相加（FP32 精度防长 seq 漂移）
    x = tl.load(x_ptr        + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    z = x + r                           # ← 融合点：不写回 HBM

    # 2) RMSNorm：rsqrt(mean(z²) + eps) · z · weight
    var  = tl.sum(z * z, axis=0) / hidden
    rstd = 1.0 / tl.sqrt(var + eps)
    w    = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out  = z * rstd * w

    # 3) 写回（cast 回原 dtype）
    tl.store(out_ptr + row_offset + cols, out.to(x_ptr.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    n_tokens, hidden = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(hidden)
    grid = (n_tokens,)                  # 一维 grid，每 program 一行
    fused_add_rmsnorm_kernel[grid](
        x, residual, out, weight,
        n_tokens, hidden, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ====== 参考实现 + benchmark ======

def pytorch_reference(x, residual, weight, eps=1e-6):
    z    = (x + residual).to(torch.float32)              # kernel 1: add
    var  = z.pow(2).mean(-1, keepdim=True)               # kernel 2: pow + mean
    rstd = torch.rsqrt(var + eps)
    return (z * rstd * weight).to(x.dtype)               # kernel 3: scale

def bench(fn, *args, iters=1000):
    import time
    for _ in range(10): fn(*args)                        # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters      # μs

if __name__ == '__main__':
    n_tokens, hidden = 8192, 4096                        # LLaMA-2-7B 量级
    x   = torch.randn(n_tokens, hidden, device='cuda', dtype=torch.bfloat16)
    res = torch.randn_like(x)
    w   = torch.randn(hidden, device='cuda', dtype=torch.bfloat16)

    out_triton = fused_add_rmsnorm(x, res, w)
    out_ref    = pytorch_reference (x, res, w)
    print(f'max diff      : {(out_triton - out_ref).abs().max().item():.4f}')

    t_triton = bench(fused_add_rmsnorm, x, res, w)
    t_torch  = bench(pytorch_reference, x, res, w)
    print(f'Triton fused  : {t_triton:5.1f} μs')
    print(f'PyTorch (3 kn): {t_torch :5.1f} μs   ({t_torch / t_triton:.1f}× slower)')
