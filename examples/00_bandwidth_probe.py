# bandwidth_probe.py — H100 / A100 / MI300X 都能跑
import torch, time

def hbm_bw_gbps(gb=4.0, iters=20):
    n = int(gb * 1024**3 / 4)
    x = torch.empty(n, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)                          # HBM→HBM，读+写 = 2 × gb
    torch.cuda.synchronize()
    return 2 * gb * iters / (time.perf_counter() - t0)

def p2p_bw_gbps(src=0, dst=1, gb=1.0, iters=20):
    x = torch.empty(int(gb * 1024**3 / 4), dtype=torch.float32, device=f'cuda:{src}')
    y = torch.empty_like(x, device=f'cuda:{dst}')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()
    return gb * iters / (time.perf_counter() - t0)

def launch_overhead_us(iters=10000):
    a = torch.zeros(1, device='cuda')      # 接近空 kernel
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a.add_(1.0)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters

if __name__ == '__main__':
    print(f'HBM copy bandwidth : {hbm_bw_gbps():.0f} GB/s')
    if torch.cuda.device_count() >= 2:
        print(f'P2P GPU0 -> GPU1   : {p2p_bw_gbps():.0f} GB/s')
    print(f'Kernel launch cost : {launch_overhead_us():.1f} μs')
