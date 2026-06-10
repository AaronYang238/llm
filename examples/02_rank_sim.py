# rank_sim.py —— 3D 并行 rank 拓扑模拟器（纯 Python，无需 GPU/torch）
from __future__ import annotations
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台强制 UTF-8


@dataclass
class Layout:
    tp: int
    pp: int
    dp: int
    gpus_per_node: int

    @property
    def world(self) -> int:
        return self.tp * self.pp * self.dp


def coords(rank: int, L: Layout) -> tuple[int, int, int]:
    """Megatron 默认排布：tp 最内、dp 居中、pp 最外。"""
    tp_i = rank % L.tp
    dp_i = (rank // L.tp) % L.dp
    pp_i = rank // (L.tp * L.dp)
    return tp_i, pp_i, dp_i


def group(rank: int, L: Layout, kind: str) -> list[int]:
    tp_i, pp_i, dp_i = coords(rank, L)
    out = []
    for r in range(L.world):
        t, p, d = coords(r, L)
        if kind == "tp" and (p, d) == (pp_i, dp_i):
            out.append(r)
        elif kind == "pp" and (t, d) == (tp_i, dp_i):
            out.append(r)
    return out


def node_of(rank: int, L: Layout) -> int:
    return rank // L.gpus_per_node


def report(L: Layout) -> None:
    print(f"配置: TP={L.tp} × PP={L.pp} × DP={L.dp} = world {L.world}; "
          f"{L.world // L.gpus_per_node} 节点 × {L.gpus_per_node} GPU\n")
    print(f"{'rank':>4} {'node':>4} {'tp':>3} {'pp':>3} {'dp':>3}")
    for r in range(L.world):
        t, p, d = coords(r, L)
        print(f"{r:>4} {node_of(r, L):>4} {t:>3} {p:>3} {d:>3}")

    print("\nTP 组（必须 intra-node）:")
    seen, ok = set(), True
    for r in range(L.world):
        g = tuple(group(r, L, "tp"))
        if g in seen:
            continue
        seen.add(g)
        nodes = {node_of(x, L) for x in g}
        flag = "OK" if len(nodes) == 1 else "CROSS-NODE!!"
        ok &= len(nodes) == 1
        print(f"  {list(g)}  -> node {sorted(nodes)}  [{flag}]")

    print("\n校验:", "TP 全部 intra-node，拓扑合法 [PASS]" if ok
          else "存在 TP 跨节点，会严重掉速 [FAIL]")


if __name__ == "__main__":
    report(Layout(tp=2, pp=2, dp=4, gpus_per_node=8))
