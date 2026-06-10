# examples｜书里代码的"可跑版"

正文里的可运行代码片段，都抽到了这里——**clone 下来直接跑，不用从 markdown 里手抄**（手抄最容易缩进错、版本对不上）。每个脚本和书里的内容**逐字一致**，跑通后回对应章节看讲解。

## 脚本一览

| 脚本 | 对应章节 | 跑它学什么 | 需要 GPU？ |
|---|---|---|---|
| [`P_first_run.py`](P_first_run.py) | 前置篇 P.4 | 你的第一个大模型推理：文字 → token → 模型 → 回复 | ❌ CPU 可跑 |
| [`02_rank_sim.py`](02_rank_sim.py) | 阶段 2 §2.4.4 | 3D 并行的 rank 拓扑：16 卡怎么映射、TP 是否 intra-node | ❌ 纯 Python，无需 torch |
| [`01_llama_mini.py`](01_llama_mini.py) | 阶段 1 §1.4 | ~110 行手写 LLaMA forward（RMSNorm+GQA+RoPE+SwiGLU） | ⚠️ CPU 能跑（默认 cuda，见下方改法） |
| [`00_bandwidth_probe.py`](00_bandwidth_probe.py) | 阶段 0 §0.4 | 实测你这台机器的 HBM 带宽 / P2P / kernel launch 开销 | ✅ 需要 NVIDIA GPU |
| [`04_fused_rmsnorm_triton.py`](04_fused_rmsnorm_triton.py) | 阶段 4 §4.6 | 手写 Triton 融合 kernel（add+RMSNorm）+ 和 PyTorch 比速度 | ✅ 需要 GPU + triton |

## 怎么跑

### 1. 建环境（推荐）

```bash
python -m venv llm-env
source llm-env/bin/activate        # Windows: llm-env\Scripts\activate
pip install -r requirements.txt    # 按需装，见 requirements.txt 注释
```

### 2. 从"零依赖"的开始

**完全不用装任何东西**，先跑这个纯 Python 的：

```bash
python 02_rank_sim.py
# 输出：16 个 rank 的 (tp,pp,dp) 坐标 + "TP 全部 intra-node [PASS]"
```

**只需 transformers**（CPU 也行），跑你的第一个真实推理：

```bash
pip install transformers torch
python P_first_run.py
# 输出：输入被切成了 N 个 token / 模型回复：……
```

### 3. 有 GPU 再跑这些

```bash
# 手写 LLaMA forward（默认用 cuda；没 GPU 就把脚本里 .cuda() 改成 .cpu()、bfloat16() 去掉）
python 01_llama_mini.py

# 测你的卡的真实带宽（必须有 NVIDIA GPU）
python 00_bandwidth_probe.py

# Triton 融合 kernel + benchmark（需要 GPU + triton）
python 04_fused_rmsnorm_triton.py
```

## 小白提示

- **先跑 `02_rank_sim.py` 和 `P_first_run.py`**——它们不挑硬件，跑通就建立了信心。
- 没有 GPU 也能学完前置篇 + 阶段 0/1 的核心，GPU 相关的脚本等你上云（前置篇 P.3.1）再跑。
- 某个脚本报错先看 [前置篇 P.6 常见坑](../chapters/0-onboarding.md#p6-常见坑与-faq)（下载慢、CUDA 不可用、显存不够等）。
