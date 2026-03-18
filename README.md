# vLLM Mixtral 8x7B MoE Optimization

[![vLLM](https://img.shields.io/badge/vLLM-0.13.0-blue?logo=python)](https://github.com/vllm-project/vllm)
[![A100](https://img.shields.io/badge/GPU-2×A100_80GB-76B900?logo=nvidia)](https://www.nvidia.com/en-us/data-center/a100/)
[![Mixtral](https://img.shields.io/badge/Model-Mixtral_8x7B-orange)](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

MoE inference optimization on dual A100 80GB GPUs. **2.1× throughput improvement** over baseline vLLM BF16 serving — from 57 tok/s to 120 tok/s — using AWQ Marlin quantization, expert parallelism, FP8 KV cache, and continuous batching.

---

## Performance Results

| Metric | Baseline (vLLM BF16, TP2) | Optimized (AWQ + EP + FP8 KV) | Delta |
|--------|--------------------------|-------------------------------|-------|
| **Throughput** | ~57 tok/s | 120 tok/s | **+2.1×** |
| **TTFT** | ~85ms | 21ms | **−75%** |
| **ITL** | ~18ms | 8ms | **−56%** |
| **Model VRAM** | ~90GB | ~26GB | **−71%** |
| **Prefix Cache Hit Rate** | — | 60% (multi-turn) | — |

**Hardware:** 2× NVIDIA A100 PCIe 80GB | CUDA 12.x | vLLM 0.13.0  
**Benchmark:** Single-turn requests, 256 token output, measured after CUDA graph warmup.

---

## Optimization Decisions

**AWQ Marlin 4-bit Quantization**  
Reduced model VRAM from ~90GB to ~26GB — the single change that made dual A100 deployment viable. Without it, the model doesn't fit (90GB > 2×80GB with KV cache headroom). Used `casperhansen/mixtral-instruct-awq` pretrained AWQ checkpoint rather than quantizing from scratch. Marlin kernel is specifically optimized for NVIDIA GPU GEMM at 4-bit, unlike generic AWQ kernels.

**Expert Parallelism over pure Tensor Parallelism**  
Mixtral activates only 2 of 8 experts per token. With pure TP2, both GPUs participate in every expert computation — high all-reduce communication overhead on every forward pass. With EP, each GPU owns 4 experts exclusively:
- GPU 0: Experts 0, 1, 2, 3
- GPU 1: Experts 4, 5, 6, 7

Expert dispatch replaces all-reduce with point-to-point routing. Communication overhead drops significantly for the decode phase, which is the primary bottleneck on memory-bandwidth-limited hardware like PCIe A100s. Combined with `--tensor-parallel-size 2` for the attention layers, this hybrid EP+TP approach is what drives the throughput gain over naive TP2.

**FP8 KV Cache**  
`--kv-cache-dtype fp8_e5m2` cuts KV cache memory by 50%, freeing VRAM for larger batch sizes. Enabled 1.9M token cache capacity on this configuration. The accuracy tradeoff at FP8 for KV cache is minimal — KV values are already computed post-attention and don't compound quantization error the way weight quantization does.

**Chunked Prefill**  
Without it, a long prompt monopolizes the GPU during prefill, stalling all ongoing decode requests. Chunked prefill interleaves prefill chunks with decode steps, keeping ITL consistent even under mixed traffic. This is why ITL dropped from ~18ms to 8ms — not from the quantization itself.

**Prefix Caching**  
60% cache hit rate in multi-turn conversations. System prompts and conversation history reuse KV cache from previous turns. Single-turn queries get no benefit — the 60% figure is specific to multi-turn workloads.

**What I tried that didn't work:**  
Pure INT4 GPTQ without the Marlin kernel was slower than AWQ Marlin despite same bit-width — Marlin's dequantization-fused GEMM is what makes 4-bit viable at this throughput. Also tested `--tensor-parallel-size 2` without expert parallelism — the all-reduce overhead on PCIe interconnect (vs NVLink) was the primary bottleneck, costing roughly 30% throughput vs the EP+TP hybrid. On NVLink hardware, pure TP2 would likely close that gap.

---

## Tech Stack

| Component        | Technology                           |
|------------------|--------------------------------------|
| Inference Engine | vLLM 0.13.0                          |
| Model            | casperhansen/mixtral-instruct-awq    |
| Quantization     | AWQ Marlin 4-bit                     |
| KV Cache         | FP8 (fp8_e5m2)                       |
| Monitoring       | Prometheus + Reflex dashboard        |
| GPU              | 2× NVIDIA A100 PCIe 80GB             |

---

## Prerequisites

- 2× NVIDIA A100 80GB (or equivalent ~160GB VRAM)
- Python 3.10+
- CUDA 12.x
- 50GB+ disk space

### Cloud GPU Options
- [Shadeform](https://shadeform.ai) — 2×A100 ~$3.5/hr
- [Lambda Labs](https://lambdalabs.com)
- [RunPod](https://runpod.io)

---

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/IneshReddy249/vLLM-mixtral-MoE-optimization.git
cd vLLM-mixtral-MoE-optimization

python -m venv venv
source venv/bin/activate
pip install vllm reflex httpx
```

### 2. Start vLLM Server
```bash
python server_optimized.py
```

Server starts at `http://localhost:8000`

Wait for:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 3. Start Dashboard
```bash
cd vllm_dashboard
source ../venv/bin/activate
reflex run
```

Dashboard at `http://localhost:3000`

---

## Server Configuration
```python
python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/mixtral-instruct-awq \
    --quantization awq_marlin \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --kv-cache-dtype fp8_e5m2 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-expert-parallel \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --host 0.0.0.0 \
    --port 8000
```

---

## Real-Time Dashboard

Dashboard pulls live metrics from vLLM's Prometheus endpoint (`/metrics`):

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to first token (ms) |
| **Speed** | Tokens per second |
| **ITL** | Inter-token latency (ms) |
| **Tokens** | Total tokens generated |
| **Latency** | End-to-end request time (s) |

Features: real-time streaming responses, multi-turn conversation support, sticky metrics header, dark theme UI.

---

## Project Structure
```
vLLM-mixtral-MoE-optimization/
├── server_optimized.py
├── README.md
├── requirements.txt
├── vllm_dashboard/
│   ├── rxconfig.py
│   └── vllm_dashboard/
│       ├── __init__.py
│       ├── vllm_dashboard.py
│       ├── state.py
│       └── metrics.py
└── assets/
    └── dashboard.png
```

---

## API Usage

### Chat Completion
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "casperhansen/mixtral-instruct-awq",
    "messages": [{"role": "user", "content": "Explain PagedAttention"}],
    "max_tokens": 512,
    "stream": true
  }'
```

### Metrics Endpoint
```bash
curl http://localhost:8000/metrics
```

---

## Troubleshooting

**First request is slow**  
Normal — first request triggers CUDA graph capture and kernel compilation. Subsequent requests are fast.

**Out of memory**  
Reduce `--gpu-memory-utilization` to 0.85 or lower `--max-num-seqs`.

**Low cache hit rate**  
Prefix caching requires repeated prefixes. Single-turn queries won't benefit. Multi-turn chat shows the 60% hit rate.

**Dashboard not connecting**  
Ensure vLLM server is running on port 8000. Check Reflex backend on port 8002.

---

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [AWQ Quantization](https://github.com/mit-han-lab/llm-awq)
- [Mixtral Paper](https://arxiv.org/abs/2401.04088)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Marlin Kernel](https://github.com/IST-DASLab/marlin)

---

## Related Projects

- [Llama-3.1-8B on H100 — 1,700+ tok/s, 11ms TTFT with TRT-LLM FP8](https://github.com/IneshReddy249/LLAMA-TRT-OPTIMIZATION)
- [Speculative Decoding — 2.26× latency reduction on Qwen 2.5 (dual A100s)](https://github.com/IneshReddy249/SPECULATIVE_DECODING)
- [Qwen2.5-32B on H200 — 3,981 tok/s at 64 concurrent users](https://github.com/IneshReddy249/LLM_INFERENCE_OPTIMIZATION)

---

## Author

**Inesh Reddy Chappidi** — LLM Inference & Systems Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Inesh_Reddy-0077B5?logo=linkedin)](https://www.linkedin.com/in/inesh-reddy)
[![GitHub](https://img.shields.io/badge/GitHub-IneshReddy249-181717?logo=github)](https://github.com/IneshReddy249)
