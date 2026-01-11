# vLLM Mixtral 8x7B MoE Optimization

Production-grade LLM inference optimization for Mixtral 8x7B MoE on 2×A100 80GB GPUs with real-time metrics dashboard.

![Dashboard Screenshot](assets/dashboard.png)

## 🚀 Performance Results

| Metric | Value |
|--------|-------|
| **Throughput** | ~120 tok/s |
| **TTFT** | ~21 ms |
| **ITL** | ~8 ms |
| **Model** | Mixtral 8x7B (AWQ Marlin 4-bit) |
| **Hardware** | 2×A100 80GB |

## 🔧 Optimizations Enabled

| Optimization | Flag/Auto | Impact |
|--------------|-----------|--------|
| **AWQ Marlin** | `--quantization awq_marlin` | 3.5x memory reduction |
| **FP8 KV Cache** | `--kv-cache-dtype fp8_e5m2` | 50% KV cache savings |
| **Tensor Parallel** | `--tensor-parallel-size 2` | Model across 2 GPUs |
| **Expert Parallel** | `--enable-expert-parallel` | MoE expert distribution |
| **Prefix Caching** | `--enable-prefix-caching` | Reuse KV for repeated prefixes |
| **Chunked Prefill** | `--enable-chunked-prefill` | Overlap prefill/decode |
| **FlashInfer** | Auto-detected | Optimized attention kernels |

## 📊 Real-Time Metrics Dashboard

Server-side metrics from vLLM Prometheus endpoint:

- **TTFT** - Time to First Token
- **ITL** - Inter-Token Latency
- **Speed** - Tokens per second
- **Latency** - End-to-end request time

Features: Real-time streaming, multi-turn chat, sticky metrics header.

## 🚀 Quick Start
```bash
git clone https://github.com/IneshReddy249/vLLM-mixtral-MoE-optimization.git
cd vLLM-mixtral-MoE-optimization

python -m venv venv
source venv/bin/activate
pip install vllm reflex httpx

python server_optimized.py

# New terminal
cd vllm_dashboard && reflex run
```

Dashboard at `http://localhost:3000`

## 📁 Project Structure
```
├── server_optimized.py
├── vllm_dashboard/
│   └── vllm_dashboard/
│       ├── vllm_dashboard.py
│       ├── state.py
│       └── metrics.py
└── README.md
```

## 📄 License

MIT
