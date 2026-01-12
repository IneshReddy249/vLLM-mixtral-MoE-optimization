# vLLM Mixtral 8x7B MoE Optimization

[![vLLM](https://img.shields.io/badge/vLLM-0.13.0-blue?logo=python)](https://github.com/vllm-project/vllm)
[![A100](https://img.shields.io/badge/GPU-2×A100_80GB-76B900?logo=nvidia)](https://www.nvidia.com/en-us/data-center/a100/)
[![Mixtral](https://img.shields.io/badge/Model-Mixtral_8x7B-orange)](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

Production-ready MoE inference optimization achieving **120 tok/s** with **21ms TTFT** on dual A100 GPUs with real-time metrics dashboard.

---

## 🚀 Performance Results

| Metric | Value |
|--------|-------|
| **Throughput** | 120 tok/s |
| **TTFT** (Time to First Token) | 21 ms |
| **ITL** (Inter-Token Latency) | 8 ms |
| **Prefix Cache Hit Rate** | 60% |
| **Model Size** | 90GB → 26GB (3.5x reduction) |

---

## ⚡ Optimizations Enabled

| Optimization | Flag | Impact |
|--------------|------|--------|
| **AWQ Marlin Quantization** | `--quantization awq_marlin` | 3.5x memory reduction, 4-bit weights |
| **FP8 KV Cache** | `--kv-cache-dtype fp8_e5m2` | 50% KV cache memory savings |
| **Tensor Parallelism** | `--tensor-parallel-size 2` | Split model across 2 GPUs |
| **Expert Parallelism** | `--enable-expert-parallel` | 4 experts per GPU (8 total) |
| **Prefix Caching** | `--enable-prefix-caching` | Reuse KV for repeated context |
| **Chunked Prefill** | `--enable-chunked-prefill` | Prevent decode stalls |
| **FlashInfer Attention** | Auto-detected | Optimized attention kernels |
| **PagedAttention** | Built-in | Eliminates memory fragmentation |
| **Continuous Batching** | Built-in | Dynamic request scheduling |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Inference Engine | vLLM 0.13.0 |
| Model | casperhansen/mixtral-instruct-awq |
| Quantization | AWQ Marlin 4-bit |
| KV Cache | FP8 (fp8_e5m2) |
| Dashboard | Reflex + Prometheus |
| GPU | 2× NVIDIA A100 80GB |

---

## 📋 Prerequisites

- 2× NVIDIA A100 80GB (or equivalent ~160GB VRAM)
- Python 3.10+
- CUDA 12.x
- 50GB+ disk space

### Cloud GPU Options
- [Shadeform](https://shadeform.ai) - 2×A100 ~$3.5/hr
- [Lambda Labs](https://lambdalabs.com)
- [RunPod](https://runpod.io)

---

## 🏃 Quick Start

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
# New terminal
cd vllm_dashboard
source ../venv/bin/activate
reflex run
```

Dashboard at `http://localhost:3000`

---

## 🖥️ Server Configuration

### server_optimized.py
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

## 📊 Real-Time Dashboard

Dashboard pulls metrics from vLLM's Prometheus endpoint (`/metrics`):

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to first token (ms) |
| **Speed** | Tokens per second |
| **ITL** | Inter-token latency (ms) |
| **Tokens** | Total tokens generated |
| **Latency** | End-to-end request time (s) |

**Features:**
- Real-time streaming responses
- Multi-turn conversation support
- Sticky metrics header
- Dark theme UI

---

## 📁 Project Structure
```
vLLM-mixtral-MoE-optimization/
├── server_optimized.py          # vLLM server config
├── README.md
├── requirements.txt
├── vllm_dashboard/
│   ├── rxconfig.py              # Reflex config
│   └── vllm_dashboard/
│       ├── __init__.py
│       ├── vllm_dashboard.py    # UI components
│       ├── state.py             # Chat state management
│       └── metrics.py           # Prometheus parser
└── assets/
    └── dashboard.png
```

---

## 🔌 API Usage

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

## 📈 What Each Optimization Does

### AWQ Marlin Quantization
Compresses model weights from FP16 to 4-bit. Marlin kernel is optimized for NVIDIA GPUs. Reduces model from 90GB to 26GB.

### FP8 KV Cache
Stores attention key-value cache in 8-bit instead of 16-bit. Cuts KV memory by 50%. Enables 1.9M token cache capacity.

### Expert Parallelism
Mixtral has 8 experts, only 2 activate per token. Expert parallelism distributes experts across GPUs:
- GPU 0: Experts 0, 1, 2, 3
- GPU 1: Experts 4, 5, 6, 7

### Prefix Caching
In multi-turn chat, conversation history repeats. Prefix caching stores KV computations and reuses them. Hit rate reaches 60% after few turns.

### Chunked Prefill
Breaks long prompts into chunks. Interleaves prefill with decode. Prevents long prompts from blocking ongoing generation.

### PagedAttention
Manages KV cache like virtual memory pages. Eliminates fragmentation. Allows more concurrent requests without OOM.

### Continuous Batching
Dynamically adds/removes requests from batch. No waiting for slowest request. Maximizes GPU utilization.

---

## ⚠️ Troubleshooting

### First Request is Slow
Normal. First request triggers CUDA graph capture and kernel compilation. Subsequent requests are fast.

### Out of Memory
Reduce `--gpu-memory-utilization` to 0.85 or lower `--max-num-seqs`.

### Low Cache Hit Rate
Prefix caching needs repeated prefixes. Single-turn queries won't benefit. Multi-turn chat shows improvement.

### Dashboard Not Connecting
Ensure vLLM server is running on port 8000. Check Reflex backend on port 8002.

---

## 🔗 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [AWQ Quantization](https://github.com/mit-han-lab/llm-awq)
- [Mixtral Paper](https://arxiv.org/abs/2401.04088)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Reflex Framework](https://reflex.dev)

---

## 👤 Author

**Inesh Reddy**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-inesh--reddy-blue?logo=linkedin)](https://www.linkedin.com/in/inesh-reddy)
[![GitHub](https://img.shields.io/badge/GitHub-IneshReddy249-black?logo=github)](https://github.com/IneshReddy249)

---

**⭐ Star this repo if it helped you!**
