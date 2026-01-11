import subprocess

if __name__ == "__main__":
    subprocess.run([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "casperhansen/mixtral-instruct-awq",
        "--quantization", "awq_marlin",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "8192",
        "--kv-cache-dtype", "fp8_e5m2",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-expert-parallel",
        "--max-num-seqs", "256",
        "--max-num-batched-tokens", "8192",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
