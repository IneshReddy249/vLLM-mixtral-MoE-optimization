import asyncio
import aiohttp
import time
import json
import sys

BASE_URL = "http://localhost:8000/v1/chat/completions"

PROMPTS = [
    ("short", "What is machine learning?"),
    ("medium", "Explain supervised vs unsupervised learning with examples."),
    ("long", "Explain how transformer networks work including attention mechanism and positional encoding.")
]

BATCH_SIZES = [1, 4, 8]  # Concurrent requests

async def benchmark_request(session, prompt, max_tokens):
    payload = {
        "model": "casperhansen/mixtral-instruct-awq",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    start = time.perf_counter()
    ttft = None
    tokens = 0
    
    async with session.post(BASE_URL, json=payload) as resp:
        async for line in resp.content:
            text = line.decode().strip()
            if not text.startswith("data: "):
                continue
            data = text[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if ttft is None and chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                ttft = time.perf_counter() - start
            if "usage" in chunk:
                tokens = chunk["usage"].get("completion_tokens", 0)
    
    latency = time.perf_counter() - start
    tps = tokens / latency if latency > 0 else 0
    return {"ttft": ttft, "latency": latency, "tps": tps, "tokens": tokens}

async def run_batch(session, prompt, max_tokens, batch_size):
    """Run batch_size requests concurrently"""
    tasks = [benchmark_request(session, prompt, max_tokens) for _ in range(batch_size)]
    return await asyncio.gather(*tasks)

async def run_benchmark(output_file, config_name):
    results = {"config": config_name, "runs": []}
    
    async with aiohttp.ClientSession() as session:
        print(f"\n=== {config_name} Benchmark ===\n")
        
        # Warmup
        print("Warmup (3 runs)...")
        for i in range(3):
            await benchmark_request(session, "Explain what AI is.", 64)
            print(f"  Warmup {i+1}/3 done")
        print("Warmup complete.\n")
        
        for prompt_name, prompt in PROMPTS:
            for max_tokens in [128, 256]:
                for batch_size in BATCH_SIZES:
                    print(f"{prompt_name} | {max_tokens} tokens | batch={batch_size}:")
                    
                    # Run 3 batches
                    all_runs = []
                    for i in range(3):
                        batch_results = await run_batch(session, prompt, max_tokens, batch_size)
                        all_runs.extend(batch_results)
                        
                    # Calculate averages
                    avg = {
                        "prompt": prompt_name,
                        "max_tokens": max_tokens,
                        "batch_size": batch_size,
                        "ttft_ms": sum(r["ttft"] for r in all_runs) / len(all_runs) * 1000,
                        "latency_s": sum(r["latency"] for r in all_runs) / len(all_runs),
                        "tps": sum(r["tps"] for r in all_runs) / len(all_runs),
                        "tokens_avg": sum(r["tokens"] for r in all_runs) / len(all_runs),
                        "throughput": sum(r["tokens"] for r in all_runs) / max(r["latency"] for r in all_runs)
                    }
                    results["runs"].append(avg)
                    print(f"  TTFT={avg['ttft_ms']:.0f}ms | Latency={avg['latency_s']:.2f}s | TPS={avg['tps']:.1f} | Throughput={avg['throughput']:.1f} tok/s\n")
    
    # Calculate overall averages
    all_runs = results["runs"]
    overall = {
        "avg_ttft_ms": sum(r["ttft_ms"] for r in all_runs) / len(all_runs),
        "avg_latency_s": sum(r["latency_s"] for r in all_runs) / len(all_runs),
        "avg_tps": sum(r["tps"] for r in all_runs) / len(all_runs),
        "avg_tokens": sum(r["tokens_avg"] for r in all_runs) / len(all_runs),
        "avg_throughput": sum(r["throughput"] for r in all_runs) / len(all_runs)
    }
    results["overall"] = overall
    
    print(f"\n{'='*50}")
    print(f"OVERALL AVERAGES - {config_name}")
    print(f"{'='*50}")
    print(f"  Avg TTFT:       {overall['avg_ttft_ms']:.1f} ms")
    print(f"  Avg Latency:    {overall['avg_latency_s']:.2f} s")
    print(f"  Avg TPS:        {overall['avg_tps']:.1f} tokens/sec")
    print(f"  Avg Tokens:     {overall['avg_tokens']:.1f}")
    print(f"  Avg Throughput: {overall['avg_throughput']:.1f} tok/s")
    print(f"{'='*50}\n")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python benchmark_suite.py results/baseline.json Baseline")
        sys.exit(1)
    asyncio.run(run_benchmark(sys.argv[1], sys.argv[2]))