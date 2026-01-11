import httpx
import re
import json

VLLM_URL = "http://localhost:8000"
MODEL_ID = "casperhansen/mixtral-instruct-awq"


def parse_metric(text: str, metric_name: str) -> float:
    pattern = rf'vllm:{metric_name}\{{[^}}]*\}} ([\d.e+-]+)'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return 0.0


async def fetch_vllm_metrics() -> dict:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{VLLM_URL}/metrics", timeout=5)
            text = resp.text
            
            return {
                "ttft_sum": parse_metric(text, "time_to_first_token_seconds_sum"),
                "ttft_count": parse_metric(text, "time_to_first_token_seconds_count"),
                "itl_sum": parse_metric(text, "inter_token_latency_seconds_sum"),
                "itl_count": parse_metric(text, "inter_token_latency_seconds_count"),
                "e2e_sum": parse_metric(text, "e2e_request_latency_seconds_sum"),
                "e2e_count": parse_metric(text, "e2e_request_latency_seconds_count"),
                "total_tokens": int(parse_metric(text, "generation_tokens_total")),
            }
    except Exception as e:
        print(f"Metrics error: {e}")
        return {}


async def stream_chat(messages: list, max_tokens: int = 1024):
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": True,
            }
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    yield content
