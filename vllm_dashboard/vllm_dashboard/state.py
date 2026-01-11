import reflex as rx
import asyncio
import time
from .metrics import fetch_vllm_metrics, stream_chat


class Message(rx.Base):
    role: str
    content: str


class State(rx.State):
    messages: list[Message] = []
    
    itl_avg: float = 0.0
    last_ttft: float = 0.0
    last_latency: float = 0.0
    last_tokens: int = 0
    last_tps: float = 0.0
    
    input_text: str = ""
    is_generating: bool = False

    def set_input_text(self, value: str):
        self.input_text = value

    async def generate(self):
        if not self.input_text.strip():
            return
        
        # Add user message to history
        user_msg = Message(role="user", content=self.input_text)
        self.messages = self.messages + [user_msg]
        self.input_text = ""
        
        self.is_generating = True
        self.last_ttft = 0.0
        self.last_latency = 0.0
        self.last_tokens = 0
        self.last_tps = 0.0
        self.itl_avg = 0.0
        
        # Add empty assistant message
        assistant_msg = Message(role="assistant", content="")
        self.messages = self.messages + [assistant_msg]
        yield
        
        before = await fetch_vllm_metrics()
        start_time = time.perf_counter()
        token_count = 0
        response_text = ""
        
        # Build messages for API (full history)
        api_messages = [{"role": m.role, "content": m.content} for m in self.messages[:-1]]
        
        async for chunk in stream_chat(api_messages, max_tokens=2048):
            response_text += chunk
            token_count += 1
            elapsed = time.perf_counter() - start_time
            
            # Update last message
            self.messages[-1] = Message(role="assistant", content=response_text)
            
            self.last_tokens = token_count
            self.last_latency = round(elapsed * 1000, 0)
            if token_count == 1:
                self.last_ttft = round(elapsed * 1000, 0)
            if elapsed > 0:
                self.last_tps = round(token_count / elapsed, 1)
            yield
        
        await asyncio.sleep(0.3)
        after = await fetch_vllm_metrics()
        
        if before and after and "e2e_count" in before and "e2e_count" in after:
            ttft_count_diff = after["ttft_count"] - before["ttft_count"]
            if ttft_count_diff > 0:
                ttft_sum_diff = after["ttft_sum"] - before["ttft_sum"]
                self.last_ttft = round((ttft_sum_diff / ttft_count_diff) * 1000, 0)
            
            e2e_count_diff = after["e2e_count"] - before["e2e_count"]
            if e2e_count_diff > 0:
                e2e_sum_diff = after["e2e_sum"] - before["e2e_sum"]
                self.last_latency = round((e2e_sum_diff / e2e_count_diff) * 1000, 0)
            
            tokens_diff = after["total_tokens"] - before["total_tokens"]
            if tokens_diff > 0:
                self.last_tokens = int(tokens_diff)
            
            itl_count_diff = after["itl_count"] - before["itl_count"]
            if itl_count_diff > 0:
                itl_sum_diff = after["itl_sum"] - before["itl_sum"]
                self.itl_avg = round((itl_sum_diff / itl_count_diff) * 1000, 1)
            
            if self.last_latency > 0:
                self.last_tps = round(self.last_tokens / (self.last_latency / 1000), 1)
        
        self.is_generating = False
        yield

    def clear_chat(self):
        self.messages = []
