"""
Astra Full Model Benchmark (Phase 5A)
Benchmarks full forward latency, throughput, and memory consumption.
"""

import time
from typing import Dict, Any, Optional
import torch

from configs.schema import AstraConfig, ModelConfig, GDNConfig, AttentionConfig
from model.reference.astra import AstraForCausalLM


def benchmark_model_forward(
    config: Optional[AstraConfig] = None,
    batch_size: int = 1,
    seq_len: int = 512,
    dtype: torch.dtype = torch.float32,
    num_warmup: int = 2,
    num_runs: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    # Default to small test config for CPU/GPU validation
    if config is None:
        config = AstraConfig(
            model=ModelConfig(
                name="astra-bench",
                vocab_size=1024,
                hidden_size=512,
                intermediate_size=1024,
                num_layers=4,
            ),
            gdn=GDNConfig(num_heads=8, head_dim=64),
            attention=AttentionConfig(num_q_heads=8, num_kv_heads=2, head_dim=64),
        )

    model = AstraForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    input_ids = torch.randint(
        0, config.model.vocab_size, (batch_size, seq_len), device=device
    )

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            model(input_ids=input_ids)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_runs):
            with torch.no_grad():
                model(input_ids=input_ids)
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event) / num_runs
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        start_time = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                model(input_ids=input_ids)
        latency_ms = ((time.perf_counter() - start_time) / num_runs) * 1000.0
        peak_memory_mb = 0.0

    tokens = batch_size * seq_len
    tokens_per_sec = (tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

    return {
        "model_name": config.model.name,
        "num_layers": config.model.num_layers,
        "hidden_size": config.model.hidden_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "latency_ms": latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": peak_memory_mb,
    }


if __name__ == "__main__":
    res = benchmark_model_forward(batch_size=1, seq_len=256)
    print(f"Model: {res['model_name']} | Latency: {res['latency_ms']:.2f} ms | Throughput: {res['tokens_per_sec']:.1f} tok/s")
