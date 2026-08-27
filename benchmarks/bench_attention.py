"""
Astra Attention Microbenchmark Suite (Phase 5A)
Benchmarks Gated GQA Attention across sequence lengths and batch sizes.
"""

import time
from typing import Dict, Any, List, Optional
import torch

from configs.schema import AttentionConfig
from model.reference.attention import ReferenceGQA


def benchmark_attention_run(
    batch_size: int = 1,
    seq_len: int = 512,
    d_model: int = 2048,
    config: Optional[AttentionConfig] = None,
    dtype: torch.dtype = torch.float32,
    num_warmup: int = 3,
    num_runs: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    config = config or AttentionConfig()
    model = ReferenceGQA(d_model=d_model, config=config).to(device=device, dtype=dtype)
    model.eval()

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            model(x)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()

        start_event.record()
        for _ in range(num_runs):
            with torch.no_grad():
                model(x)
        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event) / num_runs
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        start_time = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                model(x)
        total_time = time.perf_counter() - start_time
        latency_ms = (total_time / num_runs) * 1000.0
        peak_memory_mb = 0.0

    tokens = batch_size * seq_len
    tokens_per_sec = (tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dtype": str(dtype).split(".")[-1],
        "latency_ms": latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": peak_memory_mb,
    }


def run_attention_matrix():
    print("=" * 65)
    print("Astra GQA Attention Benchmark")
    print("=" * 65)
    for b in [1, 2]:
        for t in [128, 512, 2048]:
            res = benchmark_attention_run(
                batch_size=b,
                seq_len=t,
                d_model=512,
                config=AttentionConfig(num_q_heads=8, num_kv_heads=2, head_dim=64),
                num_warmup=1,
                num_runs=3,
            )
            print(
                f"B={res['batch_size']:<2} | T={res['seq_len']:<5} | "
                f"Latency={res['latency_ms']:<8.2f} ms | Throughput={res['tokens_per_sec']:<10.1f} tok/s"
            )


if __name__ == "__main__":
    run_attention_matrix()
