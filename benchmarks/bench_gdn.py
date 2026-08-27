"""
Astra-GDN Microbenchmark Suite (Phase 5A)
Measures latency (ms), tokens/sec, and memory footprint across:
  - Batch sizes: 1, 2, 4, 8
  - Sequence lengths: 128, 512, 2048, 4096 (plus awkward tails: 257, 4097, 4225)
  - Execution modes: full, chunk-1, chunk-32, chunk-128, chunk-512
  - Dtypes: FP32, BF16
"""

import time
from typing import Dict, Any, List, Optional
import torch

from configs.schema import GDNConfig
from model.reference.gdn import AstraGDN


def benchmark_gdn_run(
    batch_size: int = 1,
    seq_len: int = 512,
    d_model: int = 2048,
    config: Optional[GDNConfig] = None,
    dtype: torch.dtype = torch.float32,
    mode: str = "full",
    chunk_size: int = 128,
    num_warmup: int = 3,
    num_runs: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    config = config or GDNConfig()
    model = AstraGDN(d_model=d_model, config=config).to(device=device, dtype=dtype)
    model.eval()

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

    def run_forward():
        if mode == "full":
            return model(x)
        else:
            c_size = 1 if mode == "chunk-1" else chunk_size
            curr_state = None
            outputs = []
            num_chunks = (seq_len + c_size - 1) // c_size
            for i in range(num_chunks):
                start_idx = i * c_size
                end_idx = min(start_idx + c_size, seq_len)
                x_c = x[:, start_idx:end_idx]
                y_c, curr_state = model(x_c, state=curr_state)
                outputs.append(y_c)
            return torch.cat(outputs, dim=1), curr_state

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            run_forward()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()

        start_event.record()
        for _ in range(num_runs):
            with torch.no_grad():
                run_forward()
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        latency_ms = total_ms / num_runs
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        start_time = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                run_forward()
        total_time = time.perf_counter() - start_time
        latency_ms = (total_time / num_runs) * 1000.0
        peak_memory_mb = 0.0

    tokens = batch_size * seq_len
    tokens_per_sec = (tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "mode": mode,
        "dtype": str(dtype).split(".")[-1],
        "latency_ms": latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": peak_memory_mb,
    }


def run_gdn_matrix(
    batches: List[int] = (1, 2, 4),
    seq_lens: List[int] = (128, 512, 2048),
    modes: List[str] = ("full", "chunk-128"),
    dtypes: List[torch.dtype] = (torch.float32,),
) -> List[Dict[str, Any]]:
    results = []
    for b in batches:
        for t in seq_lens:
            for m in modes:
                for dt in dtypes:
                    c_size = 128 if m == "chunk-128" else 32
                    res = benchmark_gdn_run(
                        batch_size=b,
                        seq_len=t,
                        d_model=512,
                        config=GDNConfig(num_heads=8, head_dim=64),
                        dtype=dt,
                        mode=m,
                        chunk_size=c_size,
                        num_warmup=1,
                        num_runs=3,
                    )
                    results.append(res)
                    print(
                        f"B={res['batch_size']:<2} | T={res['seq_len']:<5} | Mode={res['mode']:<9} | "
                        f"Latency={res['latency_ms']:<8.2f} ms | Throughput={res['tokens_per_sec']:<10.1f} tok/s"
                    )
    return results


if __name__ == "__main__":
    print("=" * 75)
    print("Astra-GDN Performance Benchmark Suite")
    print("=" * 75)
    run_gdn_matrix()
