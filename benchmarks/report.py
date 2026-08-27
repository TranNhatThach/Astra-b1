"""
Astra Benchmark Report Generator (Phase 5A)
Runs benchmarks and exports structured JSON and Markdown reports.
"""

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import torch

from .bench_state import profile_state_memory
from .bench_gdn import run_gdn_matrix
from .bench_model import benchmark_model_forward


def generate_benchmark_report(output_dir: str = "benchmarks") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Git commit hash
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown"

    timestamp = datetime.now().isoformat()
    system_info = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    state_data = profile_state_memory()
    gdn_data = run_gdn_matrix(batches=[1, 2], seq_lens=[128, 512], modes=["full", "chunk-128"])
    model_data = benchmark_model_forward(batch_size=1, seq_len=256)

    report_payload = {
        "system": system_info,
        "state_footprint": state_data,
        "gdn_microbenchmark": gdn_data,
        "model_benchmark": model_data,
    }

    # 1. Export JSON
    json_file = out_path / "benchmark.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # 2. Export Markdown
    md_file = out_path / "benchmark.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Astra-1B Performance & State Memory Benchmark Report\n\n")
        f.write(f"- **Timestamp:** {timestamp}\n")
        f.write(f"- **Git Commit:** `{git_commit}`\n")
        f.write(f"- **Hardware/Device:** {system_info['device_name']}\n")
        f.write(f"- **PyTorch:** `{torch.__version__}` | **Python:** `{platform.python_version()}`\n\n")

        f.write("## 1. Recurrent State Memory Footprint (18 GDN Layers)\n\n")
        f.write("| Batch Size | FP32 Total (MB) | BF16 Total (MB) | BF16 / Sample |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for row in state_data["batch_breakdown"]:
            f.write(f"| {row['batch_size']} | {row['fp32_mb']:.2f} MB | {row['bf16_mb']:.2f} MB | {row['bf16_per_sample_mb']:.2f} MB |\n")

        f.write("\n## 2. GDN Microbenchmark\n\n")
        f.write("| Batch | Sequence | Mode | Latency (ms) | Throughput (tok/s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for row in gdn_data:
            f.write(f"| {row['batch_size']} | {row['seq_len']} | {row['mode']} | {row['latency_ms']:.2f} | {row['tokens_per_sec']:.1f} |\n")

        f.write(f"\n## 3. Full Model Forward Benchmark\n\n")
        f.write(f"- **Model:** `{model_data['model_name']}` (Layers: {model_data['num_layers']}, Hidden: {model_data['hidden_size']})\n")
        f.write(f"- **Batch=1, Seq=256 Latency:** {model_data['latency_ms']:.2f} ms\n")
        f.write(f"- **Throughput:** {model_data['tokens_per_sec']:.1f} tokens/sec\n")

    print(f"\n[OK] Benchmark reports exported to {json_file} and {md_file}")


if __name__ == "__main__":
    generate_benchmark_report()
