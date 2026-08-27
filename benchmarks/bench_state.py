"""
Astra-1B State Memory Footprint Profiler (Phase 5A)
Calculates and verifies exact theoretical and allocated memory for recurrent state tensors:
  H = 16 heads
  D = 96 head_dim
  Per-head state: 96 x 96 = 9,216 elements
  Per-layer state: 16 x 9,216 = 147,456 elements
  18 GDN layers total: 18 x 147,456 = 2,654,208 elements per sample
"""

from typing import Dict, Any, List
import torch


def profile_state_memory(
    num_heads: int = 16,
    head_dim: int = 96,
    num_gdn_layers: int = 18,
    batch_sizes: List[int] = None,
) -> Dict[str, Any]:
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 64, 256]

    elements_per_layer = num_heads * head_dim * head_dim
    total_elements_per_sample = elements_per_layer * num_gdn_layers

    results = {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_gdn_layers": num_gdn_layers,
        "elements_per_layer": elements_per_layer,
        "total_elements_per_sample": total_elements_per_sample,
        "batch_breakdown": [],
    }

    for b in batch_sizes:
        # Theoretical Bytes
        fp32_bytes = b * total_elements_per_sample * 4
        bf16_bytes = b * total_elements_per_sample * 2

        # Actual Tensor Allocation
        tensors_bf16 = [
            torch.zeros(b, num_heads, head_dim, head_dim, dtype=torch.bfloat16)
            for _ in range(num_gdn_layers)
        ]
        allocated_bf16 = sum(t.element_size() * t.nelement() for t in tensors_bf16)

        results["batch_breakdown"].append(
            {
                "batch_size": b,
                "fp32_mb": fp32_bytes / (1024 * 1024),
                "bf16_mb": bf16_bytes / (1024 * 1024),
                "allocated_bf16_mb": allocated_bf16 / (1024 * 1024),
                "bf16_per_sample_mb": (bf16_bytes / b) / (1024 * 1024),
            }
        )

    return results


def print_state_report():
    res = profile_state_memory()
    print("=" * 70)
    print("Astra-1B Recurrent State Memory Profiler")
    print("=" * 70)
    print(f"GDN Configuration: H={res['num_heads']}, D={res['head_dim']}, Layers={res['num_gdn_layers']}")
    print(f"Elements per GDN layer / sample : {res['elements_per_layer']:,}")
    print(f"Total elements (18 layers) / sample: {res['total_elements_per_sample']:,}")
    print("-" * 70)
    print(f"{'Batch Size':<12} | {'FP32 Total (MB)':<18} | {'BF16 Total (MB)':<18} | {'BF16 / Sample'}")
    print("-" * 70)
    for row in res["batch_breakdown"]:
        print(
            f"{row['batch_size']:<12} | "
            f"{row['fp32_mb']:<18.2f} | "
            f"{row['bf16_mb']:<18.2f} | "
            f"{row['bf16_per_sample_mb']:.2f} MB"
        )
    print("=" * 70)


if __name__ == "__main__":
    print_state_report()
