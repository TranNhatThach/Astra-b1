# Astra-1B Performance & State Memory Benchmark Report

- **Timestamp:** 2026-08-27T11:20:07.966384
- **Git Commit:** `7faaad2ee6c683a5f80ecab40f0e14635b3c25a4`
- **Hardware/Device:** NVIDIA GeForce RTX 4060 Laptop GPU
- **PyTorch:** `2.6.0+cu124` | **Python:** `3.13.3`

## 1. Recurrent State Memory Footprint (18 GDN Layers)

| Batch Size | FP32 Total (MB) | BF16 Total (MB) | BF16 / Sample |
| :--- | :--- | :--- | :--- |
| 1 | 10.12 MB | 5.06 MB | 5.06 MB |
| 4 | 40.50 MB | 20.25 MB | 5.06 MB |
| 16 | 162.00 MB | 81.00 MB | 5.06 MB |
| 64 | 648.00 MB | 324.00 MB | 5.06 MB |
| 256 | 2592.00 MB | 1296.00 MB | 5.06 MB |

## 2. GDN Microbenchmark

| Batch | Sequence | Mode | Latency (ms) | Throughput (tok/s) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 128 | full | 82.00 | 1561.0 |
| 1 | 128 | chunk-128 | 78.56 | 1629.4 |
| 1 | 512 | full | 292.27 | 1751.8 |
| 1 | 512 | chunk-128 | 293.69 | 1743.3 |
| 2 | 128 | full | 77.73 | 3293.3 |
| 2 | 128 | chunk-128 | 76.33 | 3353.9 |
| 2 | 512 | full | 315.25 | 3248.2 |
| 2 | 512 | chunk-128 | 317.90 | 3221.1 |

## 3. Full Model Forward Benchmark

- **Model:** `astra-bench` (Layers: 4, Hidden: 512)
- **Batch=1, Seq=256 Latency:** 447.44 ms
- **Throughput:** 572.1 tokens/sec
