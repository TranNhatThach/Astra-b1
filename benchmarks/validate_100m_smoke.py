"""
Astra-100M Real-Data Training Smoke & Checkpoint/Resume Validation (Phase 7B)
Validates:
  Dataset -> DataLoader -> Batch -> Astra-100M Forward -> Loss -> Backward ->
  Optimizer Step -> Checkpoint Save -> Terminate -> Resume -> Segment B -> Throughput & Sanity.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.schema import AstraConfig
from model.reference.astra import AstraForCausalLM
from data.dataset import AstraOfflineDataset


def run_100m_training_smoke_test(
    shards_dir: str = "data/shards",
    config_path: str = "configs/astra_100m.yaml",
    checkpoint_dir: str = "checkpoints",
    batch_size: int = 2,
    device: str = "cpu",
) -> Dict[str, Any]:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "smoke_step_5.pt"

    # 1. Load Dataset & DataLoader
    dataset = AstraOfflineDataset(shards_dir=shards_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_iter = iter(dataloader)

    # 2. Load Config & Model
    cfg = AstraConfig.from_yaml(config_path)
    model = AstraForCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)

    # SEGMENT A: Steps 1 to 5
    segment_a_losses = []
    step_times = []
    max_test_seq_len = min(512, cfg.model.max_position_embeddings)
    tokens_per_step = batch_size * max_test_seq_len

    t0_start = time.perf_counter()
    for step in range(1, 6):
        t_step_start = time.perf_counter()
        batch = next(data_iter)

        input_ids = batch["input_ids"][:batch_size, :max_test_seq_len].to(device).long()
        doc_ids = batch["doc_ids"][:batch_size, :max_test_seq_len].to(device).long()

        # Forward with loss calculation
        out = model(input_ids, doc_ids=doc_ids, compute_loss=True)
        total_loss = out["loss"]

        # Assert finite
        assert torch.isfinite(total_loss), f"Loss is non-finite at step {step}: {total_loss.item()}"
        segment_a_losses.append(total_loss.item())

        # Backward
        optimizer.zero_grad()
        total_loss.backward()

        # Check gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        assert torch.isfinite(grad_norm), f"Gradient norm is non-finite at step {step}"

        optimizer.step()
        t_step = time.perf_counter() - t_step_start
        step_times.append(t_step)

    # 3. Save Checkpoint at Step 5
    ckpt_payload = {
        "step": 5,
        "tokens_seen": 5 * tokens_per_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": segment_a_losses[-1],
        "config": cfg.to_dict(),
    }
    torch.save(ckpt_payload, ckpt_file)

    # 4. SEGMENT B: Terminate, Re-initialize, and Resume
    del model
    del optimizer

    model_resumed = AstraForCausalLM(cfg).to(device)
    optimizer_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)

    loaded_ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    model_resumed.load_state_dict(loaded_ckpt["model_state_dict"])
    optimizer_resumed.load_state_dict(loaded_ckpt["optimizer_state_dict"])
    start_step = loaded_ckpt["step"]

    segment_b_losses = []
    for step in range(start_step + 1, start_step + 6):
        t_step_start = time.perf_counter()
        batch = next(data_iter)

        input_ids = batch["input_ids"][:batch_size, :max_test_seq_len].to(device).long()
        doc_ids = batch["doc_ids"][:batch_size, :max_test_seq_len].to(device).long()

        out = model_resumed(input_ids, doc_ids=doc_ids, compute_loss=True)
        total_loss = out["loss"]
        assert torch.isfinite(total_loss)
        segment_b_losses.append(total_loss.item())

        optimizer_resumed.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_resumed.parameters(), max_norm=1.0)
        optimizer_resumed.step()

        t_step = time.perf_counter() - t_step_start
        step_times.append(t_step)

    avg_step_time = sum(step_times) / len(step_times)
    tokens_per_sec = tokens_per_step / max(avg_step_time, 1e-5)

    summary = {
        "status": "PASS",
        "steps_completed": 10,
        "segment_a_losses": segment_a_losses,
        "segment_b_losses": segment_b_losses,
        "checkpoint_path": str(ckpt_file),
        "avg_step_time_sec": round(avg_step_time, 4),
        "tokens_per_second": round(tokens_per_sec, 2),
        "finite_losses": True,
        "no_nan_or_inf": True,
        "resume_success": True,
    }

    print(f"[OK] Training Smoke Test PASS: 10 steps completed, avg {avg_step_time:.3f}s/step ({tokens_per_sec:.1f} tokens/s)")
    return summary


if __name__ == "__main__":
    run_100m_training_smoke_test()
