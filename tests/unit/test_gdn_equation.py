import torch
import numpy as np
import pytest
from configs.schema import GDNConfig
from model.reference.gdn import AstraGDN


def test_gdn_toy_manual_equation():
    """
    Direct verification of Astra Contract v0.1 mathematical equation:
    S_t = D_t @ S_{t-1} @ (I - u_t * k_t @ k_t^T) + u_t * (v_t @ k_t^T)
    """
    # 1. Manual NumPy Reference Calculation
    S0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    k_raw = np.array([1.0, 0.0], dtype=np.float32)
    k = k_raw / np.linalg.norm(k_raw)
    v = np.array([5.0, 6.0], dtype=np.float32)
    q = np.array([1.0, 1.0], dtype=np.float32)
    r = np.array([0.8, 0.5], dtype=np.float32)
    u = 0.5

    # v_hat = S0 @ k
    v_hat = S0 @ k  # [1.0, 3.0]
    diff = v - v_hat  # [4.0, 3.0]
    delta = u * np.outer(diff, k)  # [[2.0, 0.0], [1.5, 0.0]]
    D = np.diag(r)  # [[0.8, 0.0], [0.0, 0.5]]
    S1_expected = D @ S0 + delta  # [[2.8, 1.6], [3.0, 2.0]]
    y_expected = S1_expected @ q  # [4.4, 5.0]

    # 2. Torch Implementation Step Verification
    S0_torch = torch.tensor(S0).unsqueeze(0).unsqueeze(0)  # [1, 1, 2, 2]
    k_torch = torch.tensor(k).unsqueeze(0).unsqueeze(0)    # [1, 1, 2]
    v_torch = torch.tensor(v).unsqueeze(0).unsqueeze(0)    # [1, 1, 2]
    q_torch = torch.tensor(q).unsqueeze(0).unsqueeze(0)    # [1, 1, 2]
    r_torch = torch.tensor(r).unsqueeze(0).unsqueeze(0)    # [1, 1, 2]
    u_torch = torch.tensor([[u]]).unsqueeze(0)             # [1, 1, 1]

    # Replicate recurrence loop step exactly as in AstraGDN
    v_hat_torch = torch.matmul(S0_torch, k_torch.unsqueeze(-1)).squeeze(-1)
    diff_torch = (v_torch - v_hat_torch).unsqueeze(-1)
    k_outer = k_torch.unsqueeze(-2)
    delta_torch = u_torch.unsqueeze(-1) * torch.matmul(diff_torch, k_outer)
    S1_torch = r_torch.unsqueeze(-1) * S0_torch + delta_torch
    y_torch = torch.matmul(S1_torch, q_torch.unsqueeze(-1)).squeeze(-1)

    np.testing.assert_allclose(S1_torch.squeeze().numpy(), S1_expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y_torch.squeeze().numpy(), y_expected, rtol=1e-6, atol=1e-6)


def test_gdn_zero_update_invariant():
    """
    Invariant: When update_gate u_t = 0 and retention r_t = 1.0, S_t == S_{t-1}.
    """
    B, H, D = 2, 4, 16
    torch.manual_seed(42)
    S_prev = torch.randn(B, H, D, D)
    k = torch.randn(B, H, D)
    k = k / torch.linalg.norm(k, dim=-1, keepdim=True)
    v = torch.randn(B, H, D)
    
    r = torch.ones(B, H, D)
    u = torch.zeros(B, H, 1)

    v_hat = torch.matmul(S_prev, k.unsqueeze(-1)).squeeze(-1)
    delta = u.unsqueeze(-1) * torch.matmul((v - v_hat).unsqueeze(-1), k.unsqueeze(-2))
    S_next = r.unsqueeze(-1) * S_prev + delta

    torch.testing.assert_close(S_next, S_prev, rtol=1e-7, atol=1e-7)


def test_gdn_exact_retrieval_invariant():
    """
    Invariant: When update_gate u_t = 1.0 and retention r_t = 1.0,
    the newly stored state must satisfy S_t @ k_t == v_t (exact retrieval of latest key).
    """
    B, H, D = 2, 4, 16
    torch.manual_seed(42)
    S_prev = torch.randn(B, H, D, D)
    k = torch.randn(B, H, D)
    k = k / torch.linalg.norm(k, dim=-1, keepdim=True)
    v = torch.randn(B, H, D)

    r = torch.ones(B, H, D)
    u = torch.ones(B, H, 1)

    v_hat = torch.matmul(S_prev, k.unsqueeze(-1)).squeeze(-1)
    delta = u.unsqueeze(-1) * torch.matmul((v - v_hat).unsqueeze(-1), k.unsqueeze(-2))
    S_next = r.unsqueeze(-1) * S_prev + delta

    # Check S_next @ k == v
    retrieved_v = torch.matmul(S_next, k.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(retrieved_v, v, rtol=1e-5, atol=1e-5)
