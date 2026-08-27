import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network:
    SwiGLU(x) = W_down (SiLU(W_gate(x)) * W_up(x))
    """
    def __init__(self, d_model: int = 2048, intermediate_size: int = 3456):
        super().__init__()
        self.w_gate = nn.Linear(d_model, intermediate_size, bias=False)
        self.w_up = nn.Linear(d_model, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)
