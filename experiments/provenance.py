"""
Astra Provenance Utilities (Phase 6)
Extracts verifiable Git and Hardware provenance without fabrication or placeholders.
"""

import platform
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def get_git_commit(repo_path: Optional[str] = None) -> Optional[str]:
    """
    Returns the 40-character SHA-1 of the current HEAD commit.
    Returns None if git is not available or the path is not a valid git repository.
    NEVER returns placeholders like 'unknown' or 'latest'.
    """
    cmd = ["git", "rev-parse", "HEAD"]
    cwd = repo_path or str(Path(__file__).parents[1])
    try:
        commit = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        if re.match(r"^[0-9a-fA-F]{40}$", commit):
            return commit
        return None
    except Exception:
        return None


def get_git_dirty_state(repo_path: Optional[str] = None) -> bool:
    """
    Returns True if the working directory contains uncommitted modifications or staged files.
    """
    cmd = ["git", "status", "--porcelain"]
    cwd = repo_path or str(Path(__file__).parents[1])
    try:
        output = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return len(output) > 0
    except Exception:
        # If git status cannot be checked, consider it dirty/unsafe for fail-closed policy
        return True


def get_hardware_provenance() -> Dict[str, Any]:
    """
    Captures platform, software, and accelerator information.
    """
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None"
    num_gpus = torch.cuda.device_count() if cuda_avail else 0

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "num_gpus": num_gpus,
    }
