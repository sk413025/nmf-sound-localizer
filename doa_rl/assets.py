from typing import Tuple
import torch


def load_H(tf_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load directional transfer functions H and angles from .pth.

    Returns:
        H: (F, D) float32
        angles: (D,) float32 degrees
    """
    data = torch.load(tf_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "H" in data:
            H = data["H"].float()
        elif "H_linear" in data:
            H = data["H_linear"].float()
        else:
            H = torch.as_tensor(data).float()
        if "angles" in data:
            angles = data["angles"]
            if not torch.is_tensor(angles):
                angles = torch.tensor(angles, dtype=torch.float32)
        else:
            angles = torch.linspace(0, 360, H.shape[1] + 1)[:-1]
    else:
        H = torch.as_tensor(data).float()
        angles = torch.linspace(0, 360, H.shape[1] + 1)[:-1]
    return H, angles.float()


def load_W(w_path: str) -> torch.Tensor:
    """Load USM dictionary W (F, K) from .pth without conversion."""
    data = torch.load(w_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "W" in data:
        return data["W"].float()
    return torch.as_tensor(data).float()

