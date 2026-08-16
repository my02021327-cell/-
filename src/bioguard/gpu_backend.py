"""
연산 백엔드 — PROMPT §12: NVIDIA GPU(CuPy / PyTorch CUDA) 자동 감지, 없으면 NumPy 폴백.

원본 `docs/prior_reports/frontend_lit_original.py` 가 `from gpu_backend import BK` 로
참조하던 모듈이다. 인터페이스(`BK.info()`, `BK.conv_causal`)를 그대로 유지한다.
"""

from __future__ import annotations

import numpy as np


class _Backend:
    """가능하면 GPU, 아니면 NumPy. 결과는 항상 numpy 배열로 돌려준다."""

    def __init__(self) -> None:
        self.kind, self._xp, self._torch = "numpy", np, None
        try:
            import cupy  # type: ignore

            if cupy.cuda.runtime.getDeviceCount() > 0:
                self.kind, self._xp = "cupy", cupy
                return
        except Exception:
            pass
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                self.kind, self._torch = "torch", torch
        except Exception:
            pass

    def info(self) -> str:
        if self.kind == "cupy":
            return "CuPy — NVIDIA GPU"
        if self.kind == "torch":
            return f"PyTorch CUDA — {self._torch.cuda.get_device_name(0)}"
        return "NumPy — CPU (NVIDIA GPU 미탐지 — NumPy 폴백)"

    def conv_causal(self, x, kernel, n: int | None = None) -> np.ndarray:
        """인과 합성곱: y[i] = Σ_τ kernel[τ]·x[i−τ]. 길이는 len(x) 로 자른다."""
        x = np.nan_to_num(np.asarray(x, dtype=float))
        kernel = np.asarray(kernel, dtype=float)
        n = len(x) if n is None else n
        if self.kind == "cupy":
            xp = self._xp
            out = xp.convolve(xp.asarray(x), xp.asarray(kernel))[:n]
            return xp.asnumpy(out)
        if self.kind == "torch":
            t = self._torch
            xt = t.tensor(x, dtype=t.float64, device="cuda").view(1, 1, -1)
            kt = t.tensor(kernel[::-1].copy(), dtype=t.float64, device="cuda").view(1, 1, -1)
            pad = len(kernel) - 1
            out = t.nn.functional.conv1d(t.nn.functional.pad(xt, (pad, 0)), kt)
            return out.view(-1).cpu().numpy()[:n]
        return np.convolve(x, kernel)[:n]


BK = _Backend()
