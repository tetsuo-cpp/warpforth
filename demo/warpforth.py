"""PyCUDA wrapper for the WarpForth attention kernel.

Uses pycuda.autoprimaryctx to share PyTorch's CUDA context — device pointers
from torch tensors can be passed directly to kernel launches (zero-copy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycuda.autoprimaryctx  # noqa: F401 — activates PyTorch's primary context
import pycuda.driver as cuda


class AttentionKernel:
    """Loads and launches the WarpForth attention kernel.

    Accepts contiguous float32 CUDA tensors and passes device pointers
    directly (zero-copy). Launches one kernel invocation per call.

    The kernel computes: O = softmax(Q @ K^T / sqrt(head_dim)) @ V
    with a causal mask, using f64 shared memory for softmax precision.
    """

    def __init__(self, ptx_path: str | Path) -> None:
        ptx_bytes = Path(ptx_path).read_bytes()
        module = cuda.module_from_buffer(ptx_bytes)
        self._function = module.get_function("attention")

    def __call__(
        self,
        q: object,  # torch.Tensor (seq_len, head_dim) float32 CUDA
        k: object,  # torch.Tensor (seq_len, head_dim) float32 CUDA
        v: object,  # torch.Tensor (seq_len, head_dim) float32 CUDA
        o: object,  # torch.Tensor (seq_len, head_dim) float32 CUDA
        seq_len: int,
        head_dim: int,
    ) -> None:
        self._function(
            np.intp(q.data_ptr()),
            np.intp(k.data_ptr()),
            np.intp(v.data_ptr()),
            np.intp(o.data_ptr()),
            np.int64(seq_len),
            np.int64(head_dim),
            block=(seq_len, 1, 1),
            grid=(seq_len, 1, 1),
        )
