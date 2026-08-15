import errno
import gc

import torch


_OUT_OF_MEMORY_MARKERS = (
    "out of memory",
    "cannot allocate memory",
    "can't allocate memory",
    "not enough memory",
    "cudnn_status_alloc_failed",
    "cuda error: memory allocation",
)


def is_out_of_memory_error(error):
    """Return whether an exception represents a host or accelerator OOM."""
    if isinstance(error, MemoryError):
        return True
    if isinstance(error, OSError) and error.errno == errno.ENOMEM:
        return True
    if not isinstance(error, RuntimeError):
        return False

    message = str(error).lower()
    return any(marker in message for marker in _OUT_OF_MEMORY_MARKERS)


def release_memory():
    """Release unreferenced Python and accelerator memory after an OOM."""
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            # A failed CUDA context may reject cleanup calls. The original OOM
            # remains the useful error to report in that case.
            pass

    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except RuntimeError:
            pass
