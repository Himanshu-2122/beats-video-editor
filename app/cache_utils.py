import os
import json
import hashlib
import numpy as np
from pathlib import Path

def _hash_key(*parts: str) -> str:
    """Create a short deterministic hash from given string parts.
    Used for cache file naming.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
    return h.hexdigest()[:12]

def load_or_compute(
    cache_root: str,
    key_parts: list[str],
    compute_fn,
    *,
    description: str = "cached result",
) -> np.ndarray:
    """Load a cached numpy array or compute and store it.

    Args:
        cache_root: Directory where caches are stored (absolute).
        key_parts: List of strings that uniquely identify the computation.
        compute_fn: Callable with no args that returns a numpy array.
        description: Human readable description (used only for logging).
    Returns:
        np.ndarray containing the result.
    """
    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (f"{_hash_key(*key_parts)}.npz")
    if cache_file.is_file():
        try:
            data = np.load(cache_file, allow_pickle=True)
            return data["arr"]
        except Exception:
            # Corrupted cache – fall back to recompute
            pass
    # Compute and cache
    arr = compute_fn()
    np.savez(cache_file, arr=arr)
    print(f"[CACHE] {description} saved to {cache_file}")
    return arr
