import os
import json
import numpy as np
from pathlib import Path

# Lazy import for CLIP model (placeholder implementation)
def _load_clip_model():
    """Load the CLIP model lazily.
    Returns a tuple (model, preprocess) where model has a .encode_image method.
    In this placeholder we return None, as actual model loading is resource heavy.
    """
    try:
        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device
    except Exception as e:
        print(f"[SEMANTIC] CLIP model load failed: {e}, using dummy embeddings.")
        return None, None, None

def _dummy_embedding(dim=512):
    """Return a deterministic random-like embedding for placeholder purposes."""
    rng = np.random.default_rng(42)
    vec = rng.random(dim)
    return vec / np.linalg.norm(vec)

def audio_beat_embeddings(audio_path: str, beats: list[dict], cache_root: str) -> np.ndarray:
    """Compute (or load cached) embeddings for each beat.

    Placeholder implementation returns dummy normalized vectors.
    Real implementation would extract audio segment around each beat and encode with CLIP.
    """
    from app.cache_utils import load_or_compute
    key = ["audio_beat_emb", os.path.abspath(audio_path), str(len(beats))]
    def compute():
        # Generate dummy embeddings per beat
        dim = 512
        vecs = np.stack([_dummy_embedding(dim) for _ in beats])
        return vecs
    return load_or_compute(cache_root, key, compute, description="Audio beat embeddings")

def video_frame_embeddings(video_path: str, fps: float = 1.0, cache_root: str = None) -> np.ndarray:
    """Compute (or load cached) embeddings for video frames at given fps.

    Placeholder returns a single dummy embedding for the whole video.
    Real implementation would extract frames via ffmpeg and encode with CLIP.
    """
    from app.cache_utils import load_or_compute
    key = ["video_frame_emb", os.path.abspath(video_path), f"fps{fps}"]
    def compute():
        dim = 512
        # For placeholder, just one embedding per video
        return np.expand_dims(_dummy_embedding(dim), axis=0)
    return load_or_compute(cache_root, key, compute, description="Video frame embeddings")

def semantic_similarity_matrix(beat_embs: np.ndarray, video_embs: dict[str, np.ndarray]) -> np.ndarray:
    """Compute cosine similarity matrix between beats and videos.

    beat_embs: shape (B, D)
    video_embs: dict mapping video_path -> array shape (1, D) (placeholder).
    Returns matrix of shape (B, V) where V = number of videos.
    """
    beat_norm = beat_embs / np.linalg.norm(beat_embs, axis=1, keepdims=True)
    # Stack video embeddings
    vids = list(video_embs.keys())
    vid_arr = np.stack([video_embs[v][0] for v in vids])
    vid_norm = vid_arr / np.linalg.norm(vid_arr, axis=1, keepdims=True)
    sim = beat_norm @ vid_norm.T  # (B, V)
    return sim

def semantic_bonus(beat_idx: int, video_path: str, sim_matrix: np.ndarray, video_list: list[str], weight: float) -> float:
    """Return weighted semantic bonus for a given beat and video.
    weight is the user-configurable scalar (0..1).
    """
    if weight == 0.0:
        return 0.0
    try:
        v_idx = video_list.index(video_path)
        sim = sim_matrix[beat_idx, v_idx]
        return weight * sim
    except ValueError:
        return 0.0

# End of semantic_match module
