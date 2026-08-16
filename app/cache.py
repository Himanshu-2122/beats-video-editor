import os
import json
import hashlib
import threading
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


class CacheManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._file_hashes = {}
        self._hash_lock = threading.Lock()

    def get_file_hash(self, filepath: str) -> str:
        """Generate hash for file based on path, size, and mtime."""
        with self._hash_lock:
            if filepath in self._file_hashes:
                return self._file_hashes[filepath]
            
            try:
                stat = os.stat(filepath)
                content = f"{filepath}:{stat.st_size}:{stat.st_mtime}"
                hash_val = hashlib.md5(content.encode()).hexdigest()[:16]
                self._file_hashes[filepath] = hash_val
                return hash_val
            except Exception:
                return hashlib.md5(filepath.encode()).hexdigest()[:16]

    def get_cache_path(self, category: str, file_hash: str, suffix: str = "") -> Path:
        """Get cache file path for a category and file hash."""
        category_dir = self.cache_dir / category
        category_dir.mkdir(exist_ok=True)
        name = f"{file_hash}{suffix}.json"
        return category_dir / name

    def load(self, category: str, filepath: str, suffix: str = "") -> Optional[Any]:
        """Load cached data if file hasn't changed."""
        file_hash = self.get_file_hash(filepath)
        cache_path = self.get_cache_path(category, file_hash, suffix)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def save(self, category: str, filepath: str, data: Any, suffix: str = "") -> bool:
        """Save data to cache."""
        file_hash = self.get_file_hash(filepath)
        cache_path = self.get_cache_path(category, file_hash, suffix)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            return True
        except Exception:
            return False

    def clear_category(self, category: str):
        """Clear all cache for a category."""
        category_dir = self.cache_dir / category
        if category_dir.exists():
            for f in category_dir.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass

    def clear_all(self):
        """Clear all cache."""
        for cat in ["beats", "scenes", "motion", "video_analysis", "embeddings"]:
            self.clear_category(cat)


cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    return cache_manager


def cached(category: str, suffix: str = "", key_func: Optional[Callable] = None):
    """Decorator for caching function results based on file hash."""
    def decorator(func: Callable):
        def wrapper(filepath: str, *args, **kwargs):
            mgr = get_cache_manager()
            
            # Try to load from cache
            if key_func:
                cache_key = key_func(filepath, *args, **kwargs)
                cached_data = mgr.load(category, filepath, f"_{cache_key}")
            else:
                cached_data = mgr.load(category, filepath, suffix)
            
            if cached_data is not None:
                return cached_data
            
            # Compute and cache
            result = func(filepath, *args, **kwargs)
            if result is not None:
                if key_func:
                    mgr.save(category, filepath, result, f"_{cache_key}")
                else:
                    mgr.save(category, filepath, result, suffix)
            return result
        
        return wrapper
    return decorator


def parallel_process(items: list, func: Callable, max_workers: int = 4, 
                     progress_callback: Optional[Callable] = None) -> list:
    """Process items in parallel with optional progress callback."""
    results = [None] * len(items)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(func, item): idx for idx, item in enumerate(items)}
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e
            
            if progress_callback:
                progress_callback(len([r for r in results if r is not None]), len(items))
    
    return results


def run_in_background(func: Callable, *args, **kwargs):
    """Run function in background thread, return future."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(func, *args, **kwargs)