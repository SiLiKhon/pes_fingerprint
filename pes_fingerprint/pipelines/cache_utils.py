import os
from typing import Optional

import joblib


CACHE_ENV_VAR = "PFP_CACHE"
DEFAULT_CACHE_PATH = "./cache"

def setup_cache(
    cache_path: Optional[str] = None,
) -> joblib.Memory:
    if cache_path is None:
        cache_path = os.getenv(CACHE_ENV_VAR, DEFAULT_CACHE_PATH)
        print(f"PES fingerprint cache path set to \"{cache_path}\". To change, set {CACHE_ENV_VAR} environment variable.")

    return joblib.Memory(cache_path)
