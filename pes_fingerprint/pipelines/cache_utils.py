import os
from typing import Dict, List, Optional, Union
from functools import lru_cache

import joblib
from ase import Atoms
import numpy as np


CACHE_ENV_VAR = "PFP_CACHE"
DEFAULT_CACHE_PATH = "./cache"

@lru_cache
def setup_cache(
    cache_path: Optional[str] = None,
) -> joblib.Memory:
    if cache_path is None:
        cache_path = os.getenv(CACHE_ENV_VAR, DEFAULT_CACHE_PATH)
        print(f"PES fingerprint cache path set to \"{cache_path}\". To change, set {CACHE_ENV_VAR} environment variable.")

    return joblib.Memory(cache_path)

def serialize_atoms(atoms: Atoms) -> Dict[str, Union[List[str], np.ndarray]]:
    return dict(
        symbols=[str(el) for el in atoms.symbols],
        positions=atoms.positions,
        cell=atoms.cell.array,
    )

def deserialize_atoms(structure_params: Dict[str, Union[List[str], np.ndarray]]) -> Atoms:
    assert len(structure_params) == 3
    return Atoms(**structure_params, pbc=True)
