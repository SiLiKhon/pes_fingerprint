from typing import Tuple

import numpy as np
from ase import Atoms

def deduce_grid_size(atoms: Atoms, max_step_size: float) -> Tuple[int, int, int]:
    lattice_vectors_lengths = np.sqrt((atoms.cell.array**2).sum(axis=1))
    grid_size = np.ceil(lattice_vectors_lengths / max_step_size).astype(int)
    assert grid_size.shape == (3,)
    assert (grid_size > 0).all()
    return tuple(grid_size)
