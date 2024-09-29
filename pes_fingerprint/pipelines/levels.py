from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import ase

from .cache_utils import setup_cache, serialize_atoms, deserialize_atoms
from .mpe import calculate_mpe
from ..topological import wave_search

_memory = setup_cache()


@_memory.cache
def _calculate_levels(
    pes: np.ndarray,
    cell: np.ndarray,
    seed: Tuple[int, int, int],
    early_stop_level: float,
    minimal_lvl_increase: float,
    connectivity: Union[float, Literal["all", "minimal", "sqrt3"]],
) -> np.ndarray:
    assert pes.ndim == 3
    assert all(dim % 2 == 1 for dim in pes.shape)
    assert cell.shape == (3, 3)

    args = dict(
        potential=pes,
        seed=seed,
        early_stop="level",
        early_stop_level=early_stop_level,
        minimal_lvl_increase=minimal_lvl_increase,
        progress_bar=False,
        connectivity=connectivity,
    )

    if connectivity != "all":
        grid_size = (np.array(pes.shape) - 1) / 2
        assert (tuple(np.array(grid_size) * 2 + 1) == pes.shape)
        multipliers = np.array(pes.shape) / grid_size
        args["cell"] = cell * multipliers[:, None]

    return wave_search(**args)

def calculate_levels(
    *,
    atoms: ase.Atoms,
    mobile_id: int,
    early_stop_level: float = 6.0,
    minimal_lvl_increase: float = 0.1,
    connectivity: Union[float, Literal["all", "minimal", "sqrt3"]] = "sqrt3",
    mpe_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if mpe_params is None:
        mpe_params = {}
    mpe_res = calculate_mpe(
        atoms=atoms,
        mobile_id=mobile_id,
        **mpe_params,
    )
    return _calculate_levels(
        pes=mpe_res["pes"],
        cell=atoms.cell.array,
        seed=mpe_res["imin"],
        early_stop_level=early_stop_level,
        minimal_lvl_increase=minimal_lvl_increase,
        connectivity=connectivity,
    )
