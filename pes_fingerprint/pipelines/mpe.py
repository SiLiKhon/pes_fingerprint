from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import numpy as np
import ase
import ase.calculators
import ase.calculators.calculator
import joblib

from ..pes import BarrierCalculator
from .calculators import get_calculator

_memory = joblib.Memory("cache")

@_memory.cache
def _calculate_mpe(**kwargs) -> Dict[str, Any]:
    params = dict()
    params.update(kwargs)

    calculator_params = kwargs.pop("calculator_params")
    calculator = get_calculator(**calculator_params)

    structure_params = kwargs.pop("structure_params")
    assert len(structure_params) == 3
    atoms = ase.Atoms(**structure_params, pbc=True)

    mobile_id = kwargs.pop("mobile_id")
    calc = BarrierCalculator(
        source_atoms=atoms,
        calculator=calculator,
        ids_of_interest=[mobile_id],
        **kwargs,
    )
    calc.run()
    (pes,) = calc.wrapped_pes
    (imin,) = calc.wrapped_imin
    (wavefront,) = calc.wavefronts if hasattr(calc, "wavefronts") else (None,)
    (levels,) = calc.levels
    (barrier,) = calc.barrier
    (original_pes,) = calc.pes
    (original_imin,) = calc.imin
    return dict(
        pes=pes,
        imin=imin,
        original_imin=original_imin,
        original_shape=original_pes.shape,
        wavefront=wavefront,
        levels=levels,
        barrier=barrier,
        config=params,
        error_msgs=calc._error_msgs[:],
    )

def calculate_mpe(
    *,
    atoms: ase.Atoms,
    mobile_id: int,
    calculator_params: Union[Dict[str, Any], str] = "basic_m3gnet",
    grid_size: Union[Tuple[int, int, int], float] = 0.25,
    radius_cutoff: float = 1.2,
    connectivity: Union[float, Literal["all", "minimal", "sqrt3"]] = "sqrt3",
    store_wavefront=False,
    **kwargs,
) -> Dict[str, Any]:
    if isinstance(calculator_params, str):
        calculator_params = dict(key=calculator_params)
    params = dict(
        grid_size=grid_size,
        radius_cutoff=radius_cutoff,
        connectivity=connectivity,
        store_wavefront=store_wavefront,
        assert_distance_validation=False,
        assert_energy_validation=False,
        assert_minimum_inside=False,
    )
    params.update(kwargs)

    return _calculate_mpe(
        **params,
        mobile_id=mobile_id,
        calculator_params=calculator_params,
        structure_params=dict(
            symbols=[str(el) for el in atoms.symbols],
            positions=atoms.positions,
            cell=atoms.cell.array,
        ),
    )
