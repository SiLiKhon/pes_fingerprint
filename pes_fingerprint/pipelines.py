from typing import Tuple, Callable
from pydantic import BaseModel
from functools import lru_cache

import numpy as np
from joblib.memory import Memory
from tqdm.auto import trange

from data.catalogue import find_and_load
from models import Model, I_M3GNet
from models.i_m3gnet.utils import predict_in_batches

from .pes import PESCalculator
from .topological import wave_search

memory = Memory("cache/")

class PESBarrierPipelineConfig(BaseModel):
    model_checkpoint: str
    dataset_id: str
    grid_size: Tuple[int, int, int]
    radius_cutoff: float = 1.2
    assert_distance_validation: bool = False
    assert_energy_validation: bool = False

@lru_cache
def _load_model_from_cp(cp: str) -> Model:
    config = I_M3GNet.get_basic_predict_config()
    config["load_cp"] = cp
    return Model(config)

def make_calculator(cp: str, superbatch_size: int = 2000, **kwargs) -> Callable:
    potential = _load_model_from_cp(cp).i_model.potential
    def _calc(structs):
        return np.concatenate([
            predict_in_batches(
                potential, structs[i: i + superbatch_size], verbose=False, include_stresses=False, **kwargs
            )["energies"]
            for i in trange(0, len(structs), superbatch_size)
        ], axis=0)
    return _calc

@memory.cache
def pes_barrier_pipeline(
    config: PESBarrierPipelineConfig
):
    data = find_and_load("", id=config.dataset_id)
    assert np.allclose(
        data.structures[0].positions[:-1],
        data.structures[1].positions[:-1],
    )
    assert not np.allclose(
        data.structures[0].positions[-1],
        data.structures[1].positions[-1],
    )
    base_structure = data.structures[np.argmin(data.energies)]

    pes_mgr = PESCalculator(
        base_structure,
        ids_of_interest=[len(base_structure) - 1],
        calculator=make_calculator(config.model_checkpoint),
        grid_size=config.grid_size,
        radius_cutoff=config.radius_cutoff,
        assert_distance_validation=config.assert_distance_validation,
        assert_energy_validation=config.assert_energy_validation,
    )
    pes_mgr.run()
    (pes,) = pes_mgr.wrapped_pes
    (imin,) = pes_mgr.wrapped_imin
    wavefront = []
    levels = wave_search(pes, imin, fill_wavefront_ids_list=wavefront, early_stop="any_face")

    smallest_barrier = min(
        levels[0].min(), levels[-1].min(),
        levels[:, 0].min(), levels[:, -1].min(),
        levels[..., 0].min(), levels[..., -1].min(),
    )

    return dict(
        pes=pes,
        imin=imin,
        wavefront=wavefront,
        levels=levels,
        smallest_barrier=smallest_barrier,
        config=config,
    )
