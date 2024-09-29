from typing import Callable, Dict, List, Union

import numpy as np
from tqdm.auto import trange
from ase.calculators.calculator import Calculator
from ase import Atoms

CalcType = Union[Calculator, Callable[[List[Atoms]], List[float]]]
FactoryType = Callable[..., CalcType]
_CALCULATOR_FACTORIES: Dict[str, FactoryType] = {}


def factory(key: str) -> Callable[[FactoryType], FactoryType]:
    def _wrapper(func: FactoryType) -> FactoryType:
        assert key not in _CALCULATOR_FACTORIES
        _CALCULATOR_FACTORIES[key] = func
        return func
    return _wrapper

def get_calculator(key: str, **kwargs) -> CalcType:
    factory = _CALCULATOR_FACTORIES[key]
    return factory(**kwargs)

@factory("basic_m3gnet")
def basic_m3gnet_calc_factory(**kwargs) -> Calculator:
    from .m3gnet_utils import setup_tensorflow
    setup_tensorflow(
        gpu_memory_growth=kwargs.pop("gpu_memory_growth", True),
        disable_tensor_float_32=kwargs.pop("disable_tensor_float_32", True),
    )
    from m3gnet.models import M3GNet, M3GNetCalculator, Potential
    return M3GNetCalculator(Potential(M3GNet.load()))

@factory("batched_m3gnet")
def batched_m3gnet_calc_factory(
    superbatch_size: int = 2000,
    gpu_memory_goal: float = 2500.0,
    **kwargs,
) -> Callable[[List[Atoms]], List[float]]:
    from .m3gnet_utils import setup_tensorflow
    setup_tensorflow(
        gpu_memory_growth=kwargs.pop("gpu_memory_growth", True),
        disable_tensor_float_32=kwargs.pop("disable_tensor_float_32", True),
    )
    from m3gnet.models import M3GNet, Potential
    from .m3gnet_utils import predict_in_batches
    potential = Potential(M3GNet.load())

    # This func aims to estimate the GPU memory usage and keep it around `gpu_memory_goal` GB.
    # Note that at least up to x2 outliers are possible.
    def _get_batch_size_for_structure(s, model):
        graph = model.graph_converter(s)
        size = sum([i.nbytes / 1024 for i in graph.as_list() if i is not None])
        return np.ceil(32 * 200 / size * gpu_memory_goal / 2500.0).astype(int)

    def _calc(structs):
        minibatch_size = int(_get_batch_size_for_structure(structs[0], potential.model))
        return np.concatenate([
            predict_in_batches(
                potential, structs[i: i + superbatch_size], verbose=False, include_stresses=False, batch_size=minibatch_size, **kwargs
            )["energies"]
            for i in trange(0, len(structs), superbatch_size)
        ], axis=0).tolist()
    return _calc
