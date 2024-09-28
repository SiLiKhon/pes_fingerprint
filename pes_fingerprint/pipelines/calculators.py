from typing import Callable, Dict, List, Union

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
def basic_m3gnet_calc_factory() -> Calculator:
    from m3gnet.models import M3GNet, M3GNetCalculator, Potential
    return M3GNetCalculator(Potential(M3GNet.load()))
