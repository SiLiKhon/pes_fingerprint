from typing import Callable, Dict

import numpy as np
from ase import Atom, Atoms


PreprocessorType = Callable[[Atoms], Atoms]
FactoryType = Callable[..., PreprocessorType]
_PREPROCESSOR_FACTORIES: Dict[str, FactoryType] = {}


def factory(key: str) -> Callable[[FactoryType], FactoryType]:
    def _wrapper(func: FactoryType) -> FactoryType:
        assert key not in _PREPROCESSOR_FACTORIES
        _PREPROCESSOR_FACTORIES[key] = func
        return func
    return _wrapper


def get_preprocessor(key: str, **kwargs) -> PreprocessorType:
    factory = _PREPROCESSOR_FACTORIES[key]
    return factory(**kwargs)


@factory("single_mobile_ion")
def single_mobile_ion_factory(mobile_species: str = "Li") -> PreprocessorType:
    def _prep(ats: Atoms) -> Atoms:
        ats_subset = ~np.array(ats.symbols == mobile_species)
        assert not ats_subset.all(), f"got a structure without mobile species ({mobile_species})"
        ats_subset[(~ats_subset).argmax()] = True
        return ats[ats_subset].copy()

    return _prep


@factory("single_mobile_ion_interstitial")
def single_mobile_ion_interstitial_factory(mobile_tag: int = -1, mobile_species: str = "Li") -> PreprocessorType:
    def _prep(ats: Atoms) -> Atoms:
        ats = ats.copy()
        # quick and dirty location picking logic (ignoring neighbor images but limited to inner subcell)
        candidate_positions = np.random.default_rng().uniform(0.25, 0.75, size=(10, 3)) @ ats.cell.array
        position = candidate_positions[
            np.linalg.norm(
                candidate_positions[:, None, :] - ats.get_positions(wrap=True)[None, :, :],
                axis=-1,
            ).min(axis=-1).argmax()
        ]
        ats.append(Atom(mobile_species, position=position, tag=mobile_tag))
        return ats

    return _prep
