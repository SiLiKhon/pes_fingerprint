from typing import Tuple, Optional, List, Union, Callable

from tqdm.auto import tqdm
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


class PESManager:
    def __init__(
        self,
        source_atoms: Atoms, *,
        grid_size: Tuple[int, int, int],
        ids_of_interest: Optional[List[int]] = None,
        species_of_interest: Optional[str] = None,
    ):
        assert (ids_of_interest is None) != (species_of_interest is None), (
            "Exactly one of the two arguments (ids_of_interest, species_of_interest) "
            "should be provided."
        )
        self.atoms = source_atoms.copy()
        self.grid_size = grid_size

        if species_of_interest is not None:
            (self.soi_ids,) = np.where([el == species_of_interest for el in self.atoms.symbols])
        elif ids_of_interest is not None:
            self.soi_ids = np.array(ids_of_interest)
        else:
            assert False

        assert self.soi_ids.ndim == 1
        assert len(self.soi_ids) > 0

    @staticmethod
    def _generate_grid(cell: np.ndarray, size: Tuple[int, int, int]) -> np.ndarray:
        assert cell.shape == (3, 3)
        grid_scaled = np.stack(
            np.meshgrid(
                *[np.linspace(0, 1, ni, endpoint=False) for ni in size],
                indexing="ij",
            ),
            axis=-1,
        )

        return grid_scaled @ cell

    @staticmethod
    def _wrap_calculator(calc: Calculator) -> Callable:
        def _calculate(structs: List[Atoms]) -> List[float]:
            energies = []
            working_struct = structs[0].copy()
            working_struct.calc = calc
            for struct in tqdm(structs, desc="Calculating energies"):
                working_struct.positions = struct.positions
                assert working_struct == struct
                energies.append(working_struct.get_potential_energy())
            return energies
        return _calculate

    def calculate_pes(
        self,
        calculator: Union[Calculator, Callable],
        cutoff: float = 0.8,
    ) -> List[np.ndarray]:
        if isinstance(calculator, Calculator):
            calculator = PESManager._wrap_calculator(calculator)
        grid = PESManager._generate_grid(self.atoms.cell.array, self.grid_size)

        scaled_pos = self.atoms.get_scaled_positions()
        assert (scaled_pos >= 0).all()
        assert (scaled_pos < 1).all()

        grid_valid = [
            np.sqrt(
                (
                    (
                        self.atoms[
                            [i for i in range(len(self.atoms)) if i != i_mob]
                        ].repeat((3, 3, 3)).positions[:, None, None, None, :]
                        - (grid + self.atoms.cell.array.sum(axis=0))[None]
                    )**2
                ).sum(axis=-1)
            ).min(axis=0) >= cutoff
            for i_mob in self.soi_ids
        ]

        structs = []
        for i_mob, mask in zip(tqdm(self.soi_ids, desc="Generating structures for calculation"), grid_valid):
            mobile_positions = grid[mask]
            assert mobile_positions.ndim == 2
            assert mobile_positions.shape[1] == 3
            structs.append([])
            for pos in mobile_positions:
                structs[-1].append(self.atoms.copy())
                structs[-1][-1].positions[i_mob] = pos

        pes = []
        for structs_subset, mask in zip(structs, grid_valid):
            energies = np.empty(dtype=float, shape=mask.shape)
            energies[mask] = calculator(structs_subset)
            fill_value = energies[mask].max()
            energies[~mask] = fill_value
            pes.append(energies)
        return pes
