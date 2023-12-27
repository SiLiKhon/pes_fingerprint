from typing import Tuple, Optional, List, Union, Callable, Literal

from tqdm.auto import tqdm
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from ..topological import wave_search


class PESCalculator:
    energy_validation_threshold: float = 0.2
    assert_energy_validation: bool = True
    distance_validation_threshold: float = 0.5
    assert_distance_validation: bool = True
    radius_cutoff: float = 0.8
    grid_size: Tuple[int, int, int] = (40, 40, 40)
    assert_minimum_inside: bool = True

    def __init__(
        self,
        source_atoms: Atoms, *,
        calculator: Union[Calculator, Callable],
        ids_of_interest: Optional[List[int]] = None,
        species_of_interest: Optional[str] = None,
        **kwargs,
    ):
        assert (ids_of_interest is None) != (species_of_interest is None), (
            "Exactly one of the two arguments (ids_of_interest, species_of_interest) "
            "should be provided."
        )
        self.source_atoms = source_atoms.copy()

        if species_of_interest is not None:
            (self.ids_of_interest,) = np.where([el == species_of_interest for el in self.source_atoms.symbols])
        elif ids_of_interest is not None:
            self.ids_of_interest = np.array(ids_of_interest)
        else:
            assert False

        assert self.ids_of_interest.ndim == 1
        assert len(self.ids_of_interest) > 0
        assert ((self.ids_of_interest >= 0) & (self.ids_of_interest < len(self.source_atoms))).all()

        if isinstance(calculator, Calculator):
            calculator = PESCalculator._convert_calculator(calculator)
        self.calculator = calculator

        for k, v in kwargs.items():
            assert hasattr(self, k), f"Bad argument: {str(k)}"
            setattr(self, k, v)

        self._error_msgs = []

    def run(self) -> None:
        self._calculate_pes()
        self._find_minimum()

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
    def _convert_calculator(calc: Calculator) -> Callable:
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

    def _calculate_pes(self) -> None:
        assert not hasattr(self, "pes"), "Seems like PES was already calculated"
        grid = PESCalculator._generate_grid(self.source_atoms.cell.array, self.grid_size)

        scaled_pos = self.source_atoms.get_scaled_positions()
        assert (scaled_pos >= 0).all()
        assert (scaled_pos < 1).all()

        grid_valid = [
            np.sqrt(
                (
                    (
                        self.source_atoms[
                            [i for i in range(len(self.source_atoms)) if i != i_mob]
                        ].repeat((3, 3, 3)).positions[:, None, None, None, :]
                        - (grid + self.source_atoms.cell.array.sum(axis=0))[None]
                    )**2
                ).sum(axis=-1)
            ).min(axis=0) >= self.radius_cutoff
            for i_mob in self.ids_of_interest
        ]

        structs = []
        for i_mob, mask in zip(tqdm(self.ids_of_interest, desc="Generating structures for calculation"), grid_valid):
            mobile_positions = grid[mask]
            assert mobile_positions.ndim == 2
            assert mobile_positions.shape[1] == 3
            structs.append([])
            for pos in mobile_positions:
                structs[-1].append(self.source_atoms.copy())
                structs[-1][-1].positions[i_mob] = pos

        (self.base_energy,) = self.calculator([self.source_atoms.copy()])
        self.pes = []
        for structs_subset, mask in zip(structs, grid_valid):
            energies = np.empty(dtype=float, shape=mask.shape)
            energies[mask] = self.calculator(structs_subset)
            fill_value = energies[mask].max()
            energies[~mask] = fill_value
            self.pes.append(energies)

        failed_checks = []
        for pes_i, i_mob in zip(self.pes, self.ids_of_interest):
            if pes_i.min() + self.energy_validation_threshold < self.base_energy:
                failed_checks.append(
                    (pes_i.min(), i_mob)
                )
        if len(failed_checks):
            msg = (
                f"Energy checks failed (base structure energy = {self.base_energy:.3f} eV, "
                f"fluctuations up to {self.energy_validation_threshold:.3f} eV allowed) for the following species:\n"
                + "\n".join(
                    f"  Ion #{i_mob}: min energy = {emin:.3f} eV ({self.base_energy - emin:.3f} eV below base)"
                    for emin, i_mob in failed_checks
                )
            )
            self._error_msgs.append(msg)
            if self.assert_energy_validation:
                assert False, msg
            else:
                print(f"=== === === WARNING! === === ===")
                print(msg)
                print(f"=== === === ===  === === === ===")

    def _find_minimum(self):
        assert not hasattr(self, "imin"), "Seems like minima were already found"
        self.imin = [
            np.unravel_index(pes_i.argmin(), pes_i.shape)
            for pes_i in self.pes
        ]
        np_imin = np.array(self.imin)

        # Perform checks
        min_at_edge = (np_imin == 0) | (np_imin == self.grid_size)
        if min_at_edge.any():
            (bad_min_ids,) = np.where(min_at_edge.any(axis=1))
            msg = "Minimum at the edge of the grid for following ion(s): " + (
                ", ".join(str(self.ids_of_interest[i]) for i in bad_min_ids)
            )
            self._error_msgs.append(msg)
            if self.assert_minimum_inside:
                assert False, msg
            else:
                print(f"=== === === WARNING! === === ===")
                print(msg)
                print(f"=== === === ===  === === === ===")

        failed_checks = []
        for imin, i_mob in zip(self.imin, self.ids_of_interest):
            predicted = (np.array(imin) / np.array(self.grid_size)) @ self.source_atoms.cell.array
            initial = self.source_atoms.positions[i_mob]
            dist = np.linalg.norm(predicted - initial)
            if dist > self.distance_validation_threshold:
                failed_checks.append(
                    (imin, i_mob, dist)
                )
        if len(failed_checks):
            msg = (
                f"Minima distance checks failed for the following species (distance threshold is {self.distance_validation_threshold:.3f} A):\n"
                + "\n".join(
                    f"  Ion #{i_mob}: distance is {dist:.3f} A, predicted i_min = {imin}, "
                    f"base i_min = {tuple(np.round(self.source_atoms.get_scaled_positions()[i_mob] * self.grid_size, 3))}"
                    for imin, i_mob, dist in failed_checks
                )
            )
            self._error_msgs.append(msg)
            if self.assert_distance_validation:
                assert False, msg
            else:
                print(f"=== === === WARNING! === === ===")
                print(msg)
                print(f"=== === === ===  === === === ===")


class BarrierCalculator(PESCalculator):
    minimal_lvl_increase: float = 0.1
    verbose_wave_search: bool = True
    connectivity: Union[float, Literal["all", "minimal", "sqrt3"]] = "all"

    def __init__(self, store_wavefront: bool, **kwargs):
        self.store_wavefront = store_wavefront
        super().__init__(**kwargs)

    @staticmethod
    def _wrap_pes(
        pes: np.ndarray,
        location: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Takes a PES 3d array and wraps it around twice, such that `location` is in the middle.
        """
        assert pes.ndim == 3

        pes_3x3x3 = np.tile(pes, (3, 3, 3))
        result = pes_3x3x3[tuple(slice(loc_i, loc_i + 2 * size_i + 1) for loc_i, size_i in zip(location, pes.shape))]
        i_center = pes.shape
        assert result[i_center] == pes[location]
        return result, i_center

    def _calculate_wrapped_pes(self) -> None:
        assert not hasattr(self, "wrapped_imin")
        assert not hasattr(self, "wrapped_pes")
        self.wrapped_imin, self.wrapped_pes = [], []
        for imin, pes_i in zip(self.imin, self.pes):
            (w_pes, w_imin) = BarrierCalculator._wrap_pes(pes_i, imin)
            self.wrapped_imin.append(w_imin)
            self.wrapped_pes.append(w_pes)

    def _calculate_levels(self) -> None:
        assert not hasattr(self, "levels")
        assert not hasattr(self, "wavefronts")
        self.levels = []
        if self.store_wavefront:
            self.wavefronts = []

        for imin, pes in zip(self.wrapped_imin, self.wrapped_pes):
            args = dict(
                potential=pes,
                seed=imin,
                early_stop="any_face",
                minimal_lvl_increase=self.minimal_lvl_increase,
                progress_bar=self.verbose_wave_search,
                connectivity=self.connectivity,
            )
            if self.store_wavefront:
                wf = []
                self.wavefronts.append(wf)
                args["fill_wavefront_ids_list"] = wf
            if self.connectivity != "all":
                assert (tuple(np.array(self.grid_size) * 2 + 1) == pes.shape)
                multipliers = np.array(pes.shape) / self.grid_size
                args["cell"] = self.source_atoms.cell.array * multipliers[:, None]
            self.levels.append(
                wave_search(**args)
            )

    def run(self) -> None:
        super().run()
        self._calculate_wrapped_pes()
        self._calculate_levels()

        self.barrier = [
            min(
                levels[0].min(), levels[-1].min(),
                levels[:, 0].min(), levels[:, -1].min(),
                levels[..., 0].min(), levels[..., -1].min(),
            ) for levels in self.levels
        ]
