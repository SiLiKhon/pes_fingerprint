from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np

from pes_fingerprint.pes import PESCalculator, BarrierCalculator
from pes_fingerprint.topological.utils import visualize_wavefront


@pytest.fixture
def structure():
    stru = bulk("Al").repeat((2, 2, 2))
    stru.symbols[-1] = "Ag"
    return stru

def test_calculator_sp(structure):
    pes_calc = PESCalculator(
        source_atoms=structure,
        calculator=EMT(),
        species_of_interest="Ag",
        grid_size=(6, 6, 6),
    )
    pes_calc.run()
    assert len(pes_calc.imin) == len(pes_calc.pes) == 1
    assert pes_calc.pes[0].shape == (6, 6, 6)
    print(np.round(pes_calc.pes[0], 2))

def test_calculator_id(structure):
    pes_calc = PESCalculator(
        source_atoms=structure,
        calculator=EMT(),
        ids_of_interest=[5, 7],
        grid_size=(6, 6, 6),
        assert_minimum_inside=False,
    )
    pes_calc.run()
    assert len(pes_calc.imin) == len(pes_calc.pes) == 2
    assert pes_calc.pes[0].shape == pes_calc.pes[1].shape == (6, 6, 6)

@pytest.mark.parametrize("connectivity,minimal_lvl_increase", [
    ("all", 0.1), ("minimal", 0.1), ("all", 0.0), ("minimal", 0.0)
])
def test_barrier_search(structure, connectivity, minimal_lvl_increase):
    bar_calc = BarrierCalculator(
        source_atoms=structure,
        calculator=EMT(),
        ids_of_interest=[5, 7],
        grid_size=(20, 20, 20),
        assert_minimum_inside=False,
        store_wavefront=True,
        connectivity=connectivity,
        minimal_lvl_increase=minimal_lvl_increase,
    )
    bar_calc.run()
    for m, imob in zip(bar_calc.barrier, bar_calc.ids_of_interest):
        print(f"Minimum barrier for ion #{imob} is {m:.3f} eV (according to the EMT calculator)")

    export_path = Path(__file__).parent / "vis"
    if not export_path.exists():
        export_path.mkdir()
    for levels, wf, imob in zip(bar_calc.levels, bar_calc.wavefronts, bar_calc.ids_of_interest):
        assert levels.shape == (41, 41, 41)
        fig = visualize_wavefront(
            wf=wf,
            target_shape=levels.shape,
            unit_cell=structure.cell.array * (41 / 20),
            energy_levels=levels,
        )
        fig.write_html(
            export_path / f"barrier_search_{imob:02d}_connectivity_{connectivity}_min_inc_{minimal_lvl_increase:.1}.html",
            auto_play=False,
        )
