import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np

from pes_fingerprint.pes import PESCalculator, BarrierCalculator


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

def test_barrier_search(structure):
    bar_calc = BarrierCalculator(
        source_atoms=structure,
        calculator=EMT(),
        ids_of_interest=[5, 7],
        grid_size=(6, 6, 6),
        assert_minimum_inside=False,
        store_wavefront=True,
    )
    bar_calc.run()
    for m, imob in zip(bar_calc.barrier, bar_calc.ids_of_interest):
        print(f"Minimum barrier for ion #{imob} is {m:.3f} eV (according to the EMT calculator)")
