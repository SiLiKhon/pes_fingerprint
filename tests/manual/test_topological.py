from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from pes_fingerprint.topological import wave_search

def prepare_1d_data(
    length: int = 1000,
    width: int = 10,
    length_axis: Literal[0, 1, 2] = 0,
) -> np.ndarray:
    xx = np.linspace(0, 1, length, endpoint=False)
    yy = np.cos(2 * np.pi * 5 * xx) + 5 * (xx - np.random.uniform())**2 + np.cos(2 * np.pi * 15 * xx)
    slices = [None, None, None]
    slices[length_axis] = slice(None)
    slices = tuple(slices)
    yy = yy[slices]

    tiles = [width] * 3
    tiles[length_axis] = 1

    return np.tile(yy, tiles)

def test_1d_data():
    for axis in range(3):
        excl_axes = tuple(a for a in range(3) if a != axis)

        potential = prepare_1d_data(length_axis=axis)
        assert np.allclose(potential.std(axis=excl_axes), 0, atol=1e-10)

        levels = wave_search(potential=potential, seed=np.unravel_index(potential.argmin(), potential.shape))
        assert np.allclose(levels.std(axis=excl_axes), 0, atol=1e-10)

        selection_1d = [0] * 3
        selection_1d[axis] = slice(None)
        selection_1d = tuple(selection_1d)

        potential_1d = potential[selection_1d]
        potential_1d = potential_1d - potential_1d.min()
        levels_1d = levels[selection_1d]

        (mpl_line,) = plt.plot(potential_1d, label=f"potential ({'XYZ'[axis]})")
        plt.plot(levels_1d, "--", color=mpl_line.get_color(), label=f"levels ({'XYZ'[axis]})")


if __name__ == "__main__":
    np.random.seed(42)
    test_1d_data()
    plt.legend()
    plt.show()
