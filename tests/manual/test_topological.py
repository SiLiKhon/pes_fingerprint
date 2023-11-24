from typing import Literal, Tuple

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


def prepare_2d_spiral(
    size: Tuple[int, int] = (100, 100),
) -> np.ndarray:
    data = np.zeros(shape=size, dtype="float64")

    alpha = 0
    d_alpha = 2 * np.pi / 40
    r = 0
    d_r = 0.005 * min(size)
    while True:
        x0 = size[0] / 2 + r * np.sin(alpha)
        y0 = size[1] / 2 + r * np.cos(alpha)
        x1 = size[0] / 2 + (r + d_r) * np.sin(alpha + d_alpha)
        y1 = size[1] / 2 + (r + d_r) * np.cos(alpha + d_alpha)

        for ii in np.linspace(0, 1, int(10 * (1 + 4 * r / min(size))), endpoint=False):
            x = x0 * (1 - ii) + x1 * ii
            y = y0 * (1 - ii) + y1 * ii
            data -= np.fromfunction(
                lambda ix, iy: np.exp((-(x - ix)**2 - (y - iy)**2) / (min(size) / 30)**2),
                shape=data.shape,
            ) * (1 + 0.5 * np.sin(alpha * 10))

        r += d_r
        alpha += d_alpha

        if r > 0.5 * min(size):
            break

    return data


if __name__ == "__main__":
    data = prepare_2d_spiral(size=(100, 100))
    data_3d = np.tile(data[..., None], (1, 1, 10))
    print("data prepared")

    wavefront = []
    levels = wave_search(
        data_3d,
        np.unravel_index(data_3d.argmin(), data_3d.shape),
        minimal_lvl_increase=0.5,
        fill_wavefront_ids_list=wavefront,
    )
    print("levels calculated")
    # plt.imshow(levels.std(axis=-1))
    # plt.colorbar()
    # plt.show()
    assert np.allclose(levels.std(axis=-1), 0, atol=1e-10)

    plt.subplot(1, 2, 1)
    plt.imshow(data)
    plt.colorbar()
    plt.subplot(1, 2, 2)

    plt.imshow(levels[..., 0])
    plt.colorbar()
    plt.show()

    print("All done")
    # np.random.seed(42)
    # test_1d_data()
    # plt.legend()
    # plt.show()
