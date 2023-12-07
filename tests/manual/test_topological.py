from typing import Literal, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pes_fingerprint.topological import wave_search, utils
from pes_fingerprint.topological.barrier_validated_noopt import wave_search as wave_search_noopt


def _compare_wavefronts(
    wf1: List[np.ndarray],
    wf2: List[np.ndarray],
):
    assert len(wf1) == len(wf2)
    def _make_tokens(frame):
        tokens = np.ascontiguousarray(frame).reshape(-1).view("i8,i8,i8").tolist()
        assert len(tokens) == len(frame)
        return set(tokens)

    for f1, f2 in zip(wf1, wf2):
        assert _make_tokens(f1) == _make_tokens(f2)
    print("Wavefronts identical")


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


def test_1d_data(early_stop: bool):
    for axis in range(3):
        excl_axes = tuple(a for a in range(3) if a != axis)
        additional_args = {}
        if early_stop:
            additional_args["early_stop"] = "faces"
            additional_args["early_stop_faces"] = [axis]

        potential = prepare_1d_data(length_axis=axis)
        assert np.allclose(potential.std(axis=excl_axes), 0, atol=1e-10)

        tmp_wf = []
        levels = wave_search(
            potential=potential,
            seed=np.unravel_index(potential.argmin(),potential.shape),
            fill_wavefront_ids_list=tmp_wf,
            **additional_args,
        )
        if not early_stop:
            tmp_wf_noopt = []
            levels_noopt = wave_search_noopt(
                potential=potential,
                seed=np.unravel_index(potential.argmin(), potential.shape),
                fill_wavefront_ids_list=tmp_wf_noopt,
            )
            assert np.allclose(levels, levels_noopt)
            _compare_wavefronts(tmp_wf, tmp_wf_noopt)
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


def prepare_2d_data(
    size_2d: Tuple[int, int] = (50, 80),
    depth: int = 7,
    depth_axis: Literal[0, 1, 2] = 0,
) -> np.ndarray:
    spiral_2d = prepare_2d_spiral(size_2d)
    indexer = [slice(None)] * 2
    repeats = [1, 1]
    indexer.insert(depth_axis, None)
    repeats.insert(depth_axis, depth)
    return np.tile(spiral_2d[tuple(indexer)], repeats)


def test_2d_data(early_stop: bool) -> np.ndarray:
    levels = {}
    data = {}
    shape_v = (50, 80)
    shape_h = shape_v[::-1]
    depth = 7

    wavefront = []
    for size_2d in [shape_v, shape_h]:
        levels[size_2d] = {}
        data[size_2d] = {}
        for axis in range(3):
            additional_args = {}
            if early_stop:
                additional_args["early_stop"] = "faces"
                additional_args["early_stop_faces"] = [i for i in range(3) if i != axis]
            data[size_2d][axis] = prepare_2d_data(size_2d=size_2d, depth_axis=axis, depth=depth)
            tmp_wf = []
            levels[size_2d][axis] = wave_search(
                potential=data[size_2d][axis],
                seed=np.unravel_index(data[size_2d][axis].argmin(), data[size_2d][axis].shape),
                fill_wavefront_ids_list=tmp_wf,
                **additional_args,
            )
            if (size_2d == shape_h and axis == 0):
                wavefront = tmp_wf

            if not early_stop:
                tmp_wf_noopt = []
                lvls_noopt = wave_search_noopt(
                    potential=data[size_2d][axis],
                    seed=np.unravel_index(data[size_2d][axis].argmin(), data[size_2d][axis].shape),
                    fill_wavefront_ids_list=tmp_wf_noopt,
                )
                assert np.allclose(levels[size_2d][axis], lvls_noopt)
                _compare_wavefronts(tmp_wf, tmp_wf_noopt)
            assert np.allclose(levels[size_2d][axis].std(axis=axis), 0, atol=1e-10)
            idx = [slice(None)] * 2; idx.insert(axis, 0)
            levels[size_2d][axis] = levels[size_2d][axis][tuple(idx)]

        assert np.allclose(levels[size_2d][0], levels[size_2d][1], atol=1e-10)
        assert np.allclose(levels[size_2d][1], levels[size_2d][2], atol=1e-10)

    if not early_stop:
        assert np.allclose(
            levels[shape_v][0][:, 15: 65],
            levels[shape_h][0][15: 65, :]
        )

    plt.subplot(1, 2, 1)
    plt.imshow(data[shape_h][0][0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(levels[shape_h][0])
    plt.colorbar()

    def _make_dense_frame(ids):
        result = np.zeros(shape=(depth,) + shape_h, dtype=bool)
        result[tuple(ids.T)] = True
        return result

    wavefront = np.array([_make_dense_frame(ids) for ids in wavefront])
    return wavefront


def prepare_3d_spiral(
    size: Tuple[int, int, int] = (40, 50, 150),
) -> np.ndarray:
    data = np.zeros(shape=size, dtype="float64")

    z_id = np.argmax(size)
    (x_id, y_id) = [i for i in range(3) if i != z_id]
    rmax = min(size[x_id], size[y_id]) / 2

    alpha = 0
    d_alpha = 2 * np.pi / 40
    r = 0
    d_r = rmax / (4 * np.pi / d_alpha)
    while True:
        x0 = size[x_id] / 2 + r * np.sin(alpha)
        y0 = size[y_id] / 2 + r * np.cos(alpha)
        x1 = size[x_id] / 2 + (r + d_r) * np.sin(alpha + d_alpha)
        y1 = size[y_id] / 2 + (r + d_r) * np.cos(alpha + d_alpha)

        for ii in np.linspace(0, 1, int(10 * (1 + 2 * r / rmax)), endpoint=False):
            x = x0 * (1 - ii) + x1 * ii
            y = y0 * (1 - ii) + y1 * ii
            z = size[z_id] * r / rmax
            data -= np.fromfunction(
                lambda *ixyz: np.exp(
                    (
                        -(x - ixyz[x_id])**2
                        -(y - ixyz[y_id])**2
                        -(z - ixyz[z_id])**2
                    ) / (rmax / 5)**2
                ),
                shape=data.shape,
            ) * (1 + 0.5 * np.sin(alpha * 10))

        r += d_r
        alpha += d_alpha

        if r > 0.5 * min(size):
            break

    return data


def test_3d_data(early_stop: bool):
    shape = (40, 50, 70)
    data = prepare_3d_spiral(size=shape)
    additional_args = {}
    if early_stop:
        additional_args["early_stop"] = "any_face"
    wavefront = []
    levels = wave_search(
        data,
        seed=np.unravel_index(data.argmin(), data.shape),
        fill_wavefront_ids_list=wavefront,
        **additional_args,
    )
    if not early_stop:
        wavefront_noopt = []
        levels_noopt = wave_search_noopt(
            data,
            seed=np.unravel_index(data.argmin(), data.shape),
            fill_wavefront_ids_list=wavefront_noopt,
        )
        assert np.allclose(levels, levels_noopt)
        _compare_wavefronts(wavefront, wavefront_noopt)

    utils.visualize_wavefront(wavefront, shape).show()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("tests", type=str, choices=["all", "1d", "2d", "3d"])
    parser.add_argument("--early_stop", action="store_true", default=False)
    args = parser.parse_args()
    tests_to_do = args.tests
    early_stop = args.early_stop

    if tests_to_do in ["1d", "all"]:
        np.random.seed(42)
        test_1d_data(early_stop=early_stop)
        plt.legend()
        plt.show()

    if tests_to_do in ["2d", "all"]:
        wf = test_2d_data(early_stop=early_stop)[:, 0]
        plt.show()

        fig = plt.figure()
        img = plt.imshow(wf[0])
        def _animate_func(i):
            img.set_array(wf[i])
            return [img]
        anim = FuncAnimation(fig, _animate_func, frames=len(wf), interval=25)
        plt.show()

    if tests_to_do in ["3d", "all"]:
        test_3d_data(early_stop=early_stop)
