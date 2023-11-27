from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

def wave_search(
    potential: np.ndarray,
    seed: Tuple[int, int, int],
    minimal_lvl_increase: float = 0.1,
    progress_bar: bool = True,
    fill_wavefront_ids_list: Optional[list] = None,
) -> np.ndarray:
    """
    Function that traverses a potential map on a grid (3d array) and returns an array
    of barriers (value for each grid point, what is the highest barrier to overcome to get
    there from the seed).
    """
    assert potential.ndim == 3
    assert not np.isnan(potential).any()
    assert all(i >= 0 for i in seed)

    shape = potential.shape

    # this will be the result map of barriers (shifted by the potential value at seed)
    levels = np.empty(dtype=float, shape=shape)
    levels[seed] = current_level = float(potential[seed])

    # boolean map to track iterative wave propagation
    observed = np.zeros(dtype=bool, shape=shape)
    observed[seed] = True

    # array of steps to make at single iteration (moving the border/wavefront)
    deltas = np.stack(np.meshgrid(*[[-1, 0, 1]] * 3), axis=-1).reshape(-1, 3)
    deltas = deltas[(deltas != 0).any(axis=-1)]

    # Adding deltas makes the shift, while adding zeros of same shape gives us a "map to source"
    # to know where we came there from. To do the two additions simultaneously, keep both deltas
    # and zeros in the same array.
    zeros_like_deltas = np.zeros_like(deltas)
    deltas_and_zeros = np.stack([deltas, zeros_like_deltas], axis=-1).astype("int64")
    assert deltas_and_zeros.shape == (26, 3, 2)

    # initialize the border
    border_last = np.stack([np.array([i]) for i in seed], axis=-1).astype("int64")

    if progress_bar:
        pbar = tqdm(total=observed.size)
        pbar.update(observed.sum())
    while True:
        if fill_wavefront_ids_list is not None:
            fill_wavefront_ids_list.append(border_last)

        # make single step
        border_next_with_mapback = (
            border_last[:, None, :, None] + deltas_and_zeros[None, :, :, :]
        ).reshape(-1, 3, 2)

        # remove out of bounds hops
        border_next_with_mapback = border_next_with_mapback[
            (border_next_with_mapback[..., 0] >= 0).all(axis=-1)
            & (border_next_with_mapback[..., 0] < [shape]).all(axis=-1)
        ]

        # remove the nodes we've been in
        border_next_with_mapback = border_next_with_mapback[
            ~observed[tuple(border_next_with_mapback[..., 0].T)]
        ]
        if not len(border_next_with_mapback):
            break

        # Now we want to adjust the wave propagation to only happen in the direction of
        # smallest ascent of the potential
        values_dest = potential[tuple(border_next_with_mapback[..., 0].T)]
        if (values_dest > current_level).all():
            current_level = max(values_dest.min(), current_level + minimal_lvl_increase)
        ascent_selection = values_dest <= current_level
        excluded_too_steep = border_next_with_mapback[~ascent_selection]
        border_next_with_mapback = border_next_with_mapback[ascent_selection]

        # So far `border_next_with_mapback` contains all the hops at current steps,
        # but some of them are reduntant (multiple hops to a single position). We need to
        # group the hops by destination and only select the ones comming from smallest observed level.
        border_next = np.ascontiguousarray(border_next_with_mapback[..., 0])
        assert border_next.dtype == np.int64
        destination_tags = border_next.reshape(-1).view(dtype='i8,i8,i8')
        assert destination_tags.shape == border_next.shape[:1]

        sort_ids = destination_tags.argsort()
        destination_tags = destination_tags[sort_ids]
        border_next_with_mapback = border_next_with_mapback[sort_ids]

        # Grouping trick borrowed from https://stackoverflow.com/a/43094244/3801744
        (_, group_ids) = np.unique(destination_tags, return_index=True)
        groups = np.split(border_next_with_mapback, group_ids[1:], axis=0)
        border_next_with_mapback = np.array([g[levels[tuple(g[..., 1].T)].argmin()] for g in groups])

        # Now we update the levels by max(potential here, smallest neighbor level)
        ids_src = tuple(border_next_with_mapback[..., 1].T)
        ids_dest = tuple(border_next_with_mapback[..., 0].T)
        levels[ids_dest] = np.maximum(
            levels[ids_src],
            potential[ids_dest],
        )
        observed[ids_dest] = True

        # The `excluded_too_steep` indices are added to retain the parts of the wavefront that haven't fully propagated yet
        # TODO: move these excluded indices to a stack and only access them when we know their level is reached to avoid
        #       repeated failed propagation attempts and save CPU time.
        border_last = np.concatenate(
            [
                np.unique(
                    np.ascontiguousarray(excluded_too_steep[..., 1]).reshape(-1).view(dtype='i8,i8,i8')
                ).view("int64").reshape(-1, 3),
                border_next_with_mapback[..., 0],
            ], axis=0
        )
        if progress_bar:
            pbar.set_description(f"(wavefront size {len(border_last)}; current_level {current_level:.4f})")
            pbar.update(len(border_next_with_mapback))
    if progress_bar:
        pbar.close()

    assert observed.all()
    return levels - levels[seed]
