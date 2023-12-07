from typing import Optional, Tuple, List, Callable, Literal

import numpy as np
from tqdm.auto import tqdm

def wave_search(
    potential: np.ndarray,
    seed: Tuple[int, int, int],
    minimal_lvl_increase: float = 0.1,
    progress_bar: bool = True,
    fill_wavefront_ids_list: Optional[list] = None,
    early_stop: Literal["any_face", "faces", "off"] = "off",
    early_stop_faces: Optional[List[int]] = None,
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

    stop_condition_met = False
    if early_stop in ("any_face", "faces"):
        check_stop_condition = True
        faces_pattern = np.array([True] * 3)
        if early_stop == "faces":
            faces_pattern = ~faces_pattern
            for i in early_stop_faces:
                faces_pattern[i] = True
    elif early_stop == "off":
        assert early_stop_faces is None
        check_stop_condition = False
    else:
        raise NotImplementedError(early_stop)

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
    wf_updated = True
    border_last_min_level = np.array([-np.inf], dtype=float)

    if progress_bar:
        pbar = tqdm(total=observed.size)
        pbar.update(observed.sum())
    while True:
        if fill_wavefront_ids_list is not None and wf_updated:
            fill_wavefront_ids_list.append(border_last)
            wf_updated = False

        # Increase the level if we know for sure that it's required for propagation. Previously unobserved
        # locations are added with min_level=-inf, so this can only happen when no new locations were added.
        if (border_last_min_level > current_level).all() and not stop_condition_met:
            current_level = max(border_last_min_level.min(), current_level + minimal_lvl_increase)

        # for optimization purposes, only propagate from locations for which current level is sufficient
        border_last_selection = border_last_min_level <= current_level

        # make single step
        border_next_with_mapback = (
            border_last[border_last_selection, None, :, None] + deltas_and_zeros[None, :, :, :]
        ).reshape(-1, 3, 2)

        # remove out of bounds hops
        border_next_with_mapback = border_next_with_mapback[
            (border_next_with_mapback[..., 0] >= 0).all(axis=-1)
            & (border_next_with_mapback[..., 0] < [shape]).all(axis=-1)
        ]

        # remove the nodes we've been in
        border_next_with_mapback = border_next_with_mapback[
            ~observed[_index_with(border_next_with_mapback[..., 0])]
        ]
        if not len(border_next_with_mapback):
            # There's nowhere to propagate at this level. If level increase would allow for further propagation,
            # do it and restart current iteration:
            if not border_last_selection.all() and not stop_condition_met:
                current_level = max(
                    border_last_min_level[~border_last_selection].min(),
                    current_level + minimal_lvl_increase,
                )
                continue
            # otherwise, we are done:
            break

        # Now we want to adjust the wave propagation to only happen in the direction of
        # smallest ascent of the potential. In case current level is incufficient, increase it and restart iteration:
        values_dest = potential[_index_with(border_next_with_mapback[..., 0])]
        if (values_dest > current_level).all():
            if not stop_condition_met:
                current_level = max(
                    min(
                        values_dest.min(),
                        border_last_min_level[~border_last_selection].min() if not border_last_selection.all() else np.inf,
                    ),
                    current_level + minimal_lvl_increase,
                )
                continue
            else:
                break

        ascent_selection = values_dest <= current_level
        # keep track of the excluded stuff to process later:
        excluded_too_steep = border_next_with_mapback[~ascent_selection]
        excluded_too_steep_values = values_dest[~ascent_selection]
        # and only process the remaining part now:
        border_next_with_mapback = border_next_with_mapback[ascent_selection]

        # So far `border_next_with_mapback` contains all the hops at current steps,
        # but some of them are reduntant (multiple hops to a single position). We need to
        # group the hops by destination and only select the ones comming from smallest observed level.
        border_next_with_mapback = np.array(_group_by_ids_and_aggregate(
            border_next_with_mapback[..., 0],
            [border_next_with_mapback],
            lambda group: group[levels[_index_with(group[..., 1])].argmin()]
        ))

        # Now we update the levels by max(potential here, smallest neighbor level)
        ids_src = _index_with(border_next_with_mapback[..., 1])
        ids_dest = _index_with(border_next_with_mapback[..., 0])
        levels[ids_dest] = np.maximum(
            levels[ids_src],
            potential[ids_dest],
        )
        observed[ids_dest] = True

        # The `excluded_too_steep` source ids will be put in `border_last` to retain the parts of the
        # wavefront that haven't fully propagated yet. For optimization purposes, we keep track of smallest
        # level value needed to step out of a given source location.
        if len(excluded_too_steep):
            excl_ids, excl_vals = zip(*_group_by_ids_and_aggregate(
                excluded_too_steep[..., 1],
                [excluded_too_steep, excluded_too_steep_values],
                _aggregate_min_paired,
            ))
            excl_ids = np.array(excl_ids)
            excl_vals = np.array(excl_vals)
        else:
            excl_ids = np.array([], dtype="int64").reshape(0, 3, 2)
            excl_vals = np.array([], dtype=float)

        border_next_vals = np.ones(dtype=float, shape=len(border_next_with_mapback)) * (-np.inf)

        # update border_last and the corresponding minimal level values
        border_last = np.concatenate(
            [border_last[~border_last_selection], excl_ids[..., 1], border_next_with_mapback[..., 0]], axis=0
        )
        border_last_min_level = np.concatenate(
            [border_last_min_level[~border_last_selection], excl_vals, border_next_vals], axis=0
        )
        wf_updated = True
        if check_stop_condition:
            if not stop_condition_met:
                if (
                    ((border_last == 0) | ((border_last + 1) == shape)) & faces_pattern[None, :]
                ).any():
                    print("Early stoping")
                    stop_condition_met = True

        if progress_bar:
            pbar.set_description(f"(wavefront size {len(border_last)}; current_level {current_level:.4f})")
            pbar.update(len(border_next_with_mapback))
    if progress_bar:
        pbar.close()

    if stop_condition_met:
        if not observed.all():
            levels[~observed] = potential.max()
    else:
        assert observed.all()
    return levels - levels[seed]

def _group_by_ids_and_aggregate(
    ids: np.ndarray,
    arrays: List[np.ndarray],
    aggregation_func: Callable,
) -> List:
    tags = np.ascontiguousarray(ids).reshape(-1).view(dtype='i8,i8,i8')
    assert len(ids) == len(tags)

    # Grouping trick borrowed from https://stackoverflow.com/a/43094244/3801744
    sort_ids = np.argsort(tags)
    tags = tags[sort_ids]
    sorted_arrays = [arr[sort_ids] for arr in arrays]
    (_, group_ids) = np.unique(tags, return_index=True)
    array_groups = zip(*[np.split(arr, group_ids[1:], axis=0) for arr in sorted_arrays])
    return [aggregation_func(*g) for g in array_groups]

def _aggregate_min_paired(arr1, arr2_argmin_of):
    imin = arr2_argmin_of.argmin()
    return arr1[imin], arr2_argmin_of[imin]

def _index_with(ids: np.ndarray) -> Tuple[np.ndarray]:
    return tuple(ids.T)
