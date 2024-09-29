from typing import Any, Dict, List, Literal, Optional, Tuple
from itertools import chain
import inspect

import numpy as np
from ase import Atoms

from .mpe import calculate_mpe
from .levels import calculate_levels


def shift_map(m: np.ndarray, shift: Tuple[int, int, int], last_pix_overlap: bool = True) -> np.ndarray:
    """
    Shift a 3d map by padding it with values wrapped around in the direction of the shift and chopping
    off the values at the opposite side.

    Parameters
    ----------
    m: np.ndarray
        the 3d numpy array to shift
    shift: Tuple[int, int, int]
        len-3 tuple of integer shift values
    last_pix_overlap: bool
        if True (default), then the input is expected to satisfy `m[0] == m[-1]` (and similar for the other
        two axes); this property is also preserved in the output

    Returns
    -------
    np.ndarray
        shifted map
    """
    assert m.ndim == 3
    assert len(shift) == 3
    if last_pix_overlap:
        assert (m[0] == m[-1]).all(), np.abs(m[0] - m[-1]).max()
        assert (m[:, 0] == m[:, -1]).all(), np.abs(m[:, 0] - m[:, -1]).max()
        assert (m[:, :, 0] == m[:, :, -1]).all(), np.abs(m[:, :, 0] - m[:, :, -1]).max()

    result = np.pad(
        m[:-1, :-1, :-1] if last_pix_overlap else m,
        pad_width=[
            (max(-shift[0], 0), max(shift[0], 0)),
            (max(-shift[1], 0), max(shift[1], 0)),
            (max(-shift[2], 0), max(shift[2], 0)),
        ],
        mode="wrap",
    )
    result = result[
        max(shift[0], 0): result.shape[0] + min(shift[0], 0),
        max(shift[1], 0): result.shape[1] + min(shift[1], 0),
        max(shift[2], 0): result.shape[2] + min(shift[2], 0),
    ]
    if last_pix_overlap:
        result = np.pad(result, [(0, 1), (0, 1), (0, 1)], mode="wrap")
        assert (result[0] == result[-1]).all()
        assert (result[:, 0] == result[:, -1]).all()
        assert (result[:, :, 0] == result[:, :, -1]).all()
    assert result.shape == m.shape, (result.shape, m.shape)
    return result

def _get_equiv_faces_edges_verts(shape: Tuple[int, int, int]) -> List[Tuple[np.ndarray]]:
    ijk = np.fromfunction(
        lambda *ijk: np.stack(ijk, axis=-1),
        shape=shape,
    )

    fev_mask = ((ijk == 0) | (ijk + 1 == shape))

    faces = []
    edges = []

    face_sel = fev_mask.sum(axis=-1) == 1
    edge_sel = fev_mask.sum(axis=-1) == 2
    vert_sel = fev_mask.sum(axis=-1) == 3
    for ax in range(3):
        faces.append((
            face_sel & (ijk[..., ax] == 0),
            face_sel & (ijk[..., ax] == shape[ax] - 1),
        ))

        other_ax = np.arange(3) != ax
        edges.append((
            (edge_sel[..., None] & (ijk[..., other_ax] == (0, 0) * (np.array(shape) - 1)[other_ax])).all(axis=-1),
            (edge_sel[..., None] & (ijk[..., other_ax] == (0, 1) * (np.array(shape) - 1)[other_ax])).all(axis=-1),
            (edge_sel[..., None] & (ijk[..., other_ax] == (1, 1) * (np.array(shape) - 1)[other_ax])).all(axis=-1),
            (edge_sel[..., None] & (ijk[..., other_ax] == (1, 0) * (np.array(shape) - 1)[other_ax])).all(axis=-1),
        ))

    verts = [tuple((ijk == v).all(axis=-1) for v in ijk[vert_sel])]

    return faces + edges + verts

def make_mask(cell: np.ndarray, wrapped_shape: Tuple[int, int, int], thr: float) -> np.ndarray:
    """
    Assuming the ion initial position is in the center of the cell, this function creates a
    boolean mask of all grid locations that are at least `thr` A away from center.
    """

    def _unwrap_assert_fits(dim: int) -> int:
        result = (dim - 1) / 2
        result_int = int(result)
        assert result_int == result
        assert result_int > 0
        return result_int

    base_shape = tuple(
        _unwrap_assert_fits(dim)
        for dim in wrapped_shape
    )

    assert cell.shape == (3, 3)
    step_vecs = cell / np.array(base_shape)[:, None]

    ijk = np.stack(np.meshgrid(*[np.arange(-dim, dim + 1) for dim in base_shape], indexing="ij"), axis=-1)
    assert ijk.shape == wrapped_shape + (3,)

    xyz = ijk @ step_vecs
    return np.linalg.norm(xyz, axis=-1) >= thr

def wrap_mask_2x2x2(mask: np.ndarray) -> np.ndarray:
    shifts = np.fromfunction(
        lambda *xyz: np.stack(xyz, axis=-1),
        shape=(2, 2, 2),
    ).reshape(-1, 3).astype(int)
    half_shape = (np.array(mask.shape) - 1) / 2
    half_shape_int = half_shape.astype(int)
    np.testing.assert_allclose(half_shape - half_shape_int, 0)
    return np.all(
        [shift_map(mask, tuple(shift)) for shift in half_shape_int[None, :] * shifts],
        axis=0
    )

def calculate_free_volume(
    pes_or_levels: List[np.ndarray],
    threshold: float,
    aggregation: Literal["sum", "union"],
    masks: Optional[List[np.ndarray]] = None,
) -> float:
    """
    Calculate the "free volume" feature from a set of PES or level maps for a structure. That is a
    fractional volume of PES regions below a given threshold.

    Parameters
    ----------
    pes_or_levels: List[np.ndarray]
        3d map of PES (or levels, for the "connected" case) values over a grid of points

    threshold: float
        threshold value to calculate the "free volume" at

    aggregation: Literal["sum", "union"]
        how to aggregate over different maps (corresponding to different mobile ions); "sum" means
        the free volume values are summed over the mobile ions; "union" means that the union of regions
        is constructed and the fractional volume of the result is reported

    masks: Optional[List[np.ndarray]] = None,
        boolean arrays to mask out regions of 3d maps (`True` = keep, `False` = discard); note that masks
        are used differently in different aggregation modes (see code)

    Returns
    -------
    float:
        the value of the "free volume" feature
    """
    assert len(set(map_i.shape for map_i in pes_or_levels)) == 1
    if masks is None:
        masks = [np.ones(pes_or_levels[0].shape, dtype=bool)] * len(pes_or_levels)

    assert len(masks) == len(pes_or_levels)
    assert len(set(mask_i.shape for mask_i in masks)) == 1
    assert masks[0].shape == pes_or_levels[0].shape

    for m in chain(pes_or_levels, masks):
        assert (m[0] == m[-1]).all()
        assert (m[:, 0] == m[:, -1]).all()
        assert (m[:, :, 0] == m[:, :, -1]).all()
    pes_or_levels = [m[:-1, :-1, :-1] for m in pes_or_levels]
    masks = [m[:-1, :-1, :-1] for m in masks]

    volumes = [
        map_i < threshold
        for map_i in pes_or_levels
    ]
    result = -1.0
    if aggregation == "sum":
        result = sum(v[m].mean() for v, m in zip(volumes, masks))
    elif aggregation == "union":
        result = np.stack([v & m for v, m in zip(volumes, masks)], axis=0).any(axis=0).mean()
    else:
        raise NotImplementedError(f"Unknown aggregation method: {aggregation}")

    return result

def barrier_robust(
    levels: np.ndarray,
    percentile: float,
) -> float:
    assert levels.ndim == 3
    return min(
        np.quantile(levels[0], percentile), np.quantile(levels[-1], percentile),
        np.quantile(levels[:, 0], percentile), np.quantile(levels[:, -1], percentile),
        np.quantile(levels[..., 0], percentile), np.quantile(levels[..., -1], percentile),
    )

DEFAULT_FV_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

def process_structure(
    atoms: Atoms, *,
    mobile_species: str = "Li",
    free_volume_levels: Tuple[float] = DEFAULT_FV_LEVELS,
    fvl_smearing: List[float] = [-0.04, -0.02, 0.0, 0.02, 0.04],
    mpe_params: Optional[Dict[str, Any]] = None,
    levels_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Calculate features for a structure.

    Parameters
    ----------
    atoms: Atoms
        structure to calculate features for

    free_volume_levels: Tuple[float]
        "free volume" threshold levels to calculate these features at

    mobile_species: str
        type of atom under consideration

    fvl_smearing: List[float]
        each "free volume" is calculated as average over smeared levels: `[level + delta for delta in fvl_smearing]`

    mpe_params: Optional[Dict[str, Any]]
        parameters passed to the `calculate_mpe` call

    levels_params: Optional[Dict[str, Any]]
        parameters passed to the `calculate_levels` call

    Returns
    -------
    Dict[str, float]
        dictionary of feature values
    """
    if mpe_params is None:
        mpe_params = {}
    if levels_params is None:
        levels_params = {}
    cell = atoms.cell.array
    cell_volume = atoms.cell.volume
    assert cell_volume > 0

    early_stop_level = levels_params.get(
        "early_stop_level",
        inspect.signature(calculate_levels).parameters["early_stop_level"].default,
    )
    assert early_stop_level is not inspect.Parameter.empty
    assert max(free_volume_levels) <= early_stop_level

    (mobile_ids,) = np.where([el == mobile_species for el in atoms.symbols])
    mobile_ids = [int(i) for i in mobile_ids]
    assert len(mobile_ids) > 0
    _filter = lambda x: {
        k: x[k] for k in [
            "pes", "levels", "imin", "original_imin", "original_shape", "barrier"
        ]
    }
    barrier_preds = [
        _filter(
            calculate_mpe(atoms=atoms, mobile_id=mobile_id, **mpe_params)
        ) for mobile_id in mobile_ids
    ]
    level_preds = [
        calculate_levels(atoms=atoms, mobile_id=mobile_id, **levels_params, mpe_params=mpe_params)
        for mobile_id in mobile_ids
    ]

    # some simple checks:
    grid_size = tuple(int((dim - 1) / 2) for dim in barrier_preds[0]["pes"].shape)
    grid_size_wrapped = tuple(dim * 2 + 1 for dim in grid_size)
    assert grid_size_wrapped == barrier_preds[0]["pes"].shape
    cell_mask = make_mask(cell, grid_size_wrapped, thr=1.5)
    cell_mask_wrapped_2x2x2 = wrap_mask_2x2x2(cell_mask)
    assert len(barrier_preds) == len(level_preds)
    for bp, lvl in zip(barrier_preds, level_preds):
        common_thr = min(bp["barrier"], early_stop_level)
        _mask = (bp["levels"] < common_thr) & (lvl < common_thr)
        assert grid_size == tuple(int((dim - 1) / 2) for dim in bp["pes"].shape)
        assert bp["imin"] == grid_size
        np.testing.assert_allclose(
            bp["levels"][_mask], lvl[_mask],
        )

    results = {}
    results["cell_volume"] = cell_volume
    shifts = [  # shifts that align different PES/level maps
        tuple(np.array(bp["original_shape"]) - bp["original_imin"])
        for bp in barrier_preds
    ]
    shifted_pes = [
        shift_map(bp["pes"] - bp["pes"].min(), shift)
        for bp, shift in zip(barrier_preds, shifts)
    ]
    for bp, sh_pes_i in zip(barrier_preds, shifted_pes):  # validating shifts
        assert np.unravel_index(sh_pes_i.argmin(), bp["pes"].shape) == bp["original_imin"]
        fv_orig = np.array([calculate_free_volume([bp["pes"] - bp["pes"].min()], fvl, aggregation="union") for fvl in free_volume_levels])
        fv_new = np.array([calculate_free_volume([sh_pes_i], fvl, aggregation="sum") for fvl in free_volume_levels])
        assert (fv_orig == fv_new).all(), "\n" + " ".join(f"{a:10.6f}" for a in fv_orig) + "\n" + " ".join(f"{a:10.6f}" for a in fv_new)

    shifted_cell_masks = [
        shift_map(cell_mask, shift)
        for shift in shifts
    ]
    shifted_cell_masks_wrapped_2x2x2 = [
        shift_map(cell_mask_wrapped_2x2x2, shift)
        for shift in shifts
    ]

    # For shifting levels, need to make sure they wrap around the last pixel
    # (see `last_pix_overlap` parameter from `shift_map` func). As original
    # levels may not hold this property, we enforce it by replicating the smallest
    # face/edge/vertex values
    shifted_levels = []
    equiv_groups = _get_equiv_faces_edges_verts(grid_size_wrapped)
    for shift, lvl in zip(shifts, level_preds):
        lvl_tmp = lvl.copy()
        for eq_g in equiv_groups:
            min_vals = np.stack([lvl_tmp[g_i] for g_i in eq_g], axis=0).min(axis=0)
            for g_i in eq_g:
                lvl_tmp[g_i] = min_vals
        shifted_levels.append(shift_map(lvl_tmp, shift))

    def get_smeared_calculate_free_volume(thr):
        def _smeared_func(*args, **kwargs):
            results = [
                calculate_free_volume(*args, **kwargs, threshold=thr + delta)
                for delta in fvl_smearing
            ]
            return np.mean(results)
        return _smeared_func

    # calculating the features
    for fvl in free_volume_levels:
        calc_fv = get_smeared_calculate_free_volume(fvl)
        for masked in ["", "masked1p5_", "Wmasked1p5_"]:
            key = f"{masked}fv_{fvl}".replace(".", "p")

            current_masks = None
            if masked:
                if masked == "masked1p5_":
                    current_masks = shifted_cell_masks
                elif masked == "Wmasked1p5_":
                    current_masks = shifted_cell_masks_wrapped_2x2x2
                else:
                    raise NotImplementedError(masked)
            results[key + "_disconnected"] = calc_fv(
                shifted_pes, aggregation="sum", masks=current_masks
            )
            results[key + "_disconnected_rel"] = results[key + "_disconnected"] / cell_volume
            results[key + "_connected"] = calc_fv(
                shifted_levels, aggregation="sum", masks=current_masks
            )
            results[key + "_connected_rel"] = results[key + "_connected"] / cell_volume
            if not masked:
                results[key + "_WARNlowPES"] = any(
                    pes_i.max() < fvl for pes_i in shifted_pes
                )
            results[key + "_disconnected_union"] = calc_fv(
                shifted_pes, aggregation="union", masks=current_masks
            )
            results[key + "_connected_union"] = calc_fv(
                shifted_levels, aggregation="union", masks=current_masks
            )
    results["mpe"] = min(bp["barrier"] for bp in barrier_preds)
    results["mpe_robust_0p03"] = min(
        barrier_robust(lvl, percentile=0.03)
        for lvl in level_preds
    )
    results["mpe_robust_0p05"] = min(
        barrier_robust(lvl, percentile=0.05)
        for lvl in level_preds
    )
    results["mpe_robust_0p10"] = min(
        barrier_robust(lvl, percentile=0.10)
        for lvl in level_preds
    )

    if (
        ("fv_0p5_connected_union" in results) and
        ("fv_0p5_disconnected_union" in results)
    ):
        results["Xi"] = (
            1.0 / (1.0 + np.exp(-(np.log10(results["fv_0p5_connected_union"]) + 2.0) * 10))
            / (1.0 + np.exp(-(np.log10(results["fv_0p5_disconnected_union"]) + 1.15) * 10))
        )
    return results
