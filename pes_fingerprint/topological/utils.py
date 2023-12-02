from typing import List, Tuple, Optional

import numpy as np
from skimage.measure import marching_cubes
import plotly.graph_objects as go


def visualize_wavefront(
    wf: List[np.ndarray],
    target_shape: Tuple[int, int, int],
    max_num_frames: int = 200,
    unit_cell: Optional[np.ndarray] = None,
    energy_levels: Optional[np.ndarray] = None,
) -> go.Figure:
    def _make_dense_frame(ids: np.ndarray) -> np.ndarray:
        result = np.zeros(shape=target_shape, dtype=bool)
        result[tuple(ids.T)] = True
        return result

    wavefront = np.array([_make_dense_frame(ids) for ids in wf])
    wavefront = wavefront.cumsum(axis=0).astype(bool)
    if len(wavefront) > max_num_frames:
        stepsize = int(np.ceil(len(wavefront) / max_num_frames))
        wavefront = wavefront[::stepsize]
    if wavefront[-1].all():
        wavefront = wavefront[:-1]
    return _visualize_wavefront(wavefront, unit_cell=unit_cell, energy_levels=energy_levels)


def _visualize_wavefront(
    wf: np.ndarray,
    unit_cell: Optional[np.ndarray],
    energy_levels: Optional[np.ndarray],
) -> go.Figure:
    assert wf.ndim == 4
    assert wf.dtype == "bool"

    fig_data = []
    global_data = []

    ax_limits = wf.shape[1:]
    if unit_cell is not None:
        assert unit_cell.shape == (3, 3)
        points = np.array([
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0],
        ]) @ unit_cell
        global_data.append(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                line=dict(color="black", width=2, dash='dash'),
                marker=dict(size=0.01, opacity=0.01),
            )
        )
        ax_limits = points.max(axis=0)

    for frame in wf:
        verts, faces, _, _ = marching_cubes(frame, level=0.5, allow_degenerate=False)
        if energy_levels is not None:
            intensity=energy_levels[tuple(np.floor(verts).astype(int).T)]
        else:
            intensity=np.linspace(0, 1, len(faces))
        if unit_cell is not None:
            verts = (verts / frame.shape) @ unit_cell
        fig_data.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                colorscale=[[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'magenta']],
                intensity=intensity,
                intensitymode='cell' if energy_levels is None else 'vertex',
            ),
        )

    fig = go.Figure(
        data=fig_data[:1] + global_data,
        frames=[
            go.Frame(data=[fig_data_i] + global_data, name=str(i))
            for i, fig_data_i in enumerate(fig_data)
        ],
    )
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 20, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    )
                ],
            ),
        ],
        sliders = [
                {
                    "steps": [
                        {
                            "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "fromcurrent": True, "transition": {"duration": 0}}],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ],
        scene=dict(
            xaxis=dict(nticks=10, range=(0, ax_limits[0])),
            yaxis=dict(nticks=10, range=(0, ax_limits[1])),
            zaxis=dict(nticks=10, range=(0, ax_limits[2])),
            aspectmode='manual',
            aspectratio=dict(
                x=ax_limits[0],
                y=ax_limits[1],
                z=ax_limits[2],
            ),
        ),
        scene_camera=dict(eye=dict(x=ax_limits[0] * 2, y=ax_limits[1] * 2, z=ax_limits[2] * 2)),
    )
    return fig
