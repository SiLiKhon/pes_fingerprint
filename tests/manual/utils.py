from typing import List, Tuple

import numpy as np
from skimage.measure import marching_cubes
import plotly.graph_objects as go


def visualize_wavefront(
    wf: List[np.ndarray],
    target_shape: Tuple[int, int, int],
    max_num_frames: int = 200,
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
    return _visualize_wavefront(wavefront)


def _visualize_wavefront(wf: np.ndarray) -> go.Figure:
    assert wf.ndim == 4
    assert wf.dtype == "bool"

    fig_data = []
    for frame in wf:
        verts, faces, _, _ = marching_cubes(frame, level=0.5, allow_degenerate=False)
        fig_data.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                colorscale=[[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'magenta']],
                intensity=np.linspace(0, 1, len(faces)),
                intensitymode='cell',
            ),
        )

    fig = go.Figure(
        data=fig_data[:1],
        frames=[
            go.Frame(data=[fig_data_i], name=str(i))
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
            xaxis=dict(nticks=10, range=(0, wf.shape[1])),
            yaxis=dict(nticks=10, range=(0, wf.shape[2])),
            zaxis=dict(nticks=10, range=(0, wf.shape[3])),
            aspectmode='manual',
            aspectratio=dict(
                x=wf.shape[1],
                y=wf.shape[2],
                z=wf.shape[3],
            ),
        ),
    )
    return fig
