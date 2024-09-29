from typing import Dict

import numpy as np
from tqdm.auto import tqdm
from ase import Atoms
from m3gnet.models import Potential
from m3gnet.graph import MaterialGraphBatch


def setup_tensorflow(
    gpu_memory_growth: bool = True,
    disable_tensor_float_32: bool = True,
):
    import tensorflow as tf
    if gpu_memory_growth:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    if disable_tensor_float_32:
        tf.config.experimental.enable_tensor_float_32_execution(False)

def predict_in_batches(
    potential: Potential,
    structures: Atoms,
    batch_size: int = 32,
    verbose=True,
    include_stresses: bool = True,
) -> Dict[str, np.ndarray]:
    progress = tqdm if verbose else (lambda x, **kwargs: x)
    mgb = MaterialGraphBatch([
        potential.model.graph_converter(s) for s in progress(structures, desc="Preparing graph batches")
    ], batch_size=batch_size, shuffle=False)
    mgb.targets = None

    sum_sizes = sum(len(s) for s in structures)
    energies = np.empty(shape=(len(structures),), dtype=structures[0].positions.dtype)
    forces = np.empty(shape=(sum_sizes, 3), dtype=structures[0].positions.dtype)
    result = dict(energies=energies, forces=forces)
    if include_stresses:
        stresses = np.empty(shape=(len(structures), 3, 3), dtype=structures[0].positions.dtype)
        result["stresses"] = stresses

    predict_call_kwargs = dict(include_stresses=include_stresses)

    last_filled_es = 0
    last_filled_f = 0
    for stru_batch in progress(mgb, desc="Running over batches"):
        preds = potential.get_efs_tensor(stru_batch.as_tf().as_list(), **predict_call_kwargs)
        (pred_e,) = preds[0].numpy().T
        pred_f = preds[1].numpy()
        if include_stresses:
            pred_s = preds[2].numpy()
        energies[last_filled_es: last_filled_es + batch_size] = pred_e
        forces[last_filled_f: last_filled_f + len(pred_f)] = pred_f
        if include_stresses:
            stresses[last_filled_es: last_filled_es + batch_size] = pred_s
        last_filled_es += len(pred_e)
        last_filled_f += len(pred_f)

    return result
