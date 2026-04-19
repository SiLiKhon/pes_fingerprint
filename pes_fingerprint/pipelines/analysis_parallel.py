import warnings

import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from ase import Atoms

from .analysis_utils import process_structure


def _process_structure_noexcept(*args, **kwargs) -> dict[str, float]:
    try:
        return process_structure(*args, **kwargs)
    except:
        warnings.warn(f"process_structure failed for inputs: {args}, {kwargs}")
        return {}


def process_structures(
    atoms_list: list[Atoms],
    parallel_num_jobs: int,
    parallel_verbose: int = 5,
    parallel_other_kwargs: dict | None = None,
    fail_on_except: bool = False,
    **kwargs,
) -> pd.DataFrame:
    if parallel_other_kwargs is None:
        parallel_other_kwargs = {}

    func = process_structure if fail_on_except else _process_structure_noexcept

    if parallel_num_jobs != 1:
        parallel = Parallel(
            n_jobs=parallel_num_jobs,
            verbose=parallel_verbose,
            **parallel_other_kwargs,
        )
        predictions = parallel(
            delayed(func)(atoms, **kwargs) for atoms in atoms_list
        )
    elif parallel_num_jobs == 1:
        predictions = [func(atoms, **kwargs) for atoms in tqdm(atoms_list)]

    return pd.DataFrame(predictions)
