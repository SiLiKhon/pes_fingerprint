import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from ase import Atoms

from .analysis_utils import process_structure

def process_structures(
    atoms_list: list[Atoms],
    parallel_num_jobs: int,
    parallel_verbose: int = 5,
    parallel_other_kwargs: dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    if parallel_other_kwargs is None:
        parallel_other_kwargs = {}

    if parallel_num_jobs != 1:
        parallel = Parallel(
            n_jobs=parallel_num_jobs,
            verbose=parallel_verbose,
            **parallel_other_kwargs,
        )
        predictions = parallel(
            delayed(process_structure)(atoms, **kwargs) for atoms in atoms_list
        )
    elif parallel_num_jobs == 1:
        predictions = [process_structure(atoms, **kwargs) for atoms in tqdm(atoms_list)]

    return pd.DataFrame(predictions)
