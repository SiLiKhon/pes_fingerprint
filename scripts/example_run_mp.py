from typing import Dict, List

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pes_fingerprint.pipelines import process_structure
from .query_mp import query_mp

def process_mp_structure(mp_doc: dict, **kwargs) -> Dict[str, float]:
    atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(mp_doc["structure"]))
    return process_structure(atoms, **kwargs)

def process_mp_parallel(
    mp_docs: List[dict],
    num_jobs: int,
    **kwargs,
) -> pd.DataFrame:
    if num_jobs != 1:
        predictions = Parallel(n_jobs=num_jobs, verbose=5)(
            delayed(process_mp_structure)(doc, **kwargs) for doc in mp_docs
        )
    elif num_jobs == 1:
        predictions = [process_mp_structure(doc, **kwargs) for doc in tqdm(mp_docs)]

    mpids = [doc["material_id"] for doc in mp_docs]
    return pd.DataFrame(predictions, index=pd.Series(mpids, name="mpid"))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--first", "-f", type=int, required=True)
    parser.add_argument("--last-inclusive", "-l", type=int, required=True)
    parser.add_argument("--num-jobs", "-n", type=int, required=True)
    parser.add_argument("--export-to-file", "-o", type=str, default=None)
    args = parser.parse_args()
    assert args.last_inclusive >= args.first >= 0

    docs = query_mp()[args.first: args.last_inclusive + 1]

    predictions = process_mp_parallel(docs, num_jobs=args.num_jobs)
    print(
        predictions[
            ["mpe", "fv_0p5_connected_union", "fv_0p5_disconnected_union", "Xi"]
        ].round(3).to_markdown(),
    )
    if args.export_to_file is not None:
        predictions.to_csv(args.export_to_file, index=True)
