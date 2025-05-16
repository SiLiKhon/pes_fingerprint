import json

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from pes_fingerprint.pipelines import process_structures
from .query_mp import query_mp


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--first", "-f", type=int, required=True)
    parser.add_argument("--last-inclusive", "-l", type=int, required=True)
    parser.add_argument("--num-jobs", "-n", type=int, required=True)
    parser.add_argument("--export-to-file", "-o", type=str, default=None)
    parser.add_argument("--kwargs-json", type=str, default=None)
    args = parser.parse_args()
    assert args.last_inclusive >= args.first >= 0
    kwargs = {}
    if args.kwargs_json is not None:
        kwargs = json.loads(args.kwargs_json)

    docs = query_mp()[args.first: args.last_inclusive + 1]
    atoms_list = list(
        map(
            lambda doc: AseAtomsAdaptor.get_atoms(Structure.from_dict(doc["structure"])),
            docs,
        )
    )
    mpids = [doc["material_id"] for doc in docs]

    predictions = process_structures(
        atoms_list, parallel_num_jobs=args.num_jobs, **kwargs
    )
    predictions.index = pd.Series(mpids, name="mpid")
    print(
        predictions[
            ["mpe", "fv_0p5_connected_union", "fv_0p5_disconnected_union", "Xi"]
        ].round(3).to_markdown(),
    )
    if args.export_to_file is not None:
        predictions.to_csv(args.export_to_file, index=True)
