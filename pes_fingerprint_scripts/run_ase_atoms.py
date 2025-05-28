from argparse import ArgumentParser
import json

from ase.io import read
import pandas as pd

from pes_fingerprint.pipelines import process_structures


def parse_args_and_run() -> pd.DataFrame:
    parser = ArgumentParser()
    parser.add_argument("input_ase_traj", type=str)
    parser.add_argument("--num-jobs", "-n", type=int, required=True)
    parser.add_argument("--export-to-file", "-o", type=str, required=True)
    parser.add_argument("--kwargs-json", type=str, default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.kwargs_json is not None:
        kwargs = json.loads(args.kwargs_json)

    atoms_list = read(args.input_ase_traj, index=slice(None))
    assert isinstance(atoms_list, list)

    predictions = process_structures(
        atoms_list, parallel_num_jobs=args.num_jobs, **kwargs
    )
    print(
        predictions[
            ["mpe", "fv_0p5_connected_union", "fv_0p5_disconnected_union", "Xi"]
        ].round(3).to_markdown(),
    )
    predictions.to_csv(args.export_to_file, index=False)
    return predictions


def main() -> int:
    parse_args_and_run()
    return 0


if __name__ == "__main__":
    predictions = parse_args_and_run()
