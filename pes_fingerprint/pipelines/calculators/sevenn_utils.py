from collections import OrderedDict

import torch
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from torch_geometric.loader import DataLoader
import sevenn.util
from sevenn.train.dataload import graph_build, _set_atoms_y
from sevenn.train.dataset import AtomGraphDataset


def _assign_dummy_y(atoms: Atoms) -> Atoms:
    dummy = {"energy": np.nan, "free_energy": np.nan}
    dummy["forces"] = np.full((len(atoms), 3), np.nan)
    dummy["stress"] = np.full((6,), np.nan)
    return SinglePointCalculator(atoms.copy(), **dummy).get_atoms()


class SevenNetBatchPES:
    def __init__(
        self,
        model_name: str = "7net-0",
        num_cores: int | None = None,
        device: torch.device | None = None,
    ):
        self.num_cores = num_cores
        self.device = device or torch.device("cpu")
        path = sevenn.util.pretrained_name_to_path(model_name)
        self.sevenn_model_src, self.sevenn_config = sevenn.util.model_from_checkpoint(path)

        layers = OrderedDict(self.sevenn_model_src.named_children())
        layers.pop("force_output")
        self.sevenn_model = self.sevenn_model_src.__class__(
            layers,
            cutoff=self.sevenn_model_src.cutoff,
            type_map=self.sevenn_model_src.type_map,
            eval_type_map=self.sevenn_model_src.eval_type_map,
            data_key_atomic_numbers=self.sevenn_model_src.key_atomic_numbers,
            data_key_node_feature=self.sevenn_model_src.key_node_feature,
            data_key_grad=self.sevenn_model_src.key_grad,
        ).to(self.device)


    def __call__(
        self,
        atoms_list: list[Atoms],
        batch_size: int,
    ) -> np.ndarray:
        atoms_list = _set_atoms_y([_assign_dummy_y(ats) for ats in atoms_list])
        sevenn_data_list = graph_build(
            atoms_list,
            self.sevenn_config["cutoff"],
            num_cores=self.num_cores,
            y_from_calc=False,
        )
        sevenn_inference_set = AtomGraphDataset(
            sevenn_data_list, self.sevenn_config["cutoff"]
        )
        sevenn_inference_set.x_to_one_hot_idx(self.sevenn_config["_type_map"])
        sevenn_inference_set.toggle_requires_grad_of_data(sevenn._keys.POS, False)
        sevenn_infer_list = sevenn_inference_set.to_list()

        sevenn_data = DataLoader(
            sevenn_infer_list, batch_size=batch_size, shuffle=False
        )

        energies = []

        with torch.no_grad():
            for batch in sevenn_data:
                batch = batch.to(self.device)
                output = self.sevenn_model(batch)
                energies.append(output.inferred_total_energy.detach().cpu().numpy())

        return np.concatenate(energies, axis=0)
