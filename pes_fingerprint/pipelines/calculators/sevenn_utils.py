from collections import OrderedDict

import torch
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from torch_geometric.loader import DataLoader
import sevenn.util
from sevenn.train.dataload import graph_build, _set_atoms_y
from sevenn.train.dataset import AtomGraphDataset
from tqdm.auto import tqdm


def _assign_dummy_y(atoms: Atoms) -> Atoms:
    dummy = {"energy": np.nan, "free_energy": np.nan}
    dummy["forces"] = np.full((len(atoms), 3), np.nan)
    dummy["stress"] = np.full((6,), np.nan)
    return SinglePointCalculator(atoms.copy(), **dummy).get_atoms()


class SevenNetBatchPES:
    mem_estimate_safety_factor = 2.0

    def __init__(
        self,
        model_name: str = "7net-0",
        num_cores: int | None = None,
        device: torch.device | str | None = None,
    ):
        if isinstance(device, str):
            device = torch.device(device)
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
        target_gpu_memory_mb: float,
        base_batch_size: int = 200,
    ) -> np.ndarray:
        batch_size = base_batch_size
        if self.device.type == "cuda":
            mb_per_structure = self.estimate_memory_mb_per_structure(atoms_list[0])
            batch_size = int(
                target_gpu_memory_mb / mb_per_structure / self.mem_estimate_safety_factor
            )
        return self._call(atoms_list, batch_size)


    def _call(
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
            allow_unlabeled=True,
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
            for batch in tqdm(sevenn_data, desc="model prediction"):
                batch = batch.to(self.device)
                output = self.sevenn_model(batch)
                energies.append(output.inferred_total_energy.detach().cpu().numpy())

        return np.concatenate(energies, axis=0)


    def estimate_memory_mb_per_structure(self, atoms: Atoms, n_min: int = 1, n_max: int = 10) -> float:
        assert self.device.type == "cuda", f"Expected cuda device, got {self.device.type}"
        assert n_max > n_min
        n_vals = [n_min, n_max]
        mem_reserved = []
        for n in n_vals:
            self._call([atoms] * n, batch_size = int(n))
            mem_reserved.append(torch.cuda.memory_reserved(self.device))
            torch.cuda.empty_cache()

        slope = (mem_reserved[1] - mem_reserved[0]) / 1024**2 / (n_max - n_min)
        return slope
