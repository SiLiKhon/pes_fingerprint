import torch
import numpy as np
from ase import Atoms
import dgl
import matgl
from matgl.ext.ase import Atoms2Graph


class M3GNetBatchPES:
    def __init__(
        self,
        model_name: str = "M3GNet-MP-2021.2.8-PES",
        device: torch.device | None = None,
    ):
        potential = matgl.load_model(model_name)
        if device is not None:
            potential = potential.to(device)
        potential.calc_forces = False
        potential.calc_hessian = False
        potential.calc_magmom = False
        potential.calc_repuls = False
        potential.calc_stresses = False
        self.potential = potential
        self.converter = Atoms2Graph(
            potential.model.element_types, potential.model.cutoff
        )
        self.device = device

    def run_on_batch(self, batch_atoms: list[Atoms]) -> np.ndarray:
        batch_graph, batch_lat, batch_state_attrs = zip(
            *list(
                map(self.converter.get_graph, batch_atoms)
            )
        )
        batch_graph = dgl.batch(batch_graph)
        batch_lat = torch.cat(batch_lat, dim=0)
        if self.device is not None:
            batch_graph = batch_graph.to(self.device)
            batch_lat = batch_lat.to(self.device)

        with torch.no_grad():
            energies, _, _, _ = self.potential(
                g=batch_graph, lat=batch_lat, state_attr=batch_state_attrs
            )
        energies = energies.cpu().numpy()
        if len(batch_atoms) == 1:
            energies = energies.reshape(1)
        return energies

    def __call__(
        self,
        atoms_set: list[Atoms],
        batch_size: int,
    ) -> np.ndarray:
        return np.concatenate(
            [
                self.run_on_batch(atoms_set[i: i + batch_size])
                for i in range(0, len(atoms_set), batch_size)
            ]
        )
