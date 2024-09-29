# PES fingerprint: characterizing ionic mobility in solids

## Installation

Optionally, though recommended for the default example, install the old `m3gnet`
implementation:

```bash
pip install m3gnet==0.2.4 tensorflow==2.13
```

Then install the main requirements:

```bash
pip install -r requirements.txt
```


Calculate PES descriptors for example `mp-1185319` structure (requires `m3gnet` installed):
```bash
python -m scripts.example_structure_calculation
```

## Integrating alternative IAPs

Here's an example SevenNet integration:
```bash
pip install sevenn==0.9.3
```

Python script:
```python
from tqdm.auto import tqdm
from pes_fingerprint.pipelines.calculators import factory
from pes_fingerprint.pipelines import process_structure
from sevenn.sevennet_calculator import SevenNetCalculator
from scripts.example_structure_calculation import build_mp_1185319
import torch
torch.set_num_threads(1)

@factory("custom_iap_model")
def create_SevenNet_calculator(model_name, device):
    sevennet_calculator = SevenNetCalculator(model_name, device=device)
    # We need to return a function mapping a list of structures
    # to the list of their energies. WARNING: updates to the body
    # of this function will not trigger re-calculation of cached PES results;
    # you need to either manually delete the calculated cache or change some
    # parameter (e.g., register changed factory under a new key)
    def calculator_func(structures):
        energies = [float("nan")] * len(structures)
        for i, atoms in enumerate(tqdm(structures)):
            atoms.calc = sevennet_calculator
            energies[i] = atoms.get_potential_energy()
        return energies
    return calculator_func

# Calculating PES descriptors:
atoms = build_mp_1185319()
predictions = process_structure(
    atoms,
    mpe_params={
        "calculator_params": {
            "key": "custom_iap_model",  # pass the key defined in the `factory` decorator
            "model_name": "7net-0",     # + other parameters to `create_SevenNet_calculator`
        },
        "calculator_params_ignored_in_cache": {
            "device": "cpu",  # this will be passed too, but won't re-trigger caching if changed
        },
    },
)

print(f"Predictions for {str(atoms.symbols)} (mp-1185319):")
for k in ["mpe", "fv_0p5_connected_union", "fv_0p5_disconnected_union", "Xi"]:
    print(f"    {k:25} : {predictions[k]:.3}")
```

Note: the `calculator_func` from the above snippet is extremely inefficient and is only given as an example. The recommended way is to implement batching, similar to how it is done [in the original SevenNet code](https://github.com/MDIL-SNU/SevenNet/blob/v0.9.3/sevenn/scripts/inference.py#L178-L239).
