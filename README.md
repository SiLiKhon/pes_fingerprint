# PES fingerprint: characterizing ionic mobility in solids

The method is described in [arXiv:2411.06804](https://arxiv.org/abs/2411.06804).

You can run this code on the [Constructor Platform](https://constructor.app/platform/public/project/pes_fingerprint)!

## Installation

### Old M3GNet environment
This is optional, but required to run our example scripts and reproduce our results.

#### Docker environment
Dockerfile:
```Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y python3.10 python3.10-dev python3-pip

ARG USERNAME=container-user
ARG USERID= # your user id goes here (output of `id -u`)
ARG GROUPID= # your group id goes here (output of `id -g`)

RUN groupadd --gid $GROUPID $USERNAME \
    && useradd --uid $USERID --gid $GROUPID -m $USERNAME

USER $USERNAME
RUN python3 -m pip install jupyter m3gnet==0.2.4 tensorflow==2.13 pymatgen==2023.9.25
```

#### Without docker
These are the required packages, but it may be tricky to make the old `tensorflow` see your GPUs:
```bash
pip install m3gnet==0.2.4 tensorflow==2.13 pymatgen==2023.9.25
```

### Main requirements
Then install the main requirements:

```bash
pip install -r requirements.txt
```

## Example structure calculation
Calculate PES descriptors for the `mp-1185319` structure (requires `m3gnet` installed):
```bash
python -m scripts.example_structure_calculation
```

## Running on full Materials Project with minimal selection

### Single GPU or CPU
```bash
python3 -m scripts.example_run_mp \
  --num-jobs 10 \
  --first 0 \
  --last-inclusive 5999 \
  --export-to-file predictions.csv
```

### Multiple GPUs
This would be easy to automate, but so far one needs to manually start jobs on each GPU,
e.g. by running each line in a separate teminal session (example with 2 GPUs):
```bash
CUDA_VISIBLE_DEVICES='0' python3 -m scripts.example_run_mp \
  --num-jobs 10 --first 0 --last-inclusive 2999 --export-to-file predictions-0-2999.csv
CUDA_VISIBLE_DEVICES='1' python3 -m scripts.example_run_mp \
  --num-jobs 10 --first 3000 --last-inclusive 5999 --export-to-file predictions-3000-5999.csv
```
Note that 10 jobs per GPU as above would need ~40GB of GPU memory at peak memory usage,
so please scale that parameter based on the available memory.

## Integrating alternative IAPs

Here's an example SevenNet integration (**only given as example, very inefficient, see note below**):
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


## Citation

A. Maevskiy, A. Carvalho, E. Sataev, V. Turchyna, K. Noori, A. Rodin, A. H. Castro Neto and A. Ustyuzhanin,
Predicting ionic conductivity in solids from the machine-learned potential energy landscape, [arXiv:2411.06804](https://arxiv.org/abs/2411.06804) (2024)
