from typing import List, Dict, Any, Optional
import os

from tqdm.auto import tqdm
from mp_api.client import MPRester

from pes_fingerprint.pipelines.cache_utils import setup_cache

_memory = setup_cache()


# Taken from: https://github.com/knoori/se_energy_landscape/blob/abfee58e3f7b9ad28c063404507c317e2671aa0d/mp_extract.py
@_memory.cache
def query_mp(
    band_gap_cutoff: float = 0.5,
    energy_above_hull_cutoff: float = 0.05,
    target_element: str = "Li",
    extra_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    MP_API_KEY = os.environ["MP_API_KEY"]

    print(
        f"Querying Materials Project for {target_element}-containing materials "
        f"with a band gap greater than {band_gap_cutoff} eV and an energy above "
        f"hull less than {energy_above_hull_cutoff} eV / atom..."
    )

    with MPRester(api_key=MP_API_KEY) as mpr:
        bg_cutoff = (band_gap_cutoff, 100)
        eah_cutoff = (0.0, energy_above_hull_cutoff)
        keywords = [
            "material_id", "formula_pretty", "nelements", "elements",
            "nsites", "structure", "energy_above_hull", "volume", "band_gap",
            "database_IDs",
        ]
        if extra_fields is not None:
            keywords += extra_fields
        docs = mpr.materials.summary.search(
            elements=[target_element],
            band_gap=bg_cutoff,
            energy_above_hull=eah_cutoff,
            fields=keywords,
        )

    # Return the query results
    return sorted(
        [d.dict() for d in tqdm(docs, desc="Converting docs to dicts")],
        key=lambda x: int(x["material_id"][3:])
    )
