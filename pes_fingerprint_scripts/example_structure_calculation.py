from ase import Atoms
from pes_fingerprint.pipelines import process_structure

def build_mp_1185319() -> Atoms:
    return Atoms(
        symbols=['Li', 'Li', 'Cl', 'Cl'],
        positions=[
            [0.        , 0.        , 2.39040061],
            [1.96243315, 1.13301147, 5.56898111],
            [0.        , 0.        , 6.35070212],
            [1.96243315, 1.13301147, 3.17211527]
        ],
        cell=[
            [ 1.962441, -3.399048,  0.      ],
            [ 1.962441,  3.399048,  0.      ],
            [ 0.      ,  0.      ,  6.357161],
        ],
        pbc=True,
    )

if __name__ == "__main__":
    atoms = build_mp_1185319()
    predictions = process_structure(atoms)
    print(f"Predictions for {str(atoms.symbols)} (mp-1185319):")
    for k in ["mpe", "fv_0p5_connected_union", "fv_0p5_disconnected_union", "Xi"]:
        print(f"    {k:25} : {predictions[k]:.3}")
