from typing import List, Dict, Any
import numpy as np
from helix.core import app, images, volumes
from modal import gpu

MODEL_DIR = "/usr/local/lib/python3.10/site-packages/unimol_tools/weights/"


@app.function(
    image=images.base.pip_install("unimol_tools"),
    gpu=gpu.A100(),
    volumes={MODEL_DIR: volumes.model_weights},
    timeout=3600
)
def compute_unimol_representations(smiles_list: List[str]) -> Dict[str, Any]:
    from unimol_tools import UniMolRepr

    unimol_model = UniMolRepr(data_type='molecule', remove_hs=False)
    unimol_repr = unimol_model.get_repr(smiles_list, return_atomic_reprs=True)

    cls_repr = np.array(unimol_repr['cls_repr'])
    atomic_reprs = np.array(unimol_repr['atomic_reprs'])

    return {
        "cls_repr": cls_repr,
        "atomic_reprs": atomic_reprs,
        "cls_repr_shape": cls_repr.shape,
        "atomic_reprs_shape": atomic_reprs.shape
    }


@app.local_entrypoint()
def test_unimol():
    # Local entrypoint for testing and development
    test_smiles = ["c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]"]
    representations = compute_unimol_representations.remote(test_smiles)
    print("CLS token representation shape:", representations["cls_repr_shape"])
    print("Atomic level representation shape:",
          representations["atomic_reprs_shape"])

# Example usage:
# If running this script directly, use: `modal run unimol.py`
# This will execute the main function as the entry point
