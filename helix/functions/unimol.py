from typing import List, Dict, Any
import numpy as np
from helix.core import app, images, volumes
from modal import gpu, batched

MODEL_DIR = "/mnt/models/unimol"


@app.function(
    image=images.base.pip_install("unimol_tools").env(
        {"UNIMOL_WEIGHT_DIR": MODEL_DIR}),
    gpu=gpu.A100(),
    volumes={MODEL_DIR: volumes.model_weights},
    timeout=3600
)
@batched(max_batch_size=30, wait_ms=1000)
def compute_unimol_representations(smiles_list: List[str]) -> List[Dict[str, Any]]:
    from unimol_tools import UniMolRepr

    unimol_model = UniMolRepr(data_type='molecule', remove_hs=False)
    unimol_repr = unimol_model.get_repr(smiles_list, return_atomic_reprs=True)

    results = []
    for cls_repr, atomic_repr in zip(unimol_repr['cls_repr'], unimol_repr['atomic_reprs']):
        cls_repr_np = np.array(cls_repr)
        atomic_repr_np = np.array(atomic_repr)
        results.append({
            "cls_repr": cls_repr_np,
            "atomic_reprs": atomic_repr_np,
            "cls_repr_shape": cls_repr_np.shape,
            "atomic_reprs_shape": atomic_repr_np.shape
        })

    return results


@app.local_entrypoint()
def test():
    # Local entrypoint for testing and development
    test_smiles = ["c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]"]
    representations = list(compute_unimol_representations.map(test_smiles))

    print("CLS token representation shape:",
          representations[0]["cls_repr_shape"])
    print("Atomic level representation shape:",
          representations[0]["atomic_reprs_shape"])

# Example usage:
# If running this script directly, use: `modal run unimol.py`
# This will execute the main function as the entry point
