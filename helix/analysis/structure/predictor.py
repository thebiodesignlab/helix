from modal import method, gpu
from helix.core import app, volumes
import helix.core.images as images

MODEL_DIR = "/mnt/models"


@app.cls(gpu=gpu.H100(), timeout=6000,  image=images.base, volumes={MODEL_DIR: volumes.model_weights})
class ESMFold():
    def __init__(self, device: str = "cuda"):
        from transformers import AutoTokenizer, EsmForProteinFolding
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", cache_dir=MODEL_DIR)
        self.device = device
        if device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        # TODO: Make chunk size configurable?
        self.model.trunk.set_chunk_size(64)

    @method()
    def batch_predict(self, sequences: list[str]) -> list[str]:
        import torch
        tokenized = self.tokenizer(
            sequences, return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized = tokenized.to(self.device)
        with torch.inference_mode():
            outputs = self.model(tokenized)
        pdb_structures = self.convert_outputs_to_pdb(outputs)
        return pdb_structures

    @staticmethod
    def convert_outputs_to_pdb(outputs):
        from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
        final_atom_positions = atom14_to_atom37(
            outputs["positions"][-1], outputs)
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
            )
            pdbs.append(to_pdb(pred))
        return pdbs


@app.function(image=images.base, timeout=10000)
def predict_structures(sequences, batch_size: int = 1):
    from helix.utils import create_batches
    from tqdm import tqdm  # Import tqdm for progress tracking

    print(f"Predicting structures for {len(sequences)} sequences")
    model = ESMFold()
    batched_sequences = create_batches(sequences, batch_size)

    results = []
    total_sequences = len(sequences)
    processed_sequences = 0

    with tqdm(total=total_sequences, desc="Processing sequences") as pbar:
        for batched_results in model.batch_predict.map(batched_sequences, return_exceptions=True):
            for pdb_string in batched_results:
                if isinstance(pdb_string, Exception):
                    print(f"Error: {pdb_string}")
                else:
                    results.append(pdb_string)
                processed_sequences += 1
                # Update progress bar for each sequence processed
                pbar.update(1)
    return results
