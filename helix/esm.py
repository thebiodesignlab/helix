from io import StringIO
from modal import Image, method, Mount
from .main import CACHE_DIR, volume, stub
from Bio.SeqRecord import SeqRecord
from Bio.PDB.Structure import Structure
import transformers


def download_models():
    from transformers import EsmModel, EsmForProteinFolding
    EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1")
    EsmModel.from_pretrained(
        "facebook/esm2_t36_3B_UR50D")
    # tokenizers
    transformers.AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1")
    transformers.AutoTokenizer.from_pretrained(
        "facebook/esm2_t36_3B_UR50D")


dockerhub_image = Image.from_registry(
    "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel"
).apt_install("git"
              ).pip_install("fair-esm[esmfold]",
                            "dllogger @ git+https://github.com/NVIDIA/dllogger.git",
                            "openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307"
                            ).pip_install("gradio",
                                          "biopython",
                                          "pandas",
                                          "transformers",
                                          "scikit-learn",
                                          "matplotlib",
                                          "seaborn",
                                          ).run_function(download_models, mounts=[Mount.from_local_python_packages("helix")])


@stub.cls(gpu='A10G', timeout=2000, network_file_systems={CACHE_DIR: volume}, image=dockerhub_image, allow_cross_region_volumes=True)
class EsmModel():
    def __init__(self, device: str = "cuda", model_name: str = "facebook/esm2_t36_3B_UR50D"):
        import transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name)
        self.model = transformers.AutoModel.from_pretrained(
            model_name)
        self.device = device
        if device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()

    @method()
    def infer(self, sequences, output_hidden_states: bool = False, output_attentions: bool = False) -> transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions:
        import torch
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available")
        print(f"Running inference on {sequences} sequences")
        sequences = [str(sequence.seq) for sequence in sequences]
        tokenized = self.tokenizer(
            sequences, return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized = tokenized.to(self.device)
        with torch.inference_mode():
            outputs = self.model(tokenized, output_hidden_states=output_hidden_states,
                                 output_attentions=output_attentions)
        return outputs

    def __exit__(self, exc_type, exc_value, traceback):
        import torch
        torch.cuda.empty_cache()


@stub.cls(gpu='A10G', timeout=2000, network_file_systems={CACHE_DIR: volume}, image=dockerhub_image)
class ESMFold():
    def __init__(self, device: str = "cuda"):
        from transformers import AutoTokenizer, EsmForProteinFolding
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",)  # low_cpu_mem_usage=True
        self.device = device
        if device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        # TODO: Make chunk size configurable?
        self.model.trunk.set_chunk_size(64)

    @method()
    def infer(self, sequence: SeqRecord) -> Structure:
        import torch
        tokenized = self.tokenizer(
            [str(sequence.seq)], return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized = tokenized.to(self.device)
        with torch.inference_mode():
            outputs = self.model(tokenized)
        pdb_structures = self.convert_outputs_to_pdb(outputs)
        # Convert pdb strings to biopython structures
        from Bio.PDB import PDBParser
        parser = PDBParser()
        structures = [parser.get_structure(
            sequence.id, StringIO(pdb)) for pdb in pdb_structures]
        return structures[0]

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
