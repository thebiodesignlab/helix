from io import StringIO
from modal import Image, method
from .main import CACHE_DIR, volume, stub
import modal
from Bio.PDB.Structure import Structure
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB.PDBIO import PDBIO
import os
import transformers


def download_models():
    from transformers import EsmModel, EsmForProteinFolding, AutoTokenizer

    EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1")
    AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1")

    EsmModel.from_pretrained(
        "facebook/esm2_t36_3B_UR50D")
    AutoTokenizer.from_pretrained(
        "facebook/esm2_t36_3B_UR50D")


image = Image.debian_slim().apt_install("git").pip_install(
    "torch",
    "biopython",
    "matplotlib",
    "transformers",
    "pandas"
).run_function(download_models)


@stub.cls(gpu='A10G', timeout=2000, network_file_systems={CACHE_DIR: volume}, image=image, allow_cross_region_volumes=True, concurrency_limit=9)
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
    def infer(self, sequences, output_hidden_states: bool = False, output_attentions: bool = False, return_logits: bool = False) -> transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions:
        import torch
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available")
        print(f"Running inference on {sequences} sequences")
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


@stub.cls(gpu='A10G', timeout=2000, network_file_systems={CACHE_DIR: volume}, image=image, allow_cross_region_volumes=True, concurrency_limit=9)
class EsmForMaskedLM():
    def __init__(self, device: str = "cuda", model_name: str = "facebook/esm2_t36_3B_UR50D"):
        import transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name)
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name)
        self.device = device
        if device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()

    @method()
    def infer(self, sequence: str) -> transformers.modeling_outputs.MaskedLMOutput:
        import torch
        tokenized = self.tokenizer.encode(sequence, return_tensors='pt')
        tokenized = tokenized.to(self.device)
        with torch.inference_mode():
            outputs = self.model(tokenized, return_dict=True)
        return outputs

    @method()
    def score(self, sequence: str, batch_size: int = 32) -> float:
        # Reference: Masked Language Model Scoring
        # https://arxiv.org/abs/1910.14659
        import torch
        import numpy as np
        tokenized = self.tokenizer.encode(sequence, return_tensors='pt')
        repeat_input = tokenized.repeat(tokenized.size(-1)-2, 1)

        # mask one by one except [CLS] and [SEP]
        mask = torch.ones(tokenized.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(
            mask == 1, self.tokenizer.mask_token_id)

        labels = repeat_input.masked_fill(
            masked_input != self.tokenizer.mask_token_id, -100)

        # Initialize loss accumulator
        total_loss = 0.0

        # Process in batches
        for i in range(0, masked_input.size(0), batch_size):
            batch_masked_input = masked_input[i:i+batch_size].to(self.device)
            batch_labels = labels[i:i+batch_size].to(self.device)

            with torch.inference_mode():
                outputs = self.model(batch_masked_input, labels=batch_labels)
                loss = outputs.loss
                total_loss += loss.item() * batch_masked_input.size(0)

        # Calculate average loss
        avg_loss = total_loss / masked_input.size(0)

        return np.exp(avg_loss)

    @method()
    def entropies(self, sequence: str, batch_size: int = 32) -> list[float]:
        """
        Calculate the entropy for each residue position in a protein sequence by masking each residue and calculating the entropy of the masked position.
        """
        import torch
        tokenized = self.tokenizer.encode(sequence, return_tensors='pt')
        repeat_input = tokenized.repeat(tokenized.size(-1)-2, 1)

        # mask one by one except [CLS] and [SEP]
        mask = torch.ones(tokenized.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(
            mask == 1, self.tokenizer.mask_token_id)

        labels = repeat_input.masked_fill(
            masked_input != self.tokenizer.mask_token_id, -100)

        # Initialize entropy list with zeros for each position in the sequence
        # Subtract 2 for [CLS] and [SEP] tokens
        entropies = [0.0] * len(sequence)
        distributions = [torch.zeros(
            self.model.config.vocab_size)] * len(sequence)
        # Process in batches
        for i in range(0, masked_input.size(0), batch_size):
            batch_masked_input = masked_input[i:i+batch_size].to(self.device)
            batch_labels = labels[i:i+batch_size].to(self.device)

            with torch.inference_mode():
                outputs = self.model(batch_masked_input, labels=batch_labels)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                log_probabilities = torch.log(probabilities)
                batch_entropy = - \
                    torch.sum(probabilities * log_probabilities, dim=-1)
                # Round to 4 decimal places
                batch_entropy = torch.round(batch_entropy, decimals=4)
                probabilities = torch.round(probabilities, decimals=4)
                # check that probabilities sum to 1 roughly

                # Update the corresponding positions in the entropies list
                for j, entropy_value in enumerate(batch_entropy.cpu().tolist()):
                    position = i+j
                    entropies[position] = entropy_value[position+1]
                    prob_distribution = probabilities[j,
                                                      position + 1].cpu().numpy()
                    distributions[position] = {
                        self.tokenizer.decode([token_id]): prob
                        for token_id, prob in enumerate(prob_distribution)
                        if self.tokenizer.decode([token_id]).isalpha() and len(self.tokenizer.decode([token_id])) == 1
                    }

        return entropies, distributions

    @modal.exit()
    def empty_cache(self):
        import torch
        torch.cuda.empty_cache()


@stub.cls(gpu="A10G", timeout=6000, network_file_systems={CACHE_DIR: volume}, image=image)
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
    def infer(self, sequences: list[SeqRecord]) -> Structure:
        import torch
        from Bio.PDB import PDBParser
        parser = PDBParser()
        structures = []
        tokenized = self.tokenizer(
            [str(seq_record.seq) for seq_record in sequences], return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized = tokenized.to(self.device)
        with torch.inference_mode():
            outputs = self.model(tokenized)
        pdb_structures = self.convert_outputs_to_pdb(outputs)
        # Convert pdb strings to biopython structures
        for seq_record, pdb in zip(sequences, pdb_structures):
            structure = parser.get_structure(seq_record.id, StringIO(pdb))
            structures.append(structure)
        return structures

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


PROTEIN_STRUCTURE_MODELS = {
    "esmfold": ESMFold
}


@stub.function(network_file_systems={CACHE_DIR: volume}, image=image, timeout=10000)
def predict_structures(sequences, model_name: str = "esmfold", batch_size: int = 1):
    from helix.utils import create_batches
    if model_name not in PROTEIN_STRUCTURE_MODELS:
        raise ValueError(
            f"Model {model_name} is not supported. Supported models are: {list(PROTEIN_STRUCTURE_MODELS.keys())}")
    print(f"Using model {model_name}")
    print(f"Predicting structures for {len(sequences)} sequences")
    model = PROTEIN_STRUCTURE_MODELS[model_name]()
    batched_sequences = create_batches(sequences, batch_size)

    result = []
    for batched_results in model.infer.map(batched_sequences, return_exceptions=True):
        for struct in batched_results:
            if isinstance(struct, Exception):
                print(f"Error: {struct}")
            else:
                print(f"Successfully predicted structure for {struct.id}")
                result.append(struct)
    return result


@stub.local_entrypoint()
def predict_structures_from_fasta(fasta_file: str, output_dir: str, batch_size: int = 1):
    """
    Predicts protein structures from a given FASTA file and saves them as PDB files in the specified output directory.

    Args:
        fasta_file (str): Path to the FASTA file containing protein sequences.
        output_dir (str): Directory where the PDB files will be saved.
        batch_size (int): Number of sequences to process in a batch. Default is 1 due to high memory usage of the model.
                          For shorter sequences, the batch size can be increased.
    """
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    result = predict_structures.remote(sequences, batch_size=batch_size)
    os.makedirs(output_dir, exist_ok=True)
    for struct in result:
        io = PDBIO()
        io.set_structure(struct)
        io.save(f"{output_dir}/{struct.id}.pdb")


@stub.local_entrypoint()
def predict_structures_from_csv(csv_file: str, id_column: str, sequence_column: str, output_dir: str, batch_size: int = 1):
    """
    Predicts protein structures from a given CSV file and saves them as PDB files in the specified output directory.

    Args:
        csv_file (str): Path to the CSV file containing protein sequences with their IDs.
        id_column (str): The name of the column in the CSV file that contains the sequence IDs.
        sequence_column (str): The name of the column in the CSV file that contains the protein sequences.
        output_dir (str): Directory where the PDB files will be saved.
        batch_size (int): Number of sequences to process in a batch. Default is 1 due to high memory usage of the model.
                            For shorter sequences, the batch size can be increased.
    """
    import pandas as pd
    df = pd.read_csv(csv_file)
    if id_column not in df.columns or sequence_column not in df.columns:
        raise ValueError(
            f"CSV file must contain '{id_column}' and '{sequence_column}' columns.")

    sequences = []
    for _, row in df.iterrows():
        sequences.append(
            SeqRecord(Seq(row[sequence_column]), id=str(row[id_column])))

    result = predict_structures.remote(
        sequences, batch_size=batch_size)
    os.makedirs(output_dir, exist_ok=True)
    for struct in result:
        io = PDBIO()
        io.set_structure(struct)
        io.save(f"{output_dir}/{struct.id}.pdb")


@stub.local_entrypoint()
def calculate_entropy(sequence: str, model_name: str = "facebook/esm2_t36_3B_UR50D") -> list[float]:
    """
    Calculate the entropy for each residue position in a protein sequence.

    Args:
        sequence (str): A single protein sequence.
        model_name (str): Name of the model to use for entropy calculation. Default is 'facebook/esm2_t36_3B_UR50D'.

    Returns:
        list[float]: A list of entropy values for each residue position in the sequence.
    """
    model = EsmModel(model_name=model_name)
    entropy = model.get_entropy.remote(sequence)
    return entropy.tolist()
