from helix.core import images
from helix.analysis.sequence.scorer import score_mutations
from helix.core import app, images
import pandas as pd
import os
import tempfile


AAs = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
]


def deep_mutational_scan(sequence, exclude_noop=True):
    for pos, wt in enumerate(sequence):
        for mt in AAs:
            if exclude_noop and wt == mt:
                continue
            yield (pos, wt, mt)


@app.function(image=images.mmseqs)
def get_struc_seq(pdb_string: str,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = False,
                  plddt_threshold: float = 70.) -> dict:
    """

    Args:
        pdb_string: PDB file content as a string
        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
        process_id: Process ID for temporary files. This is used for parallel processing.
        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.
        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    import numpy as np

    # Create a temporary file to save the PDB string
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_pdb_file:
        tmp_pdb_file.write(pdb_string.encode())
        tmp_pdb_path = tmp_pdb_file.name

    tmp_save_path = f"get_struc_seq_{process_id}.tsv"
    cmd = f"foldseek structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {tmp_pdb_path} {tmp_save_path}"
    os.system(cmd)

    seq_dict = {}
    name = os.path.basename(tmp_pdb_path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]

            # Mask low plddt
            if plddt_mask:
                plddts = extract_plddt(tmp_pdb_path)
                assert len(plddts) == len(
                    struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"

                # Mask regions with plddt < threshold
                indices = np.where(plddts < plddt_threshold)[0]
                np_seq = np.array(list(struc_seq))
                np_seq[indices] = "#"
                struc_seq = "".join(np_seq)

            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]

            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower()
                                           for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)

    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    os.remove(tmp_pdb_path)
    return seq_dict


def extract_plddt(pdb_path: str):
    """
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """
    import re
    import numpy as np

    with open(pdb_path, "r") as r:
        plddt_dict = {}
        for line in r:
            line = re.sub(' +', ' ', line).strip()
            splits = line.split(" ")

            if splits[0] == "ATOM":
                # If position < 1000
                if len(splits[4]) == 1:
                    pos = int(splits[5])

                # If position >= 1000, the blank will be removed, e.g. "A 999" -> "A1000"
                # So the length of splits[4] is not 1
                else:
                    pos = int(splits[4][1:])

                plddt = float(splits[-2])

                if pos not in plddt_dict:
                    plddt_dict[pos] = [plddt]
                else:
                    plddt_dict[pos].append(plddt)

    plddts = np.array([np.mean(v) for v in plddt_dict.values()])
    return plddts


@app.function(image=images.base, timeout=4000)
def dms(sequence, metrics=["wildtype_marginal", "masked_marginal"], model_names=["facebook/esm1b_t33_650M_UR50S",  "facebook/esm2_t33_650M_UR50D", "facebook/esm2_t36_3B_UR50D"], pdb_id=None, chain_id=None):
    from helix.analysis.structure.utils import fetch_pdb_structure, pdb_to_string
    offset_idx = 1
    mutation_col = "mutant"
    data = [
        f'{wt}{pos + offset_idx}{mt}'
        for pos, wt, mt in deep_mutational_scan(sequence)
    ]
    df = pd.DataFrame(data, columns=[mutation_col])
    [sequence[:pos] + mt + sequence[pos+1:]
     for pos, wt, mt in deep_mutational_scan(sequence)]
    sa_sequence = None
    if pdb_id:
        structure = fetch_pdb_structure(pdb_id)
        pdb_string = pdb_to_string(structure)
        seq_dict = get_struc_seq.remote(pdb_string, chains=chain_id)
        sa_sequence = seq_dict[chain_id][2]
    for metric in metrics:
        results = score_mutations.starmap(
            [(model_name, sa_sequence if "saprot" in model_name.lower() else sequence, df[mutation_col], metric)
             for model_name in model_names]
        )
        for model_name, result in zip(model_names, results):
            df[model_name + "_" + metric] = result

    return df
