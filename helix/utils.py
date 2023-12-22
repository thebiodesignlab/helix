from Bio.PDB import PDBParser, Structure
from io import StringIO
import requests
from collections import defaultdict, OrderedDict


def create_batches(sequences, batch_size: int = 32):
    # Group sequences by length
    length_to_sequences = defaultdict(list)
    for sequence in sequences:
        length_to_sequences[len(sequence)].append(sequence)

    # Create batches
    batches = []
    for sequences in length_to_sequences.values():
        for i in range(0, len(sequences), batch_size):
            batches.append(sequences[i:i + batch_size])

    return batches


def fetch_pdb_structure(pdb_id: str) -> Structure:
    """
    Fetch a PDB structure by its ID and return as a Biopython Structure object.

    Parameters:
    - pdb_id (str): The PDB ID to fetch.

    Returns:
    - Bio.PDB.Structure.Structure: The fetched structure as a Biopython object.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(
            f"Failed to download PDB {pdb_id}. HTTP Status Code: {response.status_code}")

    pdb_content = response.text
    pdb_io = StringIO(pdb_content)

    parser = PDBParser(QUIET=True)  # QUIET=True suppresses warnings
    structure = parser.get_structure(pdb_id, pdb_io)

    return structure


def count_mutations(sequence, variants):
    all_mutations = []
    for variant in variants:
        # Positions are 1-indexed
        mutations = [f"{wt}{pos+1}{mut}" for pos,
                     (wt, mut) in enumerate(zip(sequence, variant)) if wt != mut]
        all_mutations.extend(mutations)

    mutation_counts = {}
    for mutation in all_mutations:
        mutation_counts[mutation] = mutation_counts.get(mutation, 0) + 1

    # Sort by count in descending order and return an OrderedDict
    sorted_mutation_counts = OrderedDict(
        sorted(mutation_counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_mutation_counts


def dataframe_to_fasta(df, id_col, seq_col, metadata_cols=None):
    """
    Writes a Pandas DataFrame to a FASTA format file with optional metadata.

    Parameters:
    - df: Pandas DataFrame containing the sequence data.
    - id_col: Name of the column in df that contains the sequence identifiers.
    - seq_col: Name of the column in df that contains the sequences.
    - metadata_cols: List of column names to include as metadata. If None, all columns except seq_col are included.
    """
    if metadata_cols is None:
        metadata_cols = [
            col for col in df.columns if col not in [id_col, seq_col]]

    fasta_str = ""
    for index, row in df.iterrows():
        metadata = [f"{col}={row[col]}" for col in metadata_cols]
        metadata_str = ' '.join(metadata)
        fasta_str += f">{row[id_col]} {metadata_str}\n"
        fasta_str += f"{row[seq_col]}\n"
    return fasta_str

# Example usage:
# Assuming `df` is your DataFrame, 'id' is the column with identifiers,
# 'sequence' is the column with sequence data, and 'metadata_cols' is a list of columns to include as metadata.
# dataframe_to_fasta(df, 'id', 'sequence', 'output.fasta', metadata_cols=['gene', 'organism'])
