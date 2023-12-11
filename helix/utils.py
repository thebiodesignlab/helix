from Bio.PDB import PDBParser, Structure
from io import StringIO
import requests
from collections import defaultdict


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
