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
