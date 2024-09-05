

from helix.core import app
from modal import Image
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
image = Image.debian_slim().pip_install(
    "dnachisel", "biopython", "biotite", "pandas", "primers", "openpyxl")


@app.function(image=image)
def codon_optimize(sequence: SeqRecord, organism="e_coli", avoid_patterns=[], gc_min=0, gc_max=1, gc_window=50):
    """
    Optimize a DNA sequence for expression in a given organism.
    Parameters
    ----------
    sequence : str
        The DNA sequence to be optimized.
    organism : str, optional
        The organism in which the DNA sequence is to be expressed. Default is "e_coli". Either a TaxID or the name of the species (e.g. "e_coli", "s_cerevisiae", "h_sapiens", "c_elegans", "b_subtilis", "d_melanogaster").
    avoid_patterns : list, optional
        A list of sequence patterns to avoid in the optimized sequence. Default is []. E.g. ["BsaI_site", "BsmBI_site"].

    """
    from dnachisel import AvoidPattern, DnaOptimizationProblem, CodonOptimize, EnforceTranslation, reverse_translate
    sequence.seq = Seq(reverse_translate(sequence.seq))
    constraints = [
        # EnforceGCContent(mini=gc_min, maxi=gc_max, window=gc_window),
        EnforceTranslation(),
    ]
    for pattern in avoid_patterns:
        if pattern:
            constraints.append(AvoidPattern(pattern))

    problem = DnaOptimizationProblem(sequence=sequence,
                                     constraints=constraints,
                                     objectives=[CodonOptimize(species=organism)])
    problem.initialize()

    print("Resolving constraints...")
    problem.resolve_constraints()
    print("Optimizing ...")
    problem.optimize()

    print(problem.constraints_text_summary())
    print(problem.objectives_text_summary())

    return problem.record


CODON_LENGTH = 3


def parse_mutation(mutation_str):
    """
    Parse a mutation string like "S2M" into its components: original amino acid,
    position, and new amino acid.

    Parameters
    ----------
    mutation_str : str
        The mutation string to parse.

    Returns
    -------
    tuple
        A tuple containing the original amino acid, the position, and the new amino acid.
    """
    if len(mutation_str) < 3:
        raise ValueError("Mutation string is too short to be valid.")

    original_aa = mutation_str[0]
    new_aa = mutation_str[-1]

    # Extract the position, which should be the substring between the two amino acids
    position_str = mutation_str[1:-1]

    # Check if the position is a valid integer
    if not position_str.isdigit():
        raise ValueError("Position in mutation string is not a valid integer.")

    position = int(position_str)

    return original_aa, position, new_aa


def mutate_sequence(plasmid_sequence, gene_start, mutation):
    from biotite.sequence import CodonTable, NucleotideSequence
    """Apply a mutation to the plasmid sequence at the specified gene start."""
    table = CodonTable.default_table()
    mut_from, mutation_position, mut_to = parse_mutation(mutation)
    # Check if the amino acid at the mutation position matches mut_from
    codon_start = gene_start + (mutation_position - 1) * CODON_LENGTH
    codon = NucleotideSequence(
        plasmid_sequence[codon_start:codon_start + CODON_LENGTH])
    amino_acid = codon.translate(complete=True)[0]
    if amino_acid != mut_from:
        raise ValueError(
            f"The amino acid at position {mutation_position} is {amino_acid}, not {mut_from}.")
    # Assuming mutation format is 'A123T'
    mutation_position = int(mutation[1:-1])
    mutated_codon = str(table[mut_to][0])
    position_in_plasmid = gene_start + (mutation_position - 1) * CODON_LENGTH
    return (plasmid_sequence[:position_in_plasmid] + mutated_codon +
            plasmid_sequence[position_in_plasmid + CODON_LENGTH:])


def circularize_sequence(sequence, cut_position):
    """Circularize the sequence by making the cut_position the new start."""
    return sequence[cut_position:] + sequence[:cut_position]


def create_primer_data(mutated_sequence, gene_start, mutation, optimal_len, penalty_len):
    """Create primers for the mutated sequence."""
    from primers import create
    _, mutation_position, _ = parse_mutation(mutation)
    circular_plasmid_sequence = circularize_sequence(
        mutated_sequence, gene_start + (mutation_position - 1) * CODON_LENGTH
    )
    fwd, rev = create(circular_plasmid_sequence,
                      optimal_len=optimal_len, penalty_len=penalty_len)
    return {
        'mutation': mutation,
        'fwd': fwd.seq,
        'rev': rev.seq,
        'fwd_tm': fwd.tm,
        'rev_tm': rev.tm,
        'fwd_gc': fwd.gc,
        'rev_gc': rev.gc,
        'fwd_length': len(fwd.seq),
        'rev_length': len(rev.seq)
    }


@app.function(image=image)
def create_kld_primers(plasmid_sequence, gene_start, mutations, optimal_len=24, penalty_len=5):
    """
    Create primers for a list of point mutations in a plasmid sequence to be used in KLD Site-Directed Mutagenesis.

    Parameters
    ----------
    plasmid_sequence : str
        The plasmid sequence to create primers for.
    gene_start : int
        The start position of the gene in the plasmid sequence starting from 1. Corresponds to position in Benchling.
    mutations : list
        A list of mutations in the form "A123T" where A is the wildtype amino acid,
        123 is the position, and T is the mutant amino acid.
    optimal_len : int, optional
        The optimal length of the primers to design. Default is 24.
    penalty_len : int, optional
        The penalty length for the primers. Default is 1.

    Returns
    -------
    DataFrame
        A DataFrame containing primer data for each mutation.
    """
    import pandas as pd
    primer_data = []
    gene_start = gene_start - 1  # Convert to 0-based index
    for mutation in mutations:
        try:
            mutated_sequence = mutate_sequence(
                plasmid_sequence.upper(), gene_start, mutation)
            int(mutation[1:-1])
            primer_info = create_primer_data(
                mutated_sequence, gene_start, mutation, optimal_len, penalty_len)
            primer_data.append(primer_info)
        except Exception as e:
            # Log the error or handle it as appropriate
            print(f"Error processing mutation {mutation}: {e}")
            continue

    return pd.DataFrame(primer_data)


@app.local_entrypoint()
def codon_optimize_from_fasta(fasta_file: str, output_path, organism: str = "e_coli", avoid_patterns: str = "", gc_min=0, gc_max=1, gc_window=50):
    """
    Optimize a DNA sequence for expression in a given organism.
    Parameters
    ----------
    fasts_file : str
        The path to a FASTA file containing the DNA sequences to be optimized.
    """
    records = []
    for optimized_sequence in codon_optimize.starmap(((record, organism, avoid_patterns.split(","), gc_min, gc_max, gc_window) for record in SeqIO.parse(fasta_file, "fasta")), return_exceptions=True):
        if isinstance(optimized_sequence, Exception):
            print(f"Error optimizing sequence: {optimized_sequence}")
        else:
            records.append(optimized_sequence)

    SeqIO.write(records, output_path, "fasta")


@app.local_entrypoint()
def create_kld_primers_to_csv(plasmid_sequence: str, gene_start: int, mutations: str, output_path: str, plate_output_path: str, start_well: str = 'A1'):
    """
    Create primers for a list of point mutations in a plasmid sequence 
    Parameters
    ----------
    plasmid_sequence : str
        The plasmid sequence to create primers for.
    gene_start : int
        The start position of the gene in the plasmid sequence starting from 0.
    mutations : list
        A list of mutations in the form "A123T" where A is the wildtype amino acid, 123 is the position, and T is the mutant amino acid.
    output_path : str
        The path to save the primers to.
    plate_output_path : str
        The path to save the plate layout to.
    start_well : str, optional
        The starting well position in the plate (e.g., 'A1'). Default is 'A1'.
    """

    # Parse the mutations string into a list of mutations
    # If there's only one mutation, it won't have a comma, so we split by comma only if it's present
    mutations = mutations.split(",") if "," in mutations else [mutations]
    df = create_kld_primers.remote(
        plasmid_sequence, gene_start, mutations)
    df.to_csv(output_path)
    if plate_output_path:
        create_primer_well_df(df, start_well=start_well).to_excel(
            plate_output_path, index=False)


def create_primer_well_df(primer_df, start_well: str = 'A1'):
    """
    Create a df mapping forward and reverse primers to the same well positions in a 96-well plate.
    Parameters
    ----------
    primer_df : pd.DataFrame
        The DataFrame containing primer information.
    start_well : str
        The well position to start assigning primers from.
    """
    import pandas as pd
    # Define well positions
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = list(range(1, 13))
    all_well_positions = [f"{row}{col}" for col in cols for row in rows]

    # Find the index of the starting well
    try:
        start_index = all_well_positions.index(start_well)
    except ValueError:
        raise ValueError(
            f"Invalid start well: {start_well}. Must be in the format 'A1', 'B2', etc.")

    # Slice the well positions to start from the specified well
    well_positions = all_well_positions[start_index:]

    # Iterate over the primer DataFrame and assign well positions
    well_data = []
    for idx, primer in enumerate(primer_df.itertuples(index=False)):
        # Check if we have enough wells left
        if idx >= len(well_positions):
            raise ValueError(
                "Not enough wells available to place all primers.")
        # Same well for forward and reverse
        well_position = well_positions[idx]
        # Add forward primer to the well
        well_data.append({
            'Well Position': well_position,
            'Sequence Name': f"{primer.mutation}_fwd",
            'Sequence': primer.fwd
        })
        # Add reverse primer to the well
        well_data.append({
            'Well Position': well_position,
            'Sequence Name': f"{primer.mutation}_rev",
            'Sequence': primer.rev
        })
    return pd.DataFrame(well_data)
