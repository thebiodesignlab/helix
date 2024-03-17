

from dnachisel import AvoidPattern, DnaOptimizationProblem, CodonOptimize, EnforceTranslation, reverse_translate
from .main import stub
from modal import Image
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
image = Image.debian_slim().pip_install(
    "dnachisel", "biopython", "biotite", "pandas", "primers")


@stub.function(image=image)
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


@stub.function(image=image)
def create_kdg_primers(plasmid_sequence: str, gene_location: tuple, mutations: list):
    import pandas as pd
    from primers import create
    from biotite.sequence import CodonTable
    table = CodonTable.default_table()

    # Initialize an empty list to store primer data
    primer_data = []

    for mutation in mutations:
        mutation_position = int(mutation[1:-1])
        plasmid_sequence[gene_location[0] + mutation_position *
                         3 - 3:gene_location[0] + mutation_position * 3]
        mutated_codon = str(table[mutation[-1]][0])
        mutated_sequence = plasmid_sequence[:gene_location[0] + mutation_position * 3 -
                                            3] + mutated_codon + plasmid_sequence[gene_location[0] + mutation_position * 3:]

        # Circularize so that mutated codon is first in the sequence
        circular_plasmid_sequence = mutated_sequence[gene_location[0] + mutation_position *
                                                     3 - 3:] + mutated_sequence[:gene_location[0] + mutation_position * 3 - 3]

        # Create primers
        fwd, rev = create(circular_plasmid_sequence)

        # Append primer data to the list
        primer_data.append({
            'mutation': mutation,
            'fwd': fwd.seq,
            'rev': rev.seq,
            'fwd_tm': fwd.tm,
            'rev_tm': rev.tm,
            'fwd_gc': fwd.gc,
            'rev_gc': rev.gc,
            'fwd_length': len(fwd.seq),
            'rev_length': len(rev.seq)
        })

    # Create a DataFrame from the list of primer data
    df = pd.DataFrame(primer_data)
    return df


@stub.local_entrypoint()
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


@stub.local_entrypoint()
def create_kdg_primers_to_csv(plasmid_sequence: str, gene_location: tuple, mutations: list, output_path: str):
    """
    Create primers for a list of mutations in a plasmid sequence.
    Parameters
    ----------
    plasmid_sequence : str
        The plasmid sequence to create primers for.
    gene_location : tuple
        The start and end positions of the gene in the plasmid sequence starting from 1.
    mutations : list
        A list of mutations in the form "A123T" where A is the wildtype amino acid, 123 is the position, and T is the mutant amino acid.
    output_path : str
        The path to save the primers to.
    """
    df = create_kdg_primers(plasmid_sequence, gene_location, mutations)
    df.to_csv(output_path)
