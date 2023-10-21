

from dnachisel import AvoidPattern, DnaOptimizationProblem, CodonOptimize, EnforceTranslation, reverse_translate
from .main import stub
from modal import Image
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
image = Image.debian_slim().pip_install("dnachisel", "biopython")


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
