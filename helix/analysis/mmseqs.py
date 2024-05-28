from modal import Image, method
from helix.core import app, volumes
import subprocess
import os


image = Image.micromamba().apt_install("wget", "git", "tar").micromamba_install(
    "mmseqs2",
    channels=[
        "bioconda",
        "conda-forge"
    ],
).pip_install("jupyter", "pandas", "numpy")

tmp_dir = "/tmp/mmseqs"

DATABASES_PATH = "/mnt/databases"


@app.cls(
    image=image,
    volumes={DATABASES_PATH: volumes.mmseqs_databases},
    timeout=3600*10,
    cpu=10.0,
    memory=250.0,
)
class MMSeqs:

    def __init__(self):
        self.local_databases = self._get_local_databases()

    @method()
    def download_db(self, db_name, local_db_name):
        """
        Download and set up a database using the MMSeqs2 'databases' command with sensible defaults.

        Args:
            db_name (str): The name of the database to download (e.g., 'UniProtKB/Swiss-Prot').
            local_db_name (str): The name to use for the local database in PROTEIN_DBS_PATH

        This method assumes that the MMSeqs2 'databases' command is available and configured properly.
        """
        import subprocess
        import os

        local_db_path = os.path.join(DATABASES_PATH, local_db_name)

        # Check if the local database already exists before attempting to download
        if not os.path.exists(local_db_path):
            command = [
                "mmseqs",
                "databases",
                db_name,
                local_db_path,
                tmp_dir
            ]
            subprocess.run(command, check=True)
            volumes.mmseqs_databases.commit()
        else:
            print(
                f"Database {local_db_name} already exists at {local_db_path}.")

    def _get_local_databases(self):
        """
        List all databases that have been downloaded and are available in the local storage.
        This method now simply lists the names of the databases without detailing the associated files.

        Returns:
            list: A list of database names available in the local storage.
        """
        import os

        # Ensure the directory exists
        if not os.path.exists(DATABASES_PATH):
            print(
                f"No databases found. Directory {DATABASES_PATH} does not exist.")
            return []

        # List only .source files in the database path
        database_files = [f for f in os.listdir(
            DATABASES_PATH) if f.endswith('.source')]
        databases = set()
        if not database_files:
            print("No databases have been downloaded yet.")
        else:
            print("Downloaded databases:")
            for db_file in database_files:
                db_name = db_file.replace('.source', '')
                databases.add(db_name)

            for db in databases:
                print(db)

        return list(databases)

    @method()
    def search_sequence(self, sequence, db_name):
        """
        Search a given sequence against a specified database using MMSeqs2 and store the results.

        Args:
            sequence (str): The protein sequence to search.
            db_name (str): The name of the database to search against.
        """
        import tempfile
        db_path = os.path.join(DATABASES_PATH, db_name)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
            tmpfile.write(f">{tmpfile.name}\n{sequence}\n")
            tmpfile_path = tmpfile.name
        result_path = "result.m8"

        # Run the search
        subprocess.run([
            "mmseqs",
            "easy-search",
            tmpfile_path,
            db_path,
            result_path,
            "tmp"
        ], check=True)

        import pandas as pd

        # Define the column headers
        columns = [
            "query", "target", "pident", "alnlen", "mismatch", "gapopen",
            "qstart", "qend", "tstart", "tend", "evalue", "bits"
        ]

        # Read the results into a pandas DataFrame
        df = pd.read_csv(result_path, sep='\t', header=None, names=columns)

        return df

    @method()
    def align(self, sequences):
        """
        Perform local alignment of a query sequence against multiple target sequences using MMSeqs2.
        This method creates temporary databases from the provided sequences array, aligns them to the query,
        and calculates alignment statistics.

        Args:
            sequences (list of str): List of target sequences to be aligned.

        Returns:
            str: Alignment results as formatted string.
        """
        import tempfile
        import subprocess

        # Create a temporary database from the input sequences
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
            for i, seq in enumerate(sequences):
                tmpfile.write(f">{i}\n{seq}\n")
            sequences_db_path = tmpfile.name

        subprocess.run([
            "mmseqs",
            "createdb",
            sequences_db_path,
            sequences_db_path + "_db"
        ], check=True)

        # Prefiltering step for all-against-all sequence comparison
        with tempfile.NamedTemporaryFile(delete=False) as result_db_pref:
            subprocess.run([
                "mmseqs",
                "prefilter",
                sequences_db_path + "_db",
                sequences_db_path + "_db",
                result_db_pref.name
            ], check=True)

        # Perform alignment
        with tempfile.NamedTemporaryFile(delete=False) as result_db_aln:
            subprocess.run([
                "mmseqs", "align",
                sequences_db_path + "_db",
                sequences_db_path + "_db",
                result_db_pref.name,
                result_db_aln.name,
            ], check=True)

        # Read and return the alignment results
        with open(result_db_aln.name, 'r') as result_file:
            results = result_file.read()

        return results

    @method()
    def cluster_sequences(self, sequences, ids: list, sequence_identity_threshold: float):
        """
        Clusters sequences using the MMSeqs2 easy-cluster method with a specified sequence identity threshold.

        Args:
            sequences (list of str): List of sequences to be clustered.
            ids (list of str): List of identifiers corresponding to the sequences.
            sequence_identity_threshold (float): The minimum sequence identity threshold for clustering.

        Returns:
            dict: A dictionary where keys are cluster identifiers and values are lists of sequence ids belonging to that cluster.
        """
        import tempfile
        import subprocess

        # Create a temporary FASTA file with sequences and their IDs
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as fasta_file:
            for seq_id, seq in zip(ids, sequences):
                fasta_file.write(f">{seq_id}\n{seq}\n")
            fasta_path = fasta_file.name

        # Define random temporary directory for MMSeqs2
        cluster_result_path = tmp_dir + "/cluster_result"

        # Run MMSeqs2 easy-cluster with the specified sequence identity threshold
        subprocess.run([
            "mmseqs",
            "easy-cluster",
            fasta_path,
            cluster_result_path,
            tmp_dir,
            "--min-seq-id",
            str(sequence_identity_threshold)
        ], check=True)

        # Define paths for cluster results
        cluster_result_path + "_all_seqs.fasta"
        cluster_result_cluster_tsv_path = cluster_result_path + "_cluster.tsv"
        cluster_result_path + "_rep_seq.fasta"

        # Read cluster results and return as a DataFrame
        import pandas as pd
        return pd.read_csv(cluster_result_cluster_tsv_path, sep='\t', header=None, names=['cluster_id', 'sequence_id'])
