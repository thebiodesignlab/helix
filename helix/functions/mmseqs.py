from modal import method
from helix.core import app, volumes, images
import subprocess
from pathlib import Path
import json

tmp_dir = "/tmp/mmseqs"
DATABASES_PATH = "/mnt/databases"
DB_DICT_PATH = Path(DATABASES_PATH) / "mmseqs_db_dict.json"


def load_db_dict():
    if DB_DICT_PATH.exists():
        with open(DB_DICT_PATH, 'r') as f:
            return {k.lower(): v for k, v in json.load(f).items()}
    return {}


def save_db_dict(db_dict):
    with open(DB_DICT_PATH, 'w') as f:
        json.dump(db_dict, f)


@app.cls(
    image=images.mmseqs,
    volumes={DATABASES_PATH: volumes.mmseqs_databases},
    timeout=3600*10,
    cpu=10.0,
    memory=32768,
)
class MMSeqs:

    def __init__(self):
        self.db_dict = load_db_dict()

    def generate_unique_db_name(self):
        """
        Generate a unique database name for storing downloaded databases.

        Returns:
            str: A unique database name.
        """
        import uuid
        return str(uuid.uuid4()).replace("-", "_")

    def get_local_db_path(self, db_name):
        """
        Get the local path of a database by its name. If the database does not exist, raise an error and provide instructions to download it.

        Args:
            db_name (str): The name of the database.

        Returns:
            str: The local path of the database.

        Raises:
            KeyError: If the database does not exist in the dictionary.
        """
        db_name_lower = db_name.lower()
        if db_name_lower not in self.db_dict:
            raise KeyError(
                f"Database '{db_name}' does not exist. Please download it using the 'download_db.remote('{db_name}')' method.")
        elif not (Path(DATABASES_PATH) / self.db_dict[db_name_lower]):
            self.db_dict.pop(db_name_lower)
            save_db_dict(self.db_dict)
            raise FileNotFoundError(
                f"Database '{db_name}' not found at path {DATABASES_PATH}/{self.db_dict[db_name_lower]}. Try downloading it again.")
        return Path(DATABASES_PATH) / self.db_dict[db_name_lower]

    @method()
    def download_db(self, db_name):
        """
        Download and set up a database using the MMSeqs2 'databases' command with sensible defaults.

        Args:
            db_name (str): The name of the database to download (e.g., 'UniProtKB/Swiss-Prot').

        This method assumes that the MMSeqs2 'databases' command is available and configured properly.
        """
        import os

        db_name_lower = db_name.lower()
        local_db_name = self.generate_unique_db_name()
        local_db_path = os.path.join(DATABASES_PATH, local_db_name)

        # Check if the local database already exists before attempting to download
        if db_name_lower not in self.db_dict:
            command = [
                "mmseqs",
                "databases",
                db_name,
                local_db_path,
                tmp_dir
            ]
            subprocess.run(command, check=True)
            volumes.mmseqs_databases.commit()
            self.db_dict[db_name_lower] = local_db_name
            save_db_dict(self.db_dict)
        else:
            print(
                f"Database {db_name} already exists at {local_db_path}.")

    @method()
    def get_downloaded_databases(self):
        """
        List all databases that have been downloaded and are available in the local storage.
        This method now simply lists the names of the databases without detailing the associated files.

        Returns:
            dict: A dictionary of database names and their corresponding local names available in the local storage.
        """
        databases = {}
        for db_name, local_db_name in self.db_dict.items():
            if (Path(DATABASES_PATH) / f"{local_db_name}.source").exists():
                databases[db_name] = local_db_name
                print(f"{db_name}: {local_db_name}")
            else:
                print(
                    f"Warning: {db_name} is in db_dict but .source file not found in {DATABASES_PATH}")

        if not databases:
            print(f"No databases found in {DATABASES_PATH}")

        return databases

    @method()
    def search_sequence(self, sequence, db_name):
        """
        Search a given sequence against a specified database using MMSeqs2 and store the results.

        Args:
            sequence (str): The protein sequence to search.
            db_name (str): The name of the database to search against.
        """
        import tempfile

        # Use the get_local_db_path method to get the database path
        db_path = self.get_local_db_path(db_name)

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

    @ method()
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

    @ method()
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

    @ method()
    def create_hmm_profiles(self, msas, match_mode=0, match_ratio=0.5):
        """
        Create HMM profiles from a list of multiple sequence alignments (MSAs) using MMSeqs2.

        Args:
            msas (list of str): List of MSAs in FASTA, A3M, or CA3M format.
            match_mode (int, optional): Profile column assignment mode. Default is 0.
            match_ratio (float, optional): Gap fraction threshold for profile column assignment. Default is 0.5.

        Returns:
            list: A list of IDs corresponding to the created HMM profiles.
        """
        import tempfile
        import subprocess
        import os
        import uuid

        profile_ids = []

        for i, msa in enumerate(msas):
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.msa') as msa_file:
                msa_file.write(msa)
                msa_path = msa_file.name

            msa_db_path = msa_path + "_db"
            profile_id = str(uuid.uuid4())
            final_hmm_profile_path = os.path.join(DATABASES_PATH, profile_id)

            # Convert MSA to MMSeqs2 database
            subprocess.run([
                "mmseqs",
                "convertmsa",
                msa_path,
                msa_db_path
            ], check=True)

            # Create HMM profile from MSA database
            subprocess.run([
                "mmseqs",
                "msa2profile",
                msa_db_path,
                final_hmm_profile_path,
                "--match-mode",
                str(match_mode),
                "--match-ratio",
                str(match_ratio)
            ], check=True)

            profile_ids.append(profile_id)

        return profile_ids

    @method()
    def cluster_and_search_db(self, sequences, target_db_name, cluster_mode=2, min_seq_id=0.9):
        """
        Cluster a set of sequences, build a sub-database of the representative sequences,
        and use that to search sequences in a specified database using the profiles of the representative sequences.

        Args:
            sequences (list of str): List of sequences to be clustered.
            target_db_name (str): Name of the target database to search against. Example: 'UniProtKB/Swiss-Prot'.
            cluster_mode (int, optional): Clustering mode for MMSeqs2. Default is 2.
            min_seq_id (float, optional): Minimum sequence identity for clustering. Default is 0.9.

        Returns:
            list: A list of search results.
        """
        import tempfile
        import subprocess
        import uuid

        target_db_path = self.get_local_db_path(target_db_name)

        # Create temporary files for input sequences
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as seq_file:
            for seq in sequences:
                seq_file.write(f">{uuid.uuid4()}\n{seq}\n")
            seq_file_path = seq_file.name

        # Define paths for intermediate and final outputs
        seq_db_path = seq_file_path + "_db"
        cluster_db_path = seq_db_path + "_cluster"
        rep_seq_db_path = seq_db_path + "_repseq"
        rep_seq_db_h_path = seq_db_path + "_repseq_h"
        profile_db_path = rep_seq_db_path + "_profile"
        result_db_path = profile_db_path + "_result"

        # Convert sequences to MMSeqs2 database
        subprocess.run([
            "mmseqs",
            "createdb",
            seq_file_path,
            seq_db_path
        ], check=True)

        # Cluster sequences
        subprocess.run([
            "mmseqs",
            "cluster",
            seq_db_path,
            cluster_db_path,
            tmp_dir,
            "--min-seq-id",
            str(min_seq_id),
            "--cluster-mode",
            str(cluster_mode)
        ], check=True)

        # Create representative sequence databases
        subprocess.run([
            "mmseqs",
            "createsubdb",
            cluster_db_path,
            seq_db_path,
            rep_seq_db_path
        ], check=True)

        subprocess.run([
            "mmseqs",
            "createsubdb",
            cluster_db_path,
            seq_db_path,
            rep_seq_db_h_path
        ], check=True)

        # Create profiles from representative sequences
        subprocess.run([
            "mmseqs",
            "result2profile",
            rep_seq_db_path,
            seq_db_path,
            cluster_db_path,
            profile_db_path
        ], check=True)

        # Search target database using representative profiles
        subprocess.run([
            "mmseqs",
            "search",
            profile_db_path,
            target_db_path,
            result_db_path,
            tmp_dir
        ], check=True)

        import pandas as pd

        # Convert search results to readable format with taxonomy information
        result_tsv_path = result_db_path + ".tsv"
        subprocess.run([
            "mmseqs",
            "convertalis",
            profile_db_path,
            target_db_path,
            result_db_path,
            result_tsv_path,
            "--format-mode", "4",
            "--format-output", "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,taxid,taxname,taxlineage,qseq,tseq,qcov,tcov"
        ], check=True)

        # Define the column headers including taxonomy information and additional columns
        columns = [
            "query", "target", "pident", "alnlen", "mismatch", "gapopen",
            "qstart", "qend", "tstart", "tend", "evalue", "bits",
            "taxid", "taxname", "taxlineage", "qseq", "tseq", "qcov", "tcov"
        ]

        # Read the results into a DataFrame with headers and return
        df = pd.read_csv(result_tsv_path, sep='\t', header=0, names=columns)
        return df
