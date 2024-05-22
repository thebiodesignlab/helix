import time
from modal import Image, App, method
import modal
from .main import PROTEIN_DBS_PATH, VOLUME_CONFIG
import subprocess
import os
app = App(name="helix-mmseqs")


image = Image.micromamba().apt_install("wget", "git", "tar").micromamba_install(
    "mmseqs2",
    channels=[
        "bioconda",
        "conda-forge"
    ],
).pip_install("jupyter")

tmp_dir = "/tmp/mmseqs"


@app.cls(
    image=image,
    volumes=VOLUME_CONFIG,
    timeout=3600*10,
    cpu=8.0,
    memory=6768
)
class MMSeqs:
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

        local_db_path = os.path.join(PROTEIN_DBS_PATH, local_db_name)

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
            VOLUME_CONFIG[PROTEIN_DBS_PATH].commit()
        else:
            print(
                f"Database {local_db_name} already exists at {local_db_path}.")

    @method()
    def search_sequence(self, sequence, db_name):
        """
        Search a given sequence against a specified database using MMSeqs2 and store the results.

        Args:
            sequence (str): The protein sequence to search.
            db_name (str): The name of the database to search against.
        """
        import tempfile
        db_path = os.path.join(PROTEIN_DBS_PATH, db_name)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
            tmpfile.write(f">{tmpfile.name}\n{sequence}\n")
            tmpfile_path = tmpfile.name
        result_path = tmpfile_path + "_result"
        # Create a database from the input sequence
        subprocess.run([
            "mmseqs",
            "createdb",
            tmpfile_path,
            tmpfile_path + "_db"
        ], check=True)

        # Run the search
        subprocess.run([
            "mmseqs",
            "easy-search",
            tmpfile_path,
            db_path,
            result_path,
            "/tmp/mmseqs",
            "--format-mode",
            "0",
        ], check=True)

        with open(result_path, 'r') as file:
            return file.read()

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

        # Read cluster results
        with open(cluster_result_cluster_tsv_path, 'r') as cluster_file:
            return cluster_file.readlines()


@app.function(concurrency_limit=1, _allow_background_volume_commits=True)
def run_jupyter(timeout: int):
    import subprocess
    import os
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": "abc"},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(
                f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main():
    m = MMSeqs()
    # m.download_db.remote("UniProtKB/TrEMBL", "uniprot_trembl")
    print(m.search_sequence.remote("MSGKIDKILIVGGGTAGWMAASYLGKALQGTADITLLQAPDIPTLGVGEATIPNLQTAFFDFLGIPEDEWMRECNASYKVAIKFINWRTAGEGTSEARELDGGPDHFYHSFGLLKYHEQIPLSHYWFDRSYRGKTVEPFDYACYKEPVILDANRSPRRLDGSKVTNYAWHFDAHLVADFLRRFATEKLGVRHVEDRVEHVQRDANGNIESVRTATGRVFDADLFVDCSGFRGLLINKAMEEPFLDMSDHLLNDSAVATQVPHDDDANGVEPFTSAIAMKSGWTWKIPMLGRFGTGYVYSSRFATEDEAVREFCEMWHLDPETQPLNRIRFRVGRNRRAWVGNCVSIGTSSCFVEPLESTGIYFVYAALYQLVKHFPDKSLNPVLTARFNREIETMFDDTRDFIQAHFYFSPRTDTPFWRANKELRLADGMQEKIDMYRAGMAINAPASDDAQLYYGNFEEEFRNFWNNSNYYCVLAGLGLVPDAPSPRLAHMPQATESVDEVFGAVKDRQRNLLETLPSLHEFLRQQHGR", "uniprot_trembl"))


@app.local_entrypoint()
def cluster_sequences_from_csv(csv_path):
    import pandas as pd
    import json
    import csv
    import os

    # Load CSV file containing sequences and their IDs
    data = pd.read_csv(csv_path)
    if 'sequence' not in data.columns or 'id' not in data.columns:
        raise ValueError("CSV must contain 'sequence' and 'id' columns")

    # Initialize MMSeqs object
    m = MMSeqs()

    # Prepare sequences and ids
    sequences = data['sequence'].tolist()
    ids = data['id'].tolist()
    for seq_identity_threshold in [0, 0.4, 0.6, 0.7, 0.9]:

        # Run clustering
        cluster_tsv = m.cluster_sequences.remote(
            sequences, ids, seq_identity_threshold)

        # Parse cluster results
        clusters = {}
        for line in cluster_tsv:
            cluster_id, seq_id = line.strip().split('\t')
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(seq_id)

        # Define output filenames based on input CSV path
        csv_output_path = os.path.splitext(
            csv_path)[0] + '_clusters_' + str(seq_identity_threshold) + '.csv'
        json_output_path = os.path.splitext(
            csv_path)[0] + ' _clusters_' + str(seq_identity_threshold) + '.json'

        # Write clusters to CSV
        with open(csv_output_path, 'w', newline='') as csvfile:
            fieldnames = ['Representative Sequence', 'Cluster Members']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for cluster_id, seq_ids in clusters.items():
                writer.writerow({'Representative Sequence': cluster_id,
                                'Cluster Members': ','.join(seq_ids)})

        # Write clusters to JSON
        with open(json_output_path, 'w') as jsonfile:
            json.dump(clusters, jsonfile)


# Example usage:
# cluster_sequences_from_csv('path/to/your/sequences.csv')
