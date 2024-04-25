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

        tmp_dir = "/tmp/mmseqs"

        command = [
            "mmseqs",
            "databases",
            db_name,
            os.path.join(PROTEIN_DBS_PATH, local_db_name),
            tmp_dir
        ]
        subprocess.run(command, check=True)
        VOLUME_CONFIG[PROTEIN_DBS_PATH].commit()

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
    m.download_db.remote("UniProtKB/TrEMBL", "uniprot_trembl")
    # print(m.search_sequence.remote("MSGKIDKILIVGGGTAGWMAASYLGKALQGTADITLLQAPDIPTLGVGEATIPNLQTAFFDFLGIPEDEWMRECNASYKVAIKFINWRTAGEGTSEARELDGGPDHFYHSFGLLKYHEQIPLSHYWFDRSYRGKTVEPFDYACYKEPVILDANRSPRRLDGSKVTNYAWHFDAHLVADFLRRFATEKLGVRHVEDRVEHVQRDANGNIESVRTATGRVFDADLFVDCSGFRGLLINKAMEEPFLDMSDHLLNDSAVATQVPHDDDANGVEPFTSAIAMKSGWTWKIPMLGRFGTGYVYSSRFATEDEAVREFCEMWHLDPETQPLNRIRFRVGRNRRAWVGNCVSIGTSSCFVEPLESTGIYFVYAALYQLVKHFPDKSLNPVLTARFNREIETMFDDTRDFIQAHFYFSPRTDTPFWRANKELRLADGMQEKIDMYRAGMAINAPASDDAQLYYGNFEEEFRNFWNNSNYYCVLAGLGLVPDAPSPRLAHMPQATESVDEVFGAVKDRQRNLLETLPSLHEFLRQQHGR", "pfam_b"))
