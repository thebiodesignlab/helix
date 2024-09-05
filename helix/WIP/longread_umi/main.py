
import os
from modal import Image
from helix.core import app


longread_umi_image = (
    Image
    .micromamba()
    .apt_install("git", "wget", "make", "g++", "bsdmainutils", "gawk")
    .copy_local_file(os.path.join(os.path.dirname(__file__), "install_conda.sh"), "install_conda.sh")
    .run_commands(
        "bash ./install_conda.sh"
    )
)


@app.function(image=longread_umi_image)
def run():
    import subprocess
    # Set the active Micromamba environment directly from Python using bash
    # Modified to capture and print stdout and stderr for better debugging
    base_cmd = 'eval "$(micromamba shell hook --shell=bash)" && micromamba activate longread_umi && '
    cmd = "longread_umi nanopore_pipeline -h"

    subprocess.call(base_cmd+cmd, shell=True, executable='/bin/bash')


@ app.local_entrypoint()
def main():
    run.remote()
