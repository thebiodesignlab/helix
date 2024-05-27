from modal import Image
from helix.core import app, volumes

ROSETTA_DEPS_PATH = '/vol/rosetta-deps'

rosettafold_image = (
    Image.micromamba()
    .apt_install("wget", "git", "tar", "gcc", "g++", "make", "aria2")
    .micromamba_install(
        "cudatoolkit=11.1",
        "cudnn=8.8.0",
        "tensorflow=2.11.0",
        channels=[
            "predector",
            "pyg",
            "bioconda",
            "pytorch",
            "nvidia",
            "biocore",
            "conda-forge"
        ],
    )
    .pip_install(
        "antlr4-python3-runtime==4.9.3",
        "assertpy==1.1",
        "configparser==6.0.1",
        "git+https://github.com/NVIDIA/dllogger.git@0540a43971f4a8a16693a9de9de73c1072020769",
        "docker-pycreds==0.4.0",
        "e3nn==0.3.3",
        "gitdb==4.0.11",
        "gitpython==3.1.42",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "opt-einsum-fx==0.1.4",
        "pathtools==0.1.2",
        "promise==2.3",
        "pynvml==11.0.0",
        "pyrsistent==0.20.0",
        "pyyaml",
        "shortuuid",
        "smmap==5.0.1",
        "subprocess32==3.5.4",
        "wandb")
    .run_commands(
        "git clone https://github.com/baker-laboratory/RoseTTAFold-All-Atom")
    .workdir("/RoseTTAFold-All-Atom")
)


@app.function(
    image=rosettafold_image,
    # secret=Secret.from_name("signalp-license-secret"),
    memory=6400,
    cpu=8,
    volumes={ROSETTA_DEPS_PATH: volumes.rosetta},
    timeout=3600*10,
    _allow_background_volume_commits=True
)
def download_dependencies():
    import platform
    import subprocess
    from pathlib import Path

    # Determine the platform and adjust for 64-bit if necessary
    platform.system().lower() + platform.architecture()[0][:-3]

    # Define the base directory for downloads
    base_dir = Path(ROSETTA_DEPS_PATH)

    # Define the list of commands for downloading and extracting files
    commands = []

    # Define the download and extraction steps
    download_steps = [
        {
            "url": "https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz",
            "dir": base_dir / "pdb100_2021Mar03",
            "tar_file": base_dir / "pdb100_2021Mar03.tar.gz"
        },
        {
            "url": "https://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz",
            "dir": base_dir / "UniRef30_2020_06",
            "tar_file": base_dir / "UniRef30_2020_06_hhsuite.tar.gz"
        },
        {
            "url": "https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz",
            "dir": base_dir / "bfd",
            "tar_file": base_dir / "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz"
        },
        {
            "url": "https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz",
            "dir": base_dir / "blast-2.2.26",
            "tar_file": base_dir / "blast-2.2.26-x64-linux.tar.gz",
            "post_commands": [
                f"cp -r {base_dir / 'blast-2.2.26/blast-2.2.26/'} {base_dir / 'blast-2.2.26_bk'}",
                f"rm -r {base_dir / 'blast-2.2.26'}",
                f"mv {base_dir / 'blast-2.2.26_bk/'} {base_dir / 'blast-2.2.26'}"
            ]
        }
    ]

    for step in download_steps:
        if not step["dir"].exists():
            if not step["tar_file"].exists():
                commands.append(
                    f"aria2c -x 16 -s 16 {step['url']} -d {base_dir}")
            commands.append(f"mkdir -p {step['dir']}")
            commands.append(f"tar xfz {step['tar_file']}")
            if "post_commands" in step:
                commands.extend(step["post_commands"])

    for cmd in commands:
        print(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


@app.function(
    image=rosettafold_image,
    # secret=Secret.from_name("signalp-license-secret"),
    # gpu=gpu.A100(memory=40),
    volumes={
        ROSETTA_DEPS_PATH: volumes.rosetta
    },
    timeout=3600*10
)
def run_rosettafold(config_path: str, out_dir: str = "/shared"):
    import os
    os.system("signalp6-register /run/secrets/signalp-license")
    # Move the distilled model weights
    os.system("mv /mambaforge/lib/python3.10/site-packages/signalp/model_weights/distilled_model_signalp6.pt /mambaforge/lib/python3.10/site-packages/signalp/model_weights/ensemble_model_signalp6.pt")
    os.system(f"python -m rf2aa.run_inference --config-name {config_path}")
    os.system(f"cp -r {out_dir} /shared")


@ app.local_entrypoint()
def main():
    download_dependencies.remote()
    # run_rosettafold.remote("rf2aa/config/inference/base.yaml")
