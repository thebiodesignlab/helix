from modal import Image, Stub
from .main import VOLUME_CONFIG, PROTEIN_DBS_PATH
stub = Stub("rosettafold")


rosettafold_image = (
    Image.micromamba()
    .apt_install("wget", "git", "tar", "gcc", "g++", "make")
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
    .run_commands("wget -nv http://wwwuser.gwdg.de/~compbiol/data/csblast/releases/csblast-2.2.3_$(uname -s | tr '[:upper:]' '[:lower:]')$(uname -m | grep -o '..$').tar.gz -O csblast-2.2.3.tar.gz", "mkdir -p csblast-2.2.3", "tar xf csblast-2.2.3.tar.gz -C csblast-2.2.3 --strip-components=1")
    .run_commands(
        "wget -nv http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz",
        "mkdir -p UniRef30_2020_06",
        "tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06"
    )
    .run_commands(
        "wget -nv http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFAA_paper_weights.pt"
    )
    # .run_commands(
    #     "wget -nv https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz")
    # .run_commands(
    #     "mkdir -p bfd")
    # .run_commands(
    #     "tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd"
    # )
    # .run_commands(
    #     "wget -nv https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz",
    #     "tar xfz pdb100_2021Mar03.tar.gz"
    # )
)


@stub.function(
    image=rosettafold_image,
    # secret=Secret.from_name("signalp-license-secret"),
    # gpu=gpu.A100(memory=40),
    volumes=VOLUME_CONFIG,
    timeout=3600*10
)
def download_dependencies():
    import platform
    import subprocess
    from pathlib import Path

    # Determine the platform and adjust for 64-bit if necessary
    platform.system().lower() + platform.architecture()[0][:-3]

    # Define the base directory for downloads
    base_dir = Path(PROTEIN_DBS_PATH)

    # Define the list of commands for downloading and extracting files
    commands = [
        "wget -nv https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -O " +
        str(base_dir / "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz"),
        "mkdir -p " + str(base_dir / "bfd"),
        "tar xfz " + str(base_dir / "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz") +
        " -C " + str(base_dir / "bfd")
    ]

    for cmd in commands:
        print(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        VOLUME_CONFIG[PROTEIN_DBS_PATH].commit()


@stub.function(
    image=rosettafold_image,
    # secret=Secret.from_name("signalp-license-secret"),
    # gpu=gpu.A100(memory=40),
    volumes=VOLUME_CONFIG,
    timeout=3600*10
)
def run_rosettafold(config_path: str, out_dir: str = "/shared"):
    import os
    os.system("signalp6-register /run/secrets/signalp-license")
    # Move the distilled model weights
    os.system("mv /mambaforge/lib/python3.10/site-packages/signalp/model_weights/distilled_model_signalp6.pt /mambaforge/lib/python3.10/site-packages/signalp/model_weights/ensemble_model_signalp6.pt")
    os.system(f"python -m rf2aa.run_inference --config-name {config_path}")
    os.system(f"cp -r {out_dir} /shared")


@stub.local_entrypoint()
def main():
    download_dependencies.remote()
    # run_rosettafold.remote("rf2aa/config/inference/base.yaml")
