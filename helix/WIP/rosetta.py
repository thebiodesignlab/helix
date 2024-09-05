import subprocess
import time
import threading
import tarfile
import sys
import pathlib
import os
from modal import Image
from helix.core import app, volumes

ROSETTA_DEPS_PATH = '/vol/rosetta-deps'

rosettafold_image = (
    Image.micromamba()
    .apt_install("wget", "git", "tar", "gcc", "g++", "make", "aria2", "rsync")
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


def start_monitoring_disk_space(interval: int = 30, stop_event: threading.Event = None) -> None:
    """Start monitoring the disk space in a separate thread."""
    task_id = os.environ["MODAL_TASK_ID"]

    def log_disk_space(interval: int, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            statvfs = os.statvfs('/')
            free_space = statvfs.f_frsize * statvfs.f_bavail
            print(
                f"{task_id} free disk space: {free_space / (1024 ** 3):.2f} GB", file=sys.stderr)
            time.sleep(interval)

    stop_event = stop_event or threading.Event()
    monitoring_thread = threading.Thread(
        target=log_disk_space, args=(interval, stop_event))
    monitoring_thread.daemon = True
    monitoring_thread.start()
    return stop_event


def decompress_file(file_path: pathlib.Path, extract_dir: pathlib.Path) -> None:
    print(f"Decompressing {file_path} into {extract_dir}...")
    try:
        subprocess.run(["tar", "-xf", str(file_path),
                       "-C", str(extract_dir)], check=True)
        print(f"Decompressed {file_path} to {extract_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to decompress {file_path}: {e}")
        raise


@app.function(
    volumes={"/mnt/": volumes.rosetta},
    timeout=60 * 60 * 12,  # 12 hours,
    image=rosettafold_image,
    ephemeral_disk=2500 * 1024,
)
def import_transform_load(
    datasets: dict = {
        "small_dataset": {
            "url": "https://github.com/DocSpring/geolite2-city-mirror/raw/master/GeoLite2-City.tar.gz",
            "path": "/tmp/GeoLite2-City.tar.gz",
            "decompressed_path": "/mnt/rosettafold/mapped_places",
            "size": "1KB"
        },
        "uniref30": {
            "url": "https://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz",
            "path": "/tmp/UniRef30_2020_06_hhsuite.tar.gz",
            "decompressed_path": "/mnt/rosettafold/UniRef30_2020_06",
            "size": "46G"
        },
        "bfd": {
            "url": "https://storage.googleapis.com/alphafold-databases/casp14_versions/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz",
            "path": "/tmp/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz",
            "decompressed_path": "/mnt/rosettafold/bfd",
            "size": "272G"
        },
        "structure_templates": {
            "url": "https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz",
            "path": "/tmp/pdb100_2021Mar03.tar.gz",
            "decompressed_path": "/mnt/rosettafold/pdb100_2021Mar03",
            "size": "unknown"
        }
    }
) -> None:
    errors = []

    def download_with_aria2c(url, output_path):
        try:
            subprocess.run(["aria2c", "-x", "16", "-s", "16", "-k", "1M", "-d",
                           str(output_path.parent), "-o", output_path.name, url], check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"aria2c failed: {e}")

    def decompress_tar_gz(file_path, temp_dir):
        try:
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=temp_dir)
        except Exception as e:
            raise Exception(f"Failed to decompress {file_path}: {e}")

    def move_files_parallel(src_dir, dest_dir):
        try:
            subprocess.run(["rsync", "-a", "--info=progress2", "--remove-source-files",
                           "--no-times", "--inplace", f"{src_dir}/", f"{dest_dir}/"], check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(
                f"Failed to move files from {src_dir} to {dest_dir}: {e}")

    for dataset_name, dataset_info in datasets.items():
        try:
            local_path = pathlib.Path(dataset_info["path"])
            decompressed_path = pathlib.Path(dataset_info["decompressed_path"])
            temp_decompressed_path = pathlib.Path(
                "/tmp") / decompressed_path.name

            if not decompressed_path.exists() or not any(decompressed_path.iterdir()):
                print(f"Downloading {dataset_name} [{dataset_info['size']}]")
                download_with_aria2c(dataset_info["url"], local_path)
                print(f"Downloaded {dataset_name} to {local_path}")

                print(
                    f"Decompressing {dataset_name} to temporary directory {temp_decompressed_path}")
                decompress_tar_gz(local_path, temp_decompressed_path)
                print(
                    f"Successfully decompressed {dataset_name} to temporary directory {temp_decompressed_path}")

                print(
                    f"Moving decompressed files to final directory {decompressed_path}")
                move_files_parallel(temp_decompressed_path, decompressed_path)
                print(
                    f"Successfully moved {dataset_name} to {decompressed_path}")

        except Exception as e:
            errors.append(
                f"Failed to download and decompress {dataset_name}: {e}")

    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        raise Exception("Failed to download and decompress all datasets")

    print("All decompression tasks completed.")
    print("Dataset is loaded ✅")


def run_rosettafold(config_path: str, out_dir: str = "/shared"):
    import subprocess
    subprocess.run(
        ["signalp6-register", "/run/secrets/signalp-license"], check=True)
    # Move the distilled model weights
    subprocess.run(["mv", "/mambaforge/lib/python3.10/site-packages/signalp/model_weights/distilled_model_signalp6.pt",
                   "/mambaforge/lib/python3.10/site-packages/signalp/model_weights/ensemble_model_signalp6.pt"], check=True)
    subprocess.run(["python", "-m", "rf2aa.run_inference",
                   "--config-name", config_path], check=True)
    subprocess.run(["cp", "-r", out_dir, "/shared"], check=True)


@app.local_entrypoint()
def main():
    import_transform_load.remote()
    # run_rosettafold.remote("rf2aa/config/inference/base.yaml")
