from .common import stub, image
import modal
import os

N_GPUS = int(os.environ.get("N_GPUS", 2))
GPU_CONFIG = os.environ.get("GPU_CONFIG", modal.gpu.H100(count=N_GPUS))


def run_cmd(cmd: str, run_folder: str):
    import subprocess

    # Ensure volumes contain latest files.
    # VOLUME_CONFIG["/confit/predicted"].reload()
    # VOLUME_CONFIG["/confit/checkpoint"].reload()

    # Propagate errors from subprocess.
    if exit_code := subprocess.call(cmd.split(), cwd=run_folder):
        exit(exit_code)

    # Commit writes to volume.
    # VOLUME_CONFIG["/confit/predicted"].commit()
    # VOLUME_CONFIG["/confit/checkpoint"].commit()


@stub.function(
    image=image,
    gpu=GPU_CONFIG,
    # volumes=VOLUME_CONFIG,
    timeout=3600 * 24,
    _allow_background_volume_commits=True,
)
def train(config_raw: str, dataset_name: str, sample_seed: int, model_seed: int):
    # write the config and data to the disk
    with open("/confit/config.yaml", "w") as f:
        f.write(config_raw)
    TRAIN_CMD = (
        f"accelerate launch confit/train.py "
        f"--config config.yaml "
        f"--dataset {dataset_name} "
        f"--sample_seed {sample_seed} "
        f"--model_seed {model_seed}")

    run_cmd(TRAIN_CMD, '/confit')


@stub.local_entrypoint()
def main(
    config: str,
):
    # Read config and data source files and pass their contents to the remote function.
    with open(config, "r") as cfg:
        train.remote(cfg.read(), "GB1_Olson2014_ddg", 0, 1)
