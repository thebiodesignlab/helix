from pathlib import PurePosixPath, Path

from modal import Volume, Image, Stub, CloudBucketMount, Secret


APP_NAME = "finetuning"

stub = Stub(name='finetuning')


def download_models():
    from transformers import EsmModel, AutoTokenizer
    EsmModel.from_pretrained(
        "facebook/esm1b_t33_650M_UR50S")
    AutoTokenizer.from_pretrained(
        "facebook/esm1b_t33_650M_UR50S")


image = Image.from_registry("pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel").apt_install([
    'wget', 'git', 'unzip']).pip_install(
    "accelerate==0.23.0",
    "appdirs==1.4.4",
    "loralib",
    "bitsandbytes==0.41.1",
    "datasets==2.12.0",
    "fire==0.5.0",
    "peft==0.5.0",
    "transformers==4.34.0",
    "gradio",
    "scipy",
    "numpy",
    "pandas",
    "pyyaml",
    "biopython"
).run_function(download_models).run_commands("git clone https://github.com/luo-group/ConFit.git /confit").run_commands("wget https://pub-cd7cf2814e25430c9be03005fcab4e39.r2.dev/mutation_data.zip -O mutation_data.zip && unzip mutation_data.zip -d /confit && rm mutation_data.zip")


# Volumes for pre-trained models and training runs.
results_volume = Volume.from_name(
    "confit-results-vol", create_if_missing=True)
checkpoint_volume = Volume.from_name(
    "confit-checkpoint-vol", create_if_missing=True)
data_volume = Volume.from_name("example-data-vol", create_if_missing=True)

eigencell_bucket = CloudBucketMount(
    "eigencell-finetuning", secret=Secret.from_name("cloudflare-r2-secrets"))

MOUNT_PATH: Path = Path("/mnt/bucket")
CHECKPOINT_PATH: Path = MOUNT_PATH / "checkpoints"
RESULTS_PATH: Path = MOUNT_PATH / "results"

VOLUME_CONFIG: dict[str | PurePosixPath, Volume] = {
    MOUNT_PATH: eigencell_bucket,
}
