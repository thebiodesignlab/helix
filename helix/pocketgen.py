from modal import Image
from helix.core import app

pocketgen_image = (
    Image.micromamba()
    .apt_install("git", "g++", "make", "wget", "cmake")
    .micromamba_install(
        "pytorch=2.2.0", "pytorch-cuda=11.8", "pyg", "rdkit", "openbabel", "tensorboard",
        "pyyaml", "easydict", "python-lmdb", "openmm", "pdbfixer", "flask",
        "numpy", "swig", "boost-cpp", "sphinx", "sphinx_rtd_theme", "openmm=8.0.0", "pdbfixer=1.9",
        channels=["pytorch", "nvidia", "pyg", "conda-forge"]
    ).pip_install("gdown")
    .run_commands(
        "python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3",
        "git clone https://github.com/zaixizhang/PocketGen.git",
        "mkdir -p /PocketGen/checkpoints",
        "gdown 1cuvdiu3bXyni71A2hoeZSWT1NOsNfeD_ -O /PocketGen/checkpoints/checkpoint.pt"
    )
    .workdir("/PocketGen")
    .run_commands("pip install pyg_lib torch-scatter==2.1.2 torch-sparse==0.6.18 torch-spline-conv==1.2.2 torch-geometric==2.3.1 torch-cluster==1.6.3 -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html")
    .pip_install("numpy==1.23.5", "tqdm==4.65.0", "meeko==0.1.dev3", "wandb", "scipy", "pdb2pqr", "vina==1.2.2", "fair-esm==2.0.0", "omegaconf==2.3.0", "biopython==1.79")
)


@app.function(image=pocketgen_image, gpu='any', timeout=3600)
def run():
    import subprocess

    def generate_new():
        try:
            result = subprocess.run(
                ['python', 'generate_new.py'], check=True, capture_output=True, text=True)
            print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("An error occurred while running generate_new.py:", e.stderr)

    generate_new()


@app.local_entrypoint()
def main():
    run.remote()
