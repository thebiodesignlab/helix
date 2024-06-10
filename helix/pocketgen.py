from helix.analysis.structure.utils import fetch_pdb_structure
import shutil
from typing import Optional
from modal import Image
from helix.core import app
from helix.utils import smiles_to_sdf

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
def predict_docking(protein_pdb_id: Optional[str] = None, pdb_content: Optional[str] = None, ligand_sdf_content: Optional[str] = None, ligand_smile: Optional[str] = None):
    import subprocess
    import tempfile
    import os
    import uuid

    if not (protein_pdb_id or pdb_content):
        raise ValueError(
            "Either protein_pdb_id or pdb_content must be provided.")
    if not (ligand_sdf_content or ligand_smile):
        raise ValueError(
            "Either ligand_sdf_content or ligand_smile must be provided.")

    temp_dir = tempfile.mkdtemp()

    try:
        # Generate a random ID
        random_id = str(uuid.uuid4())

        if protein_pdb_id:
            # Fetch PDB content using the provided PDB ID
            pdb_content = fetch_pdb_structure(protein_pdb_id)

        pdb_file_path = os.path.join(temp_dir, f"{random_id}.pdb")
        with open(pdb_file_path, 'w') as pdb_file:
            pdb_file.write(pdb_content)

        ligand_file_path = os.path.join(temp_dir, f"{random_id}_ligand.sdf")
        if ligand_sdf_content:
            with open(ligand_file_path, 'w') as ligand_file:
                ligand_file.write(ligand_sdf_content)
        else:
            try:
                ligand_sdf_content = smiles_to_sdf(ligand_smile)
                with open(ligand_file_path, 'w') as ligand_file:
                    ligand_file.write(ligand_sdf_content)
            except ValueError as e:
                raise ValueError(f"Failed to convert SMILES to SDF: {e}")

        def generate_new():
            try:
                result = subprocess.run(
                    ['python', 'generate_new.py', '--target', temp_dir], check=True, capture_output=True, text=True)
                print("Output:", result.stdout)
            except subprocess.CalledProcessError as e:
                print("An error occurred while running generate_new.py:", e.stderr)

        generate_new()

        pocket_file_path = os.path.join(temp_dir, f"{random_id}_pocket.pdb")
        if not os.path.exists(pocket_file_path):
            raise FileNotFoundError(
                f"Expected output file {pocket_file_path} not found.")

        with open(pocket_file_path, 'r') as pocket_file:
            pocket_content = pocket_file.read()

        return pocket_content

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
