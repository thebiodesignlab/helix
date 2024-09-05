from typing import Tuple, Optional, Dict, List
import io
import os
import shutil
import subprocess
import tempfile
import logging
import zipfile
from modal import Image, gpu, Volume
from helix.utils.structure import fetch_pdb_structure
from helix.core import app

TEMP_DIR = "/mnt/tmp/diffdock"
PROJECT_URL = "https://github.com/gcorso/DiffDock"

DEFAULT_CONFIG = {
    "actual_steps": 19,
    "ckpt": "best_ema_inference_epoch_model.pt",
    "confidence_ckpt": "best_model_epoch75.pt",
    "confidence_model_dir": "/home/appuser/.cache/torch/diffdock/v1.1/confidence_model",
    "different_schedules": False,
    "inf_sched_alpha": 1,
    "inf_sched_beta": 1,
    "inference_steps": 20,
    "initial_noise_std_proportion": 1.4601642460337794,
    "limit_failures": 5,
    "model_dir": "/home/appuser/.cache/torch/diffdock/v1.1/score_model",
    "no_final_step_noise": True,
    "no_model": False,
    "no_random": False,
    "no_random_pocket": False,
    "ode": False,
    "old_filtering_model": True,
    "old_score_model": False,
    "resample_rdkit": False,
    "samples_per_complex": 10,
    "sigma_schedule": "expbeta",
    "temp_psi_rot": 0.9022615585677628,
    "temp_psi_tor": 0.5946212391366862,
    "temp_psi_tr": 0.727287304570729,
    "temp_sampling_rot": 2.06391612594481,
    "temp_sampling_tor": 7.044261621607846,
    "temp_sampling_tr": 1.170050527854316,
    "temp_sigma_data_rot": 0.7464326999906034,
    "temp_sigma_data_tor": 0.6943254174849822,
    "temp_sigma_data_tr": 0.9299802531572672,
}

cache_volume = Volume.from_name("diffdock-cache", create_if_missing=True)

image = Image.from_registry("rbgcsail/diffdock", add_python="3.9").run_commands(
    "micromamba run -n diffdock python /home/appuser/DiffDock/utils/precompute_series.py"
).pip_install("pyyaml", "biopython")


def kwargs_to_cli_args(**kwargs) -> List[str]:
    """
    Converts keyword arguments to a CLI argument string.
    Boolean kwargs are added as flags if True, and omitted if False.
    """
    cli_args = []
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cli_args.append(f"--{key}")
        else:
            if value is not None and str(value) != "":
                cli_args.append(f"--{key}={value}")
    return cli_args


def parse_ligand_filename(filename: str) -> Dict:
    """
    Parses an sdf filename to extract information.
    """
    if not filename.endswith(".sdf"):
        return {}

    basename = os.path.basename(filename).replace(".sdf", "")
    tokens = basename.split("_")
    rank = tokens[0]
    rank = int(rank.replace("rank", ""))
    if len(tokens) == 1:
        return {"filename": basename, "rank": rank, "confidence": None}

    con_str = tokens[1]
    conf_val = float(con_str.replace("confidence", ""))

    return {"filename": basename, "rank": rank, "confidence": conf_val}


def process_zip_file(zip_content: bytes):
    pdb_file = []
    sdf_files = []
    with zipfile.ZipFile(io.BytesIO(zip_content)) as my_zip_file:
        for filename in my_zip_file.namelist():
            if filename.endswith("/"):
                continue

            if filename.endswith(".pdb"):
                content = my_zip_file.read(filename).decode("utf-8")
                pdb_file.append({"path": filename, "content": content})

            if filename.endswith(".sdf"):
                info = parse_ligand_filename(filename)
                info["content"] = my_zip_file.read(filename).decode("utf-8")
                info["path"] = filename
                sdf_files.append(info)

    sdf_files = sorted(sdf_files, key=lambda x: x.get("rank", 1_000))

    return pdb_file, sdf_files


def run_diffdock_command(
    protein_path: str,
    ligand: str,
    config: Optional[Dict] = None,
):
    import yaml

    all_arg_dict = {
        "protein_path": protein_path,
        "ligand": ligand,
        "loglevel": "DEBUG",
    }

    # Merge default config with provided config
    if config is None:
        config = {}
    merged_config = {**DEFAULT_CONFIG, **config}

    # Create a temporary YAML file for the merged config
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode='w') as temp_config_file:
        yaml.dump(merged_config, temp_config_file)
        temp_config_path = temp_config_file.name

    # Update the argument dictionary to include the path to the config file
    all_arg_dict["config"] = temp_config_path
    # Check device availability
    result = subprocess.run(
        ["python", "utils/print_device.py"],
        check=False,
        text=True,
        capture_output=True,
        env=os.environ,
    )
    logging.debug(f"Device check output:\n{result.stdout}")

    command = ["micromamba", "run", "-n", "diffdock",
               "python", "/home/appuser/DiffDock/inference.py"]
    command += kwargs_to_cli_args(**all_arg_dict)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = temp_dir
        command.append(f"--out_dir={temp_dir_path}")

        # Convert command list to string for printing
        command_str = " ".join(command)
        logging.info(f"Executing command: {command_str}")

        # Running the command
        try:
            result = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
            )
            logging.debug(f"Command output:\n{result.stdout}")
            full_output = f"Standard out:\n{result.stdout}"
            if result.stderr:
                # Skip progress bar lines
                stderr_lines = result.stderr.split("\n")
                stderr_lines = filter(
                    lambda x: "%|" not in x, stderr_lines)
                stderr_text = "\n".join(stderr_lines)
                logging.error(f"Command error:\n{stderr_text}")
                full_output += f"\nStandard error:\n{stderr_text}"

            with open(f"{temp_dir_path}/output.log", "w") as log_file:
                log_file.write(full_output)

        except subprocess.CalledProcessError as e:
            logging.error(
                f"An error occurred while executing the command: {e}")

        # Copy the input protein into the output directory
        sub_dirs = [os.path.join(temp_dir_path, x)
                    for x in os.listdir(temp_dir_path)]
        sub_dirs = list(filter(lambda x: os.path.isdir(x), sub_dirs))
        logging.debug(f"Output Subdirectories: {sub_dirs}")
        if len(sub_dirs) == 1:
            sub_dir = sub_dirs[0]
            # Copy the input protein from the input to the output
            trg_protein_path = os.path.join(
                sub_dir, os.path.basename(protein_path))
            shutil.copy(protein_path, trg_protein_path)

        # Zip the output directory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(temp_dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, os.path.relpath(
                        file_path, temp_dir_path))

        logging.debug(f"Directory '{temp_dir}' zipped successfully")

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


@app.function(gpu=gpu.A100(size='40GB'), image=image, timeout=5000, volumes={"/home/appuser/.cache/torch": cache_volume}, _allow_background_volume_commits=True)
def predict_docking(protein_pdb_id: Optional[str] = None, protein_file_content: Optional[str] = None, ligand_smile: Optional[str] = None, ligand_file_content: Optional[str] = None, config: Optional[Dict] = None) -> Tuple[str, Optional[str], Optional[Dict]]:
    """
    Run DiffDock with the provided inputs and configuration.

    Args:
        protein_pdb_id (Optional[str]): PDB ID of the protein.
        protein_file_content (Optional[str]): Content of the protein file.
        ligand_smile (Optional[str]): SMILES string of the ligand.
        ligand_file_content (Optional[str]): Content of the ligand file.
        *args: Additional arguments for DiffDock.
        config (Optional[Dict]): Additional configuration parameters.

    Returns:
        Tuple[str, Optional[str], Optional[Dict]]: Message, output file path, and view selector content.
    """

    if protein_pdb_id and not protein_file_content:
        protein_file_content = fetch_pdb_structure(protein_pdb_id)

    if not protein_file_content:
        return "Protein file is missing! Must provide a protein file in PDB format", None, None
    if not ligand_file_content and not ligand_smile:
        return "Ligand is missing! Must provide a ligand file in SDF format or SMILE string", None, None

    ligand_desc = ligand_file_content if ligand_file_content else ligand_smile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_protein_file:
        temp_protein_file.write(protein_file_content.encode())
        temp_protein_file_path = temp_protein_file.name

    output_file = run_diffdock_command(
        temp_protein_file_path, ligand_desc, config=config)
    os.remove(temp_protein_file_path)

    outputs = []
    if output_file:
        pdb_files, sdf_files = process_zip_file(output_file)
        pdb_file = pdb_files[0] if pdb_files else None
        for sdf_file in sdf_files:
            confidence = sdf_file.get("confidence", None)
            if confidence is None:
                continue
            label = f"Rank {sdf_file['rank']}. Confidence {confidence:.2f}"
            pdb_text = pdb_file['content'] if pdb_file else None
            sdf_text = sdf_file['content']
            output_viz = "Output visualisation unavailable"
            outputs.append({
                "label": label,
                "pdb_text": pdb_text,
                "sdf_text": sdf_text,
                "output_viz": output_viz,
            })

    return outputs
