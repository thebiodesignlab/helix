import multiprocessing as mp
import os
import random
from functools import partial

import esm
import numpy as np
from helix.analysis.structure.utils import fetch_pdb_structure
import torch
from modal import Image
from models.PD import Pocket_Design_new
from rdkit import Chem

from tqdm import tqdm


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


def convert_pdbqt_to_sdf(pdbqt_file, sdf_file):
    from openbabel import pybel
    mol = next(pybel.readfile("pdbqt", pdbqt_file))
    mol.removeh()
    mol.write("sdf", sdf_file, overwrite=True)


def calculate_vina(id, pro_path, lig_path, output=False):
    from utils.evaluation.docking_vina import PrepLig, PrepProt
    from vina import Vina
    size_factor = 1.2
    buffer = 8.0
    if id is not None:
        pro_path = os.path.join(pro_path, f"{id}.pdb")
        lig_path = os.path.join(lig_path, f"{id}.sdf")
    mol = Chem.MolFromMolFile(lig_path, sanitize=True)
    pos = mol.GetConformer(0).GetPositions()
    center = np.mean(pos, 0)
    os.makedirs('./tmp', exist_ok=True)
    ligand_pdbqt = f'./tmp/{id}lig.pdbqt'
    protein_pqr = f'./tmp/{id}pro.pqr'
    protein_pdbqt = f'./tmp/{id}pro.pdbqt'
    lig = PrepLig(lig_path, 'sdf')
    lig.addH()
    lig.get_pdbqt(ligand_pdbqt)
    prot = PrepProt(pro_path)
    prot.addH(protein_pqr)
    prot.get_pdbqt(protein_pdbqt)
    v = Vina(sf_name='vina', seed=0, verbosity=0)
    v.set_receptor(protein_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    x, y, z = (pos.max(0) - pos.min(0)) * size_factor + buffer
    v.compute_vina_maps(center=center, box_size=[x, y, z])
    energy = v.score()
    print(f'Score before minimization: {energy[0]:.3f} (kcal/mol)')
    energy_minimized = v.optimize()
    print(f'Score after minimization : {energy_minimized[0]:.3f} (kcal/mol)')
    v.dock(exhaustiveness=64, n_poses=30)
    score = v.energies(n_poses=1)[0][0]
    print(f'Score after docking : {score:.3f} (kcal/mol)')
    if output:
        v.write_poses(pro_path[:-4] + '_docked.pdbqt',
                      n_poses=1, overwrite=True)
        convert_pdbqt_to_sdf(
            pro_path[:-4] + '_docked.pdbqt', pro_path[:-4] + '_docked.sdf')
    return score


def vina_mp(pro_path, lig_path, number_list):
    pool = mp.Pool(16)
    vina_list = []
    func = partial(calculate_vina, pro_path=pro_path, lig_path=lig_path)
    for vina_score in tqdm(pool.imap_unordered(func, number_list), total=len(number_list)):
        if vina_score is not None:
            vina_list.append(vina_score)
    pool.close()
    print('Vina: ', np.average(vina_list))
    return vina_list


def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, residue_dict=None, seq=None, full_seq_idx=None, r10_idx=None):
    instance = {}
    if protein_dict is not None:
        for key, item in protein_dict.items():
            instance['protein_' + key] = item
    if ligand_dict is not None:
        for key, item in ligand_dict.items():
            instance['ligand_' + key] = item
    if residue_dict is not None:
        for key, item in residue_dict.items():
            instance[key] = item
    if seq is not None:
        instance['seq'] = seq
    if full_seq_idx is not None:
        instance['full_seq_idx'] = full_seq_idx
    if r10_idx is not None:
        instance['r10_idx'] = r10_idx
    return instance


def name2data(pdb_content, ligand_content, transform):
    from utils.protein_ligand import PDBProtein, parse_sdf_block
    from utils.data import torchify_dict

    protein = PDBProtein(pdb_content)
    seq = ''.join(protein.to_dict_residue()['seq'])
    ligand = parse_sdf_block(ligand_content, feat=False)
    r10_idx, r10_residues = protein.query_residues_ligand(
        ligand, radius=10, selected_residue=None, return_mask=False)
    full_seq_idx, _ = protein.query_residues_ligand(
        ligand, radius=3.5, selected_residue=r10_residues, return_mask=False)
    assert len(r10_idx) == len(r10_residues)

    pocket_dict = protein.residues_to_dict_atom(r10_residues)
    residue_dict = protein.to_dict_residue()

    _, residue_dict['protein_edit_residue'] = protein.query_residues_ligand(
        ligand)
    assert residue_dict['protein_edit_residue'].sum(
    ) > 0 and residue_dict['protein_edit_residue'].sum() == len(full_seq_idx)
    assert len(residue_dict['protein_edit_residue']) == len(r10_idx)
    full_seq_idx.sort()
    r10_idx.sort()

    data = from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand),
        residue_dict=torchify_dict(residue_dict),
        seq=seq,
        full_seq_idx=torch.tensor(full_seq_idx),
        r10_idx=torch.tensor(r10_idx)
    )
    return transform(data)


def generate_pockets_and_dock(protein_pdb_id, pdb_content, ligand_sdf_content, ligand_smile, config_path='./configs/train_model.yml', device='cuda:0'):
    from utils.misc import load_config, seed_all
    from utils.data import collate_mols_block
    from torch.utils.data import DataLoader
    from torch_geometric.transforms import Compose
    from utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom
    config = load_config(config_path)
    seed_all(2023)

    if protein_pdb_id is not None:
        pdb_content = fetch_pdb_structure(protein_pdb_id)

    if

    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
    ])

    name = 'esm2_t33_650M_UR50D'
    pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(
        name)
    batch_converter = alphabet.get_batch_converter()
    ckpt = torch.load(config.model.checkpoint, map_location=device)
    del pretrained_model

    model = Pocket_Design_new(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=device
    ).to(device)
    model.load_state_dict(ckpt['model'])

    print('Loading dataset...')
    data = name2data(pdb_content, ligand_sdf_content, transform)
    datalist = [data for _ in range(8)]

    model.generate_id = 0
    model.generate_id1 = 0
    test_loader = DataLoader(datalist, batch_size=4, shuffle=False, num_workers=config.train.num_workers,
                             collate_fn=partial(collate_mols_block, batch_converter=batch_converter))
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader, desc='Test'):
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            _, _ = model.generate(batch, './tmp')

    score_list = vina_mp('./tmp', './tmp', np.arange(len(datalist)))

    return score_list

# Example usage:
# scores = generate_pockets_and_dock(protein_pdb_id='2p16', pdb_content=None, ligand_sdf_content=None, ligand_smile='CCO', config_path='./configs/train_model.yml')
