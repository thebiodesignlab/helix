def smiles_to_sdf(smiles: str) -> str:
    """
    Converts a SMILES string to an SDF format string.

    Parameters:
    - smiles: A string containing the SMILES representation of the molecule.

    Returns:
    - A string containing the SDF representation of the molecule.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    # Convert to SDF
    sdf_str = Chem.MolToMolBlock(mol)
    return sdf_str
