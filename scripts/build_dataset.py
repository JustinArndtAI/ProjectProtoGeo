import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem

def get_pdbbind_index(index_file_path):
    """
    Parses the PDBbind index file, which has a tricky format, to extract PDB codes,
    resolution, release year, and binding affinity data (-logKd/Ki). This function is
    robust against extra spaces in ligand names or references.

    Args:
        index_file_path (Path): Path to the PDBbind index file.

    Returns:
        pd.DataFrame: A DataFrame with PDB codes and their associated data.
    """
    records = []
    with open(index_file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue

            # Manually parse the line due to its irregular format.
            try:
                ligand_start = line.rindex('(')
                ligand_end = line.rindex(')')
                ligand_name = line[ligand_start+1:ligand_end]
            except ValueError:
                continue

            main_part = line[:ligand_start].strip()
            parts = main_part.split()

            if len(parts) < 5:
                continue

            pdb_code = parts[0]
            resolution = parts[1]
            release_year = parts[2]
            neg_log_affinity = parts[3]
            affinity_value = parts[4]
            affinity_type = affinity_value.split('=')[0]

            records.append({
                "pdb_code": pdb_code,
                "resolution": resolution,
                "release_year": release_year,
                "neg_log_affinity": neg_log_affinity,
                "affinity_value": affinity_value,
                "affinity_type": affinity_type,
                "ligand_name": ligand_name
            })

    df = pd.DataFrame(records)
    # Convert columns to their appropriate types for later use
    df['resolution'] = pd.to_numeric(df['resolution'], errors='coerce')
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df['neg_log_affinity'] = pd.to_numeric(df['neg_log_affinity'], errors='coerce')
    df.dropna(inplace=True) # Drop any rows where conversion failed
    return df


def validate_ligand(sdf_file_path):
    """
    Validates a ligand SDF file using RDKit.

    Checks:
    1. The file can be parsed by RDKit.
    2. The ligand contains only allowed organic atoms.

    Args:
        sdf_file_path (Path): The path to the ligand's .sdf file.

    Returns:
        bool: True if the ligand is valid, False otherwise.
    """
    # Allowed elements for typical organic drug-like molecules.
    allowed_elements = {'C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I'}
    
    try:
        suppl = Chem.SDMolSupplier(str(sdf_file_path), sanitize=True)
        mol = suppl[0]
        if mol is None:
            return False # RDKit failed to parse the molecule
        
        # Check if all atoms in the molecule are from the allowed set
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in allowed_elements:
                return False # Contains a disallowed element (e.g., a metal)
        
        return True
    except Exception:
        # Catch any other RDKit parsing errors
        return False

def main(pdbbind_dir, output_file):
    """
    Main function to process PDBbind data, validate complexes, and create a
    clean dataset index file.
    """
    pdbbind_path = Path(pdbbind_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load the primary index file
    index_file = pdbbind_path / "index" / "INDEX_general_PL_data.2020"
    if not index_file.exists():
        print(f"Error: Index file not found at {index_file}")
        print("Please ensure you have extracted the PDBbind data correctly into the `data_raw` directory.")
        return

    print("Loading PDBbind index...")
    index_df = get_pdbbind_index(index_file)
    print(f"Found {len(index_df)} total entries in the index.")

    valid_complexes = []
    
    # 2. Iterate through each entry and validate
    print("Validating complexes...")
    for _, row in tqdm(index_df.iterrows(), total=index_df.shape[0]):
        pdb_code = row["pdb_code"]
        
        protein_path = pdbbind_path / "v2020" / pdb_code / f"{pdb_code}_protein.pdb"
        ligand_path = pdbbind_path / "v2020" / pdb_code / f"{pdb_code}_ligand.sdf"

        # Validation Check 1: Ensure all necessary files exist
        if not (protein_path.exists() and ligand_path.exists()):
            continue
            
        # Validation Check 2: Use RDKit to validate the ligand file
        if not validate_ligand(ligand_path):
            continue

        # If all checks pass, add it to our list of clean data
        valid_complexes.append({
            "pdb_code": pdb_code,
            "neg_log_affinity": row["neg_log_affinity"],
            "affinity_type": row["affinity_type"],
            "protein_path": str(protein_path.relative_to(pdbbind_path.parent)),
            "ligand_path": str(ligand_path.relative_to(pdbbind_path.parent)),
        })

    # 3. Create and save the final dataset
    clean_df = pd.DataFrame(valid_complexes)
    print(f"\nValidation complete. Found {len(clean_df)} valid complexes.")
    
    clean_df.to_csv(output_path, index=False)
    print(f"Clean dataset index saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the PDBbind-ProtoGeo dataset.")
    parser.add_argument(
        "--pdbbind_dir",
        type=str,
        default="data_raw/PDBbind_v2020_general_set",
        help="Path to the extracted PDBbind v2020 general set directory."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data_processed/protogeo_dataset_index.csv",
        help="Path to save the final cleaned dataset index file."
    )
    args = parser.parse_args()
    main(args.pdbbind_dir, args.output_file)