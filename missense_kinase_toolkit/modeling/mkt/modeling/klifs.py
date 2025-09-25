import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
import prolif as plf
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import MDAnalysis as mda
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


DICT_KLIFS_PROPS = {
    1: {
        "name": "Apolar contact",
        "prolif_type": "Hydrophobic",
    },
    2: {
        "name": "Aromatic face-to-face",
        "prolif_type": "PiStacking",
    },
    3: {
        "name": "Aromatic edge-to-face",
        "prolif_type": "EdgeToFace",  # Note: ProLIF does not have a direct EdgeToFace type
    },
    4: {
        "name": "Hydrogen bond donor (protein)",
        "prolif_type": "HBDonor",
    },
    5: {
        "name": "Hydrogen bond acceptor (protein)",
        "prolif_type": "HBAcceptor",
    },
    6: {
        "name": "Protein cation - ligand anion",
        "prolif_type": "Cationic",
    },
    7: {
        "name": "Protein anion - ligand cation",
        "prolif_type": "Anionic",
    },
}
"""Dict[int, Dict[str, str]]: KLIFS standard interaction properties with ProLIF types."""

LIST_INTERACTIONS_KLIFS = [i["prolif_type"] for i in DICT_KLIFS_PROPS.values()]
"""List[str]: List of KLIFS standard interaction types used in ProLIF."""

LIST_INTERACTIONS_AVAIL = [i for i in LIST_INTERACTIONS_KLIFS if i != "EdgeToFace"]
"""List[str]: List of available ProLIF interactions that can be used in KLIFS IFP generation."""

SET_STANDARD_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "HOH", # Water molecules
    "WAT", # Water molecules
}
"""set[str]: Standard amino acid residues and water molecules in MDAnalysis."""

@dataclass
class KLIFSIFPGenerator:
    """Generate KLIFS interaction fingerprints using ProLIF."""

    path_cif: str
    """Path to the CIF file containing the kinase-ligand complex."""
    u: mda.Universe = field(init=False, repr=False)
    """MDAnalysis Universe object for the loaded structure."""
    
    def __post_init__(self):
        # check that cif file exists
        if not Path(self.path_cif).is_file():
            raise FileNotFoundError(f"CIF file not found: {self.path_cif}")

        self.u = mda.Universe(self.path_cif)
        self.fp = plf.Fingerprint(LIST_INTERACTIONS_AVAIL)

        print("Initialized KLIFS IFP generator with 7 standard interaction types:")
        for k, v in DICT_KLIFS_PROPS.items():
            print(f"  {k}: {v['name']}")
        
    def load_structure_from_cif(self) -> Tuple[mda.Universe, str]:
        """Load structure from CIF file and identify ligand.
        
        Returns:
        --------
        ligand_resname: str
            The residue name of the identified ligand.
        """
        # find unique residue names
        all_resnames = set(self.u.atoms.resnames)
        ligand_candidates = all_resnames - SET_STANDARD_RESIDUES
        
        if not ligand_candidates:
            raise ValueError(f"No ligand found in {self.path_cif}")
        
        # Take the first non-standard residue as ligand
        ligand_resname = list(ligand_candidates)[0]
        print(f"Identified ligand: {ligand_resname}")
        
        return ligand_resname
    
    def extract_klifs_pocket(
        self, 
        universe: mda.Universe, 
        klifs_residues: List[Tuple[int, str]]
    ) -> mda.AtomGroup:
        """
        Extract KLIFS pocket residues from the structure.
        
        Args:
            universe: MDAnalysis Universe
            klifs_residues: List of (residue_number, chain_id) tuples for KLIFS pocket
            
        Returns:
            AtomGroup containing KLIFS pocket atoms
        """
        pocket_atoms = []
        
        for resnum, chain in klifs_residues:
            try:
                # Select residue by number and chain
                selection = f"resid {resnum} and segid {chain}"
                atoms = universe.select_atoms(selection)
                
                if len(atoms) == 0:
                    # Try alternative selection without chain if segid doesn't work
                    selection = f"resid {resnum}"
                    atoms = universe.select_atoms(selection)
                
                if len(atoms) > 0:
                    pocket_atoms.extend(atoms)
                else:
                    print(f"Warning: Could not find residue {resnum}:{chain}")
                    
            except Exception as e:
                print(f"Error selecting residue {resnum}:{chain}: {e}")
                continue
        
        if not pocket_atoms:
            raise ValueError("No KLIFS pocket residues found")
            
        return mda.AtomGroup(pocket_atoms)
    
    def generate_ifp(self, cif_path: str, 
                    klifs_residues: List[Tuple[int, str]],
                    output_prefix: Optional[str] = None) -> np.ndarray:
        """
        Generate interaction fingerprint for a single complex.
        
        Args:
            cif_path: Path to CIF file
            klifs_residues: List of KLIFS pocket residues as (resnum, chain) tuples
            output_prefix: Optional prefix for output files
            
        Returns:
            Binary interaction fingerprint as numpy array
        """
        # Load structure
        u, ligand_resname = self.load_structure_from_cif(cif_path)
        
        # Extract pocket and ligand
        pocket = self.extract_klifs_pocket(u, klifs_residues)
        ligand = u.select_atoms(f"resname {ligand_resname}")
        
        print(f"Pocket atoms: {len(pocket)}")
        print(f"Ligand atoms: {len(ligand)}")
        
        # Generate fingerprint
        # ProLIF expects protein and ligand as separate AtomGroups
        ifp = self.fp.run_from_iterable([u], 
                                       protein_sel=pocket,
                                       ligand_sel=ligand)
        
        # Convert to binary vector
        # ProLIF returns a pandas DataFrame with MultiIndex
        ifp_df = ifp.to_dataframe()
        
        # Create binary vector for KLIFS residues in order
        binary_vector = self._create_klifs_binary_vector(ifp_df, klifs_residues)
        
        # Save results if requested
        if output_prefix:
            self._save_results(ifp_df, binary_vector, output_prefix)
        
        return binary_vector
    
    def _create_klifs_binary_vector(self, ifp_df: pd.DataFrame, 
                                   klifs_residues: List[Tuple[int, str]]) -> np.ndarray:
        """
        Create ordered binary vector for KLIFS residues with standard 7 properties.
        
        Args:
            ifp_df: ProLIF interaction dataframe
            klifs_residues: Ordered list of 85 KLIFS residues
            
        Returns:
            Binary vector with shape (85 * 7,) = 595 bits total
        """
        binary_vector = np.zeros(85 * 7, dtype=int)
        
        # Process each KLIFS residue in order (1-85)
        for klifs_idx, (resnum, chain) in enumerate(klifs_residues):
            base_idx = klifs_idx * 7  # 7 properties per residue
            
            # Map ProLIF interactions to KLIFS properties
            property_mapping = {
                'Hydrophobic': 0,        # Position 1: Apolar contact
                'PiStacking': 1,         # Position 2: Aromatic face-to-face
                # Position 3: Edge-to-face (special handling needed)
                'HBDonor': 3,           # Position 4: H-bond donor (protein)
                'HBAcceptor': 4,        # Position 5: H-bond acceptor (protein) 
                'Cationic': 5,          # Position 6: Protein cation - ligand anion
                'Anionic': 6            # Position 7: Protein anion - ligand cation
            }
            
            # Look for this residue in the IFP dataframe
            residue_found = False
            
            # Try different residue identifier formats
            possible_keys = [
                f"{chain}{resnum}",
                f"{resnum}{chain}",
                f"{chain}:{resnum}",
                f"{resnum}",
                str(resnum)
            ]
            
            for residue_key in possible_keys:
                if residue_key in ifp_df.columns.get_level_values(0):
                    residue_found = True
                    
                    # Check each interaction type
                    for prolif_interaction, klifs_pos in property_mapping.items():
                        try:
                            if prolif_interaction in ifp_df.columns.get_level_values(1):
                                value = ifp_df.loc[0, (residue_key, prolif_interaction)]
                                if pd.notna(value) and value > 0:
                                    binary_vector[base_idx + klifs_pos] = 1
                        except (KeyError, IndexError):
                            continue
                    
                    # Special handling for edge-to-face aromatic interactions
                    # This might require custom geometric analysis
                    edge_to_face_bit = self._detect_edge_to_face_interaction(
                        ifp_df, residue_key, klifs_idx
                    )
                    binary_vector[base_idx + 2] = edge_to_face_bit  # Position 3
                    
                    break
            
            if not residue_found:
                print(f"Warning: KLIFS residue {klifs_idx+1} ({resnum}:{chain}) not found in IFP")
        
        return binary_vector
    
    def _detect_edge_to_face_interaction(self, ifp_df: pd.DataFrame, 
                                       residue_key: str, klifs_idx: int) -> int:
        """
        Detect aromatic edge-to-face interactions (KLIFS property 3).
        
        This is a placeholder - you may need to implement custom geometric
        analysis or use additional ProLIF features for edge-to-face detection.
        
        Args:
            ifp_df: ProLIF interaction dataframe
            residue_key: Residue identifier
            klifs_idx: KLIFS residue index (0-84)
            
        Returns:
            1 if edge-to-face interaction detected, 0 otherwise
        """
        # Placeholder implementation
        # In practice, you might need to:
        # 1. Check if PiStacking exists with specific geometric criteria
        # 2. Analyze ring orientations from the original structure
        # 3. Use additional ProLIF features or custom geometric analysis
        
        try:
            # Simple heuristic: assume some PiStacking interactions are edge-to-face
            if 'PiStacking' in ifp_df.columns.get_level_values(1):
                if residue_key in ifp_df.columns.get_level_values(0):
                    pi_value = ifp_df.loc[0, (residue_key, 'PiStacking')]
                    # This is a simplified approach - you may want more sophisticated detection
                    return int(pd.notna(pi_value) and pi_value > 0)
        except (KeyError, IndexError):
            pass
        
        return 0
    
    def _save_results(self, ifp_df: pd.DataFrame, binary_vector: np.ndarray, 
                     output_prefix: str):
        """Save IFP results to files."""
        # Save detailed IFP dataframe
        ifp_df.to_csv(f"{output_prefix}_detailed_ifp.csv")
        
        # Save binary vector with KLIFS property labels
        klifs_labels = []
        for i in range(85):
            for prop in self.klifs_properties:
                klifs_labels.append(f"Res{i+1:02d}_Prop{prop['position']}_{prop['name'].replace(' ', '_')}")
        
        # Save as CSV with labels
        binary_df = pd.DataFrame([binary_vector], columns=klifs_labels)
        binary_df.to_csv(f"{output_prefix}_klifs_binary.csv", index=False)
        
        # Also save as simple text file
        np.savetxt(f"{output_prefix}_klifs_binary.txt", binary_vector, fmt='%d')
        
    def batch_process(self, cif_files: List[str], 
                     klifs_residues_list: List[List[Tuple[int, str]]],
                     output_dir: str = "ifp_results") -> pd.DataFrame:
        """
        Process multiple CIF files in batch.
        
        Args:
            cif_files: List of CIF file paths
            klifs_residues_list: List of KLIFS residue lists for each structure
            output_dir: Directory to save results
            
        Returns:
            DataFrame with all binary fingerprints
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        results = []
        
        for i, (cif_file, klifs_residues) in enumerate(zip(cif_files, klifs_residues_list)):
            try:
                print(f"\nProcessing {cif_file} ({i+1}/{len(cif_files)})")
                
                # Generate IFP
                binary_vector = self.generate_ifp(
                    cif_file, 
                    klifs_residues,
                    output_prefix=f"{output_dir}/complex_{i:03d}"
                )
                
                # Store result
                result_dict = {
                    'complex_id': Path(cif_file).stem,
                    'cif_file': cif_file
                }
                
                # Add binary bits as columns with KLIFS labels
                for j, bit in enumerate(binary_vector):
                    res_idx = j // 7 + 1  # Residue number (1-85)
                    prop_idx = j % 7 + 1   # Property number (1-7)
                    result_dict[f'R{res_idx:02d}_P{prop_idx}'] = bit
                
                results.append(result_dict)
                
            except Exception as e:
                print(f"Error processing {cif_file}: {e}")
                continue
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{output_dir}/all_klifs_ifps.csv", index=False)
        
        return results_df

# Example usage and helper functions
def load_klifs_annotation(annotation_file: str) -> List[Tuple[int, str]]:
    """
    Load KLIFS pocket annotation from file.
    
    Expected format: CSV with columns 'residue_number', 'chain_id'
    """
    df = pd.read_csv(annotation_file)
    return list(zip(df['residue_number'], df['chain_id']))

def main():
    """Example usage."""
    # Initialize generator (no parameters needed - uses KLIFS standard)
    generator = KLIFSIFPGenerator()
    
    # Example for single structure
    cif_file = "example_kinase_complex.cif"
    
    # Load KLIFS annotation (you'll need to provide this)
    klifs_residues = load_klifs_annotation("klifs_pocket_annotation.csv")
    
    # Generate IFP
    try:
        binary_ifp = generator.generate_ifp(
            cif_file, 
            klifs_residues,
            output_prefix="example_output"
        )
        
        print(f"Generated KLIFS binary IFP with {len(binary_ifp)} bits (85 residues × 7 properties)")
        print(f"Active interactions: {np.sum(binary_ifp)}")
        
    except Exception as e:
        print(f"Error: {e}")