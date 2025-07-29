#! /usr/bin/env python3

import os
import argparse


STR_YAML = """
version: 1
sequences:
  - protein:
      id: A
      sequence: <TARGET>
  - ligand:
      id: B
      smiles: '<LIGAND>'
properties:
  - affinity:
      binder: B
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Create a YAML file for Boltz-2.")

    parser.add_argument("--targetSequence", type=str, required=True, help="Sequence of the target protein")
    parser.add_argument("--ligandSMILES", type=str, required=True, help="SMILES representation of the ligand")
    parser.add_argument("--uuid", type=str, required=True, help="Unique identifier for the complex")

    return parser.parse_args()


def create_str_yaml(
    target: str,
    ligand: str
) -> str:
    """Create a YAML string with the given target and ligand.

    Parameters
    ----------
    target : str
        The target sequence.
    ligand : str
        The ligand SMILES string.

    Returns
    -------
    str
        The formatted YAML string with the kinase and ligand.
    """
    return STR_YAML.replace("<TARGET>", target).replace("<LIGAND>", ligand)

def save_yaml_file(str_yaml: str, uuid: str) -> None:
    """Save the YAML string to a file.

    Parameters
    ----------
    str_yaml : str
        The YAML string to save.
    uuid : str
        The unique identifier for the complex, used in the filename.
    """
    filepath = os.path.join(os.getcwd(), f"{uuid}.yaml")
    with open(filepath, "w") as f:
        f.write(str_yaml)

def main():
    args = parse_args()
    str_yaml = create_str_yaml(args.targetSequence, args.ligandSMILES)
    save_yaml_file(str_yaml, args.uuid)

if __name__ == "__main__":
    main()
    return parser.parse_args()


def create_str_yaml(
    target: str,
    ligand: str
) -> str:
    """Create a YAML string with the given target and ligand.

    Parameters
    ----------
    target : str
        The target sequence.
    ligand : str
        The ligand SMILES string.

    Returns
    -------
    str
        The formatted YAML string with the kinase and ligand.
    """
    return STR_YAML.replace("<TARGET>", target).replace("<LIGAND>", ligand)

def save_yaml_file(str_yaml: str, uuid: str) -> None:
    """Save the YAML string to a file.

    Parameters
    ----------
    str_yaml : str
        The YAML string to save.
    uuid : str
        The unique identifier for the complex, used in the filename.
    """
    filepath = os.path.join(os.getcwd(), f"{uuid}.yaml")
    with open(filepath, "w") as f:
        f.write(str_yaml)

def main():
    args = parse_args()
    str_yaml = create_str_yaml(args.targetSequence, args.ligandSMILES)
    save_yaml_file(str_yaml, args.uuid)

if __name__ == "__main__":
    main()