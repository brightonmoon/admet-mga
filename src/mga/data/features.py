"""
Molecular feature extraction for MGA.

This module provides functions to extract atom and bond features
from molecular structures using RDKit.
"""

from __future__ import annotations

from typing import List

import numpy as np
from rdkit import Chem


def one_of_k_encoding(x, allowable_set: List) -> List[bool]:
    """One-hot encoding with strict set membership."""
    if x not in allowable_set:
        raise ValueError(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set: List) -> List[bool]:
    """One-hot encoding with unknown handling (maps to last element)."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def one_of_k_atompair_encoding(x: str, allowable_set: List[List[str]]) -> List[bool]:
    """One-hot encoding for atom pairs."""
    for atompair in allowable_set:
        if x in atompair:
            x = atompair
            break
        elif atompair == allowable_set[-1]:
            x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(
    atom,
    explicit_H: bool = False,
    use_chirality: bool = True,
) -> np.ndarray:
    """
    Extract atom features.

    Features include:
    - Atom type (16 types + other)
    - Degree (0-6)
    - Formal charge
    - Number of radical electrons
    - Hybridization (6 types)
    - Aromaticity
    - Number of hydrogens (if not explicit_H)
    - Chirality (if use_chirality)

    Args:
        atom: RDKit atom object
        explicit_H: Whether hydrogens are explicit in the molecule
        use_chirality: Whether to include chirality features

    Returns:
        Feature array (~40 dimensions)
    """
    # Atom type
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As',
            'Se', 'Br', 'Te', 'I', 'At', 'other'
        ]
    )

    # Degree
    results += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])

    # Formal charge and radical electrons
    results += [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]

    # Hybridization
    results += one_of_k_encoding_unk(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other'
        ]
    )

    # Aromaticity
    results += [atom.GetIsAromatic()]

    # Number of hydrogens
    if not explicit_H:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    # Chirality
    if use_chirality:
        try:
            results += one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']
            ) + [atom.HasProp('_ChiralityPossible')]
        except KeyError:
            # Chirality information not available for this atom
            results += [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def bond_features(
    bond,
    use_chirality: bool = True,
    atompair: bool = False,
) -> np.ndarray:
    """
    Extract bond features (one-hot encoded).

    Features include:
    - Bond type (single, double, triple, aromatic)
    - Is conjugated
    - Is in ring
    - Stereo configuration (if use_chirality)
    - Atom pair type (if atompair)

    Args:
        bond: RDKit bond object
        use_chirality: Whether to include stereo features
        atompair: Whether to include atom pair features

    Returns:
        Feature array (one-hot encoded)
    """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]

    if use_chirality:
        bond_feats += one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )

    if atompair:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        bond_feats += one_of_k_atompair_encoding(
            atom_pair_str,
            [
                ['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                ['CBr', 'BrC'], ['others']
            ]
        )

    return np.array(bond_feats).astype(int)


def etype_features(
    bond,
    use_chirality: bool = True,
    atompair: bool = True,
) -> int:
    """
    Extract edge type as integer index.

    Combines bond type, conjugation, ring membership, stereo, and atom pair
    into a single integer index for use with RelGraphConv.

    Args:
        bond: RDKit bond object
        use_chirality: Whether to include stereo
        atompair: Whether to include atom pair

    Returns:
        Integer edge type index
    """
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
    ]

    # Bond type index (0-3)
    a = 0
    for i, m in enumerate(bond_feats_1):
        if m:
            a = i
            break

    # Conjugation (0-1)
    b = 1 if bond.GetIsConjugated() else 0

    # Ring membership (0-1)
    c = 1 if bond.IsInRing() else 0

    index = a * 1 + b * 4 + c * 8

    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )
        d = 0
        for i, m in enumerate(bond_feats_4):
            if m:
                d = i
                break
        index = index + d * 16

    if atompair:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        bond_feats_5 = one_of_k_atompair_encoding(
            atom_pair_str,
            [
                ['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                ['CBr', 'BrC'], ['others']
            ]
        )
        e = 0
        for i, m in enumerate(bond_feats_5):
            if m:
                e = i
                break
        index = index + e * 64

    return index
