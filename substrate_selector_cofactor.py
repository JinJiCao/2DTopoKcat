from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


                                                                              
                                                                
RDLogger.DisableLog("rdApp.warning")


METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
}

WATER_SMILES = {"O", "[OH-]", "[H]O[H]"}
SIMPLE_ION_SMILES = {
    "[H+]", "[Li+]", "[Na+]", "[K+]", "[Rb+]", "[Cs+]",
    "[Mg+2]", "[Ca+2]", "[Sr+2]", "[Ba+2]", "[Al+3]",
    "[F-]", "F", "[Cl-]", "Cl", "[Br-]", "Br", "[I-]", "I",
    "[NH4+]",
}
SIMPLE_INORGANIC_FORMULAS = {
    "H2O", "HCl", "HBr", "HI", "HF", "NaCl", "KCl", "LiCl",
    "MgCl2", "CaCl2", "NaOH", "KOH", "Na2SO4", "K2SO4",
    "NaNO3", "KNO3", "NH4Cl", "Na2CO3", "K2CO3", "H2SO4",
    "H3PO4", "HNO3",
}


def _query(smarts: str):
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise RuntimeError(f"Invalid internal SMARTS: {smarts}")
    return query


                                                                    
                                                                 
ADENINE_QUERIES = (
    _query("n1cnc2c(N)ncnc12"),
    _query("N[c]1[n][cH][n][c]2[n][cH][n][c]12"),
)
NICOTINAMIDE_QUERIES = (
    _query("[n+]1ccccc1C(=O)N"),
    _query("n1ccccc1C(=O)N"),
    _query("N=C(O)c1ccc[n+]c1"),
)
PHOSPHORUS_QUERY = _query("[#15]")
CARBON_QUERY = _query("[#6]")


@dataclass(frozen=True)
class ComponentInfo:
    smiles: str
    canonical_smiles: str
    heavy_atoms: int
    total_atoms: int
    carbon_atoms: int
    nitrogen_atoms: int
    oxygen_atoms: int
    phosphorus_atoms: int
    sulfur_atoms: int
    formal_charge: int
    formula: str
    component_class: str
    cofactor_name: str


@dataclass(frozen=True)
class SelectionResult:
    original_smiles: str
    selected_smiles: str
    selection_strategy: str
    num_components: int
    num_valid_components: int
    removed_spectators: Tuple[str, ...]
    removed_cofactors: Tuple[str, ...]
    removed_cofactor_names: Tuple[str, ...]
    candidate_noncofactors: Tuple[str, ...]
    selected_component_class: str
    selected_cofactor_name: str

    def audit_record(self) -> Dict[str, object]:
        record = asdict(self)
        for key in (
            "removed_spectators", "removed_cofactors",
            "removed_cofactor_names", "candidate_noncofactors",
        ):
            record[key] = " || ".join(record[key])
        return record


def _has_any(mol, queries: Sequence[Chem.Mol]) -> bool:
    return any(mol.HasSubstructMatch(query) for query in queries)


def _count_element(mol, atomic_number: int) -> int:
    return sum(atom.GetAtomicNum() == atomic_number for atom in mol.GetAtoms())


def identify_cofactor(mol) -> str:

    heavy = mol.GetNumHeavyAtoms()
    n_n = _count_element(mol, 7)
    n_p = _count_element(mol, 15)
    n_s = _count_element(mol, 16)
    has_adenine = _has_any(mol, ADENINE_QUERIES)
    has_nicotinamide = _has_any(mol, NICOTINAMIDE_QUERIES)
    has_positive_sulfur = any(
        atom.GetAtomicNum() == 16 and atom.GetFormalCharge() > 0
        for atom in mol.GetAtoms()
    )

                                                                       
    if has_adenine and has_positive_sulfur and heavy >= 20:
        return "SAM"

                                                                                
    if has_adenine and n_p >= 2 and n_s >= 1 and heavy >= 35:
        return "CoA_or_acyl_CoA"

                                                                           
    if has_adenine and has_nicotinamide and n_p >= 2 and heavy >= 35:
        return "NAD_or_NADP"

                                                                           
                                                                
    if has_adenine and n_p >= 2 and n_n >= 8 and heavy >= 45:
        return "FAD_like"

                                                                           
                                                                  
    if has_adenine and 2 <= n_p <= 3 and 20 <= heavy <= 40:
        return "ATP_or_ADP"

    return ""


def analyze_component(smiles: str) -> Optional[ComponentInfo]:
    smiles = str(smiles).strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    formula = rdMolDescriptors.CalcMolFormula(mol)
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    carbon_atoms = atoms.count(6)
    cofactor_name = identify_cofactor(mol)

    if smiles in WATER_SMILES or formula == "H2O":
        component_class = "water"
    elif (
        smiles in SIMPLE_ION_SMILES
        or (mol.GetNumAtoms() == 1 and atoms[0] in METAL_ATOMIC_NUMBERS)
    ):
        component_class = "metal_or_simple_ion"
    elif formula in SIMPLE_INORGANIC_FORMULAS:
        component_class = "salt"
    elif carbon_atoms == 0 and mol.GetNumHeavyAtoms() <= 8:
        component_class = "small_inorganic"
    elif cofactor_name:
        component_class = "cofactor"
    elif carbon_atoms > 0:
        component_class = "organic"
    else:
        component_class = "other"

    return ComponentInfo(
        smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        total_atoms=mol.GetNumAtoms(),
        carbon_atoms=carbon_atoms,
        nitrogen_atoms=atoms.count(7),
        oxygen_atoms=atoms.count(8),
        phosphorus_atoms=atoms.count(15),
        sulfur_atoms=atoms.count(16),
        formal_charge=formal_charge,
        formula=formula,
        component_class=component_class,
        cofactor_name=cofactor_name,
    )


def _largest(components: Sequence[ComponentInfo]) -> ComponentInfo:
                                                           
    return max(
        components,
        key=lambda item: (item.heavy_atoms, item.total_atoms, item.canonical_smiles),
    )


def select_main_substrate(reaction_smiles: str) -> SelectionResult:
    original = "" if reaction_smiles is None else str(reaction_smiles).strip()
    raw_components = [part.strip() for part in original.split(".") if part.strip()]
    analyzed = [analyze_component(part) for part in raw_components]
    valid = [item for item in analyzed if item is not None]
    if not valid:
        raise ValueError(f"No valid RDKit component in SMILES: {original!r}")

    spectators = [
        item for item in valid
        if item.component_class in {
            "water", "metal_or_simple_ion", "salt", "small_inorganic"
        }
    ]
    cofactors = [item for item in valid if item.component_class == "cofactor"]
    noncofactor_organics = [
        item for item in valid if item.component_class == "organic"
    ]

    if noncofactor_organics:
        selected = _largest(noncofactor_organics)
        strategy = (
            "largest_noncofactor_organic_after_filtering"
            if len(raw_components) > 1
            else "single_organic_component"
        )
    else:
                                                                           
                                                           
        organic_or_cofactor = [
            item for item in valid
            if item.component_class in {"organic", "cofactor"}
        ]
        if organic_or_cofactor:
            selected = _largest(organic_or_cofactor)
            strategy = "all_organic_components_are_cofactors_fallback"
        else:
            selected = _largest(valid)
            strategy = "no_organic_component_fallback"

    removed_cofactors = [item for item in cofactors if item is not selected]
    return SelectionResult(
        original_smiles=original,
        selected_smiles=selected.smiles,
        selection_strategy=strategy,
        num_components=len(raw_components),
        num_valid_components=len(valid),
        removed_spectators=tuple(item.smiles for item in spectators),
        removed_cofactors=tuple(item.smiles for item in removed_cofactors),
        removed_cofactor_names=tuple(item.cofactor_name for item in removed_cofactors),
        candidate_noncofactors=tuple(item.smiles for item in noncofactor_organics),
        selected_component_class=selected.component_class,
        selected_cofactor_name=selected.cofactor_name,
    )
