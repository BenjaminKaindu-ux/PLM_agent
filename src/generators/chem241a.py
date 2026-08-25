"""CHEM 241A item generator — RDKit is the ground-truth oracle.

Two categories, mirroring MATH223's two-category shape:
- functional_group_id: RDKit's own maintained `Chem.Fragments.fr_*` counters (not our
  hand-rolled SMARTS) identify the functional group. Every candidate molecule is verified
  at generation time to trip exactly one target counter and none of the others' — an item
  is never served on an assumption about what a SMILES "should" contain.
- stereo_center_config: R/S is never guessed from the SMILES's @/@@ notation by us — it's
  computed by RDKit's CIP implementation (`Chem.FindMolChiralCenters`) at generation time.
"""

import random
from pathlib import Path

from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Fragments
from rdkit.Chem.Draw import rdMolDraw2D

CATEGORIES = {
    "functional_group_id": {"rt_threshold_s": 10.0},
    "stereo_center_config": {"rt_threshold_s": 12.0},
}

# Hand-authored display name + one-line cue (established facts, not LLM-generated at
# runtime) — same authorship model as MATH223's _FEEDBACK / ARH's _STYLE_INFO.
_FUNCTIONAL_GROUPS = {
    "alcohol": {
        "display": "Alcohol",
        "counter": Fragments.fr_Al_OH,
        "cue": "An -OH (hydroxyl) group bonded to an sp3 carbon.",
        "bank": ["CCO", "CCCO", "CC(C)O", "CCCCO", "OC1CCCCC1", "OCc1ccccc1", "CC(C)CO", "CCCCCO"],
    },
    "ketone": {
        "display": "Ketone",
        "counter": Fragments.fr_ketone,
        "cue": "A C=O (carbonyl) flanked by two carbon-containing groups, no O/N/H directly attached.",
        "bank": ["CC(=O)C", "CCC(=O)C", "CCC(=O)CC", "O=C1CCCCC1", "CC(=O)c1ccccc1", "CCCCC(=O)C"],
    },
    "carboxylic_acid": {
        "display": "Carboxylic acid",
        "counter": Fragments.fr_COO,
        "cue": "A -COOH group: carbonyl carbon also bonded to a hydroxyl.",
        "bank": ["CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "OC(=O)c1ccccc1", "CCCCC(=O)O", "CC(C)C(=O)O"],
    },
    "amine": {
        "display": "Amine",
        "counter": Fragments.fr_NH2,
        "cue": "A primary amine: -NH2 bonded to carbon, no carbonyl nearby.",
        "bank": ["CCN", "CCCN", "CC(C)N", "CCCCN", "NCc1ccccc1", "NC1CCCCC1"],
    },
    "ester": {
        "display": "Ester",
        "counter": Fragments.fr_ester,
        "cue": "A carbonyl carbon bonded to -O-R (an alkoxy group), derived from an acid + alcohol.",
        "bank": ["CC(=O)OCC", "CC(=O)OC", "CCC(=O)OCC", "COC(=O)c1ccccc1", "CC(=O)OCCC", "CCCC(=O)OCC"],
    },
}

# Chiral SMILES with an explicit @/@@ stereocenter — R/S is computed by RDKit below,
# never assumed from which symbol was used.
_CHIRAL_BANK = [
    "C[C@H](Br)CC", "C[C@@H](Br)CC",
    "C[C@H](Cl)CC", "C[C@@H](Cl)CC",
    "C[C@H](O)CC", "C[C@@H](O)CC",
    "C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O",
    "N[C@H](C)C(=O)O", "N[C@@H](C)C(=O)O",
    "[C@H](Br)(Cl)F", "[C@@H](Br)(Cl)F",
]

_STEREO_CUE = (
    "CIP priority (atomic number of attached groups) is ranked highest to lowest; viewed "
    "with the lowest-priority group pointing away, priorities 1→2→3 going clockwise = R, "
    "counterclockwise = S."
)


def _render(mol: Chem.Mol, highlight_atoms: list[int]) -> Image.Image:
    AllChem.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(450, 450)
    d.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    d.FinishDrawing()
    png_bytes = d.GetDrawingText()
    import io

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def _make_functional_group_item(rng: random.Random, difficulty: int) -> dict:
    target = rng.choice(list(_FUNCTIONAL_GROUPS))
    info = _FUNCTIONAL_GROUPS[target]
    candidates = info["bank"][:]
    rng.shuffle(candidates)

    for smiles in candidates:
        mol = Chem.MolFromSmiles(smiles)
        target_count = info["counter"](mol)
        others_clean = all(
            other_info["counter"](mol) == 0 for name, other_info in _FUNCTIONAL_GROUPS.items() if name != target
        )
        if target_count > 0 and others_clean:
            match_atoms = mol.GetSubstructMatch(Chem.MolFromSmarts(_smarts_for(target))) or []
            break
    else:
        raise RuntimeError(f"no unambiguous exemplar found for {target!r} in the bank")

    choices_keys = list(_FUNCTIONAL_GROUPS.keys())
    rng.shuffle(choices_keys)
    correct_idx = choices_keys.index(target)
    choices = [_FUNCTIONAL_GROUPS[k]["display"] for k in choices_keys]

    seed = rng.randint(0, 10**6)
    return {
        "id": f"chem241a.groupsense.{seed:06d}",
        "course": "CHEM241A",
        "category": "GroupSense",
        # Matches the CATEGORIES key ("functional_group_id"), same pattern as
        # math223/arh — NOT `target` (the specific group), which isn't a key any
        # ArtsTracker is built with and would KeyError on tracker.record(). The
        # specific group actually chosen is still recoverable from `feedback`/
        # `ground_truth_method`/`provenance`.
        "subcategory": "functional_group_id",
        "stimulus": {"type": "pil_image", "image": _render(mol, list(match_atoms))},
        "prompt": "What functional group is highlighted in this structure?",
        "choices": choices,
        "correct": correct_idx,
        "feedback": f"{info['display']} — {info['cue']}",
        "ground_truth_method": f"RDKit Fragments.fr_{target}-equivalent counter on {smiles} == {target_count} "
        f"(all other counters == 0)",
        "difficulty": difficulty,
        "transfer": False,
        "provenance": {"generator": "template_groupsense_v1", "seed": seed, "smiles": smiles},
    }


def _smarts_for(group: str) -> str:
    # Only used to compute which atoms to highlight — the actual ground-truth label
    # comes from the verified fr_* counters above, not this SMARTS.
    return {
        "alcohol": "[CX4][OX2H]",
        "ketone": "[#6][CX3](=O)[#6]",
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "amine": "[NX3;H2][CX4]",
        "ester": "[#6][CX3](=O)[OX2][#6]",
    }[group]


def _make_stereo_item(rng: random.Random, difficulty: int) -> dict:
    smiles = rng.choice(_CHIRAL_BANK)
    mol = Chem.MolFromSmiles(smiles)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    if not centers:
        raise RuntimeError(f"no chiral center found for {smiles!r} — bank entry is invalid")
    atom_idx, label = centers[0]  # label is 'R' or 'S', computed by RDKit — never assumed

    choices_keys = ["R", "S"]
    rng.shuffle(choices_keys)
    correct_idx = choices_keys.index(label)

    seed = rng.randint(0, 10**6)
    return {
        "id": f"chem241a.stereosense.{seed:06d}",
        "course": "CHEM241A",
        "category": "StereoSense",
        "subcategory": "stereo_center_config",
        "stimulus": {"type": "pil_image", "image": _render(mol, [atom_idx])},
        "prompt": "What is the CIP configuration (R or S) at the highlighted stereocenter?",
        "choices": choices_keys,
        "correct": correct_idx,
        "feedback": f"This center is ({label}) — {_STEREO_CUE}",
        "ground_truth_method": f"RDKit Chem.FindMolChiralCenters on {smiles}, atom {atom_idx} == {label}",
        "difficulty": difficulty,
        "transfer": False,
        "provenance": {"generator": "template_stereosense_v1", "seed": seed, "smiles": smiles},
    }


def make_item(category: str, rng: random.Random | None = None, difficulty: int = 1) -> dict:
    rng = rng or random.Random()
    if category == "functional_group_id":
        return _make_functional_group_item(rng, difficulty)
    elif category == "stereo_center_config":
        return _make_stereo_item(rng, difficulty)
    raise ValueError(f"unknown CHEM241A category {category!r}; expected one of {list(CATEGORIES)}")
