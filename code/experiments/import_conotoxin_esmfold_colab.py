#!/usr/bin/env python3
"""Validate and import the ω-conotoxin ESMFold Colab fallback archive.

The companion notebook writes PDBs using the exact content-addressed filenames
expected by run_conotoxin_structure_validation.jl.  Before importing anything,
this script compares a calibration prediction with the existing REST-API result
to catch a model-version or recycle-count mismatch.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
STRUCT_DIR = CODE_DIR / "data" / "omega_conotoxin" / "structures"
TMALIGN = CODE_DIR / "bin" / "TMalign"

CALIBRATION_SEQUENCE = "CKGKGASCTRLSYDCCTGSCSSGKCG"
CALIBRATION_BASENAME = (
    "CALIBRATION_SA_strong_SA_strong_0031_1b2d20c06460bc2d.pdb"
)
CALIBRATION_REFERENCE = (
    STRUCT_DIR / "SA_strong_SA_strong_0031_1b2d20c06460bc2d.pdb"
)

# Exact cache targets computed by Julia 1.12 from the current regenerated FASTAs.
EXPECTED = {
    "SA_strong_SA_strong_0372_6ee6b78bd8379515.pdb": "CRRSGSSCSRTMYICCTGRCRSGKCG",
    "SA_strong_SA_strong_0527_47a13fb789a1ba5e.pdb": "CKGKGAPCTRLSYDCCTGSCSSGRCG",
    "SA_strong_SA_strong_0589_28d988bbb71d6b03.pdb": "CKRKGAPCGRTSYDCCSGSCSRGRCG",
    "SA_strong_SA_strong_1209_e8b3cf7e166d9dbf.pdb": "CKGKGAKCSRLSYDCCSGSCSRGKCG",
    "SA_strong_SA_strong_1240_b7c85f18e6f17663.pdb": "CKGKGAPCSRLMYDCCRGSCRSGKCG",
    "SA_strong_SA_strong_1271_7d592013b63efb00.pdb": "CKSTGSSCSPTSYNCCTGSCRPGKCG",
    "SA_strong_SA_strong_1302_d83a458b7e10123b.pdb": "CKSAGKSCRRTAYDCCRGSCRSGKCG",
    "SA_strong_SA_strong_1333_2b6987efbc538240.pdb": "CKSKGASCSKTMYDCCTGSCRRGRCY",
    "SA_strong_SA_strong_1364_ef898b19e18622e.pdb": "CKGKGASCRRTSYDCCTGSCRSGKCG",
    "SA_strong_SA_strong_1395_dc5509fc03c771d4.pdb": "CKPPGAPCRVSSYNCCSGSCKSKKCT",
    "SA_strong_SA_strong_1488_66926692db1f17f.pdb": "CKSKGSKCRVTSYDCCTGSCRSGRCG",
    "SA_strong_SA_strong_1550_d085e28c3291c457.pdb": "CKGAGAPCSRTAYNCCSGSCNSGRCG",
    "SA_full_SA_full_1178_bbf0e69e966623f0.pdb": "CKEPGAKCPVTSKDCCSGFCTLFFCM",
    "SA_full_SA_full_1209_dbd5f2225d423635.pdb": "CLDGGTKCNRGNSQCCSGWCISLRCL",
    "SA_full_SA_full_1240_2bdd2ab827429963.pdb": "CSSGGSYCSSISYNCCTEFCAYLKCI",
    "SA_full_SA_full_1302_f6523b233cdfd4cc.pdb": "CKAEGEKCSSDSYDCCSGSCAYFKCE",
    "SA_full_SA_full_1550_16650992d88388c5.pdb": "CKPPGSFCRIFSLLCCKYYCSSKVCT",
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def atom_lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.startswith("ATOM")]


def pdb_sequence(path: Path) -> str:
    residues: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for line in atom_lines(path):
        if len(line) < 66 or line[12:16].strip() != "CA":
            continue
        key = (line[21:22], line[22:26], line[26:27])
        if key in seen:
            continue
        seen.add(key)
        residue = line[17:20].strip()
        if residue not in AA3_TO_1:
            raise ValueError(f"unknown residue {residue!r} in {path.name}")
        residues.append(AA3_TO_1[residue])
    return "".join(residues)


def mean_plddt(path: Path) -> float:
    values = []
    for line in atom_lines(path):
        try:
            value = float(line[60:66].strip())
        except (ValueError, IndexError):
            continue
        values.append(value * 100.0 if value < 1.5 else value)
    if not values:
        raise ValueError(f"no pLDDT values in {path.name}")
    return sum(values) / len(values)


def alignment_metrics(query: Path, reference: Path) -> tuple[float, float, int, float]:
    completed = subprocess.run(
        [str(TMALIGN), str(query), str(reference)],
        check=True,
        capture_output=True,
        text=True,
    )
    aligned_length = None
    rmsd = None
    sequence_identity = None
    score = None
    for line in completed.stdout.splitlines():
        if "Aligned length=" in line:
            match = re.search(
                r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+),.*=\s*([\d.]+)",
                line,
            )
            if match:
                aligned_length = int(match.group(1))
                rmsd = float(match.group(2))
                sequence_identity = float(match.group(3))
        if "TM-score=" in line and "Chain_2" in line:
            score = float(line.split("TM-score=", 1)[1].split()[0])
    if None in (score, rmsd, aligned_length, sequence_identity):
        raise ValueError("TM-align did not report complete alignment metrics")
    return score, rmsd, aligned_length, sequence_identity


def validate_pdb(path: Path, sequence: str) -> None:
    if path.stat().st_size <= 100 or len(atom_lines(path)) < len(sequence) * 4:
        raise ValueError(f"{path.name} is not a complete PDB")
    observed = pdb_sequence(path)
    if observed != sequence:
        raise ValueError(
            f"sequence mismatch in {path.name}: expected {sequence}, got {observed}"
        )
    score = mean_plddt(path)
    if not 0.0 < score <= 100.0:
        raise ValueError(f"invalid mean pLDDT {score} in {path.name}")


def safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    by_basename: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        member = Path(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"unsafe ZIP member: {info.filename}")
        basename = member.name
        if basename in by_basename:
            raise ValueError(f"duplicate ZIP basename: {basename}")
        by_basename[basename] = info
    return by_basename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="ZIP downloaded by the Colab notebook")
    args = parser.parse_args()

    if not CALIBRATION_REFERENCE.is_file():
        raise SystemExit(f"missing calibration reference: {CALIBRATION_REFERENCE}")
    if not TMALIGN.is_file():
        raise SystemExit(f"missing TM-align executable: {TMALIGN}")

    with tempfile.TemporaryDirectory(prefix="conotoxin_esmfold_import_") as temp_name:
        temp_dir = Path(temp_name)
        with zipfile.ZipFile(args.archive) as archive:
            members = safe_members(archive)
            required = set(EXPECTED) | {CALIBRATION_BASENAME}
            missing = sorted(required - set(members))
            if missing:
                raise SystemExit("archive is missing: " + ", ".join(missing))
            for basename in required:
                target = temp_dir / basename
                with archive.open(members[basename]) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)

        calibration = temp_dir / CALIBRATION_BASENAME
        validate_pdb(calibration, CALIBRATION_SEQUENCE)
        validate_pdb(CALIBRATION_REFERENCE, CALIBRATION_SEQUENCE)
        calibration_tm, calibration_rmsd, aligned_length, sequence_identity = (
            alignment_metrics(calibration, CALIBRATION_REFERENCE)
        )
        calibration_delta = abs(mean_plddt(calibration) - mean_plddt(CALIBRATION_REFERENCE))
        print(
            f"Calibration: TM-score={calibration_tm:.5f}, "
            f"RMSD={calibration_rmsd:.2f} Å, |ΔpLDDT|={calibration_delta:.4f}"
        )
        # On this 26-residue peptide, ordinary GPU numerical differences produce
        # a disproportionate TM-score change because TM-align's d0 is only 0.96 Å.
        # Require full sequence alignment and a tight absolute RMSD/pLDDT match as
        # the primary parity checks, with TM-score >= 0.95 as a backstop.
        if (
            aligned_length != len(CALIBRATION_SEQUENCE)
            or sequence_identity != 1.0
            or calibration_rmsd > 0.25
            or calibration_tm < 0.95
            or calibration_delta > 0.5
        ):
            raise SystemExit(
                "calibration failed; refusing to mix these predictions with the API cache"
            )

        imported = 0
        skipped = 0
        for basename, sequence in EXPECTED.items():
            source = temp_dir / basename
            validate_pdb(source, sequence)
            target = STRUCT_DIR / basename
            if target.is_file() and target.stat().st_size > 100:
                skipped += 1
                continue
            shutil.copy2(source, target)
            imported += 1

    print(f"Imported {imported} PDBs; preserved {skipped} existing cached PDBs.")
    print("Next: cd code && julia experiments/run_conotoxin_structure_validation.jl")


if __name__ == "__main__":
    main()
