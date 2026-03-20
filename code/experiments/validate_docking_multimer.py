#!/usr/bin/env python3
"""Validate ColabFold docking outputs are multimer predictions, not monomer folds."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path


def parse_fasta(path: Path):
    """Parse FASTA and return list of (header, sequence)."""
    records = []
    header = None
    seq_parts = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

    if header is not None:
        records.append((header, "".join(seq_parts)))

    return records


def unzip_result_archives(run_dir: Path):
    for zip_file in sorted(run_dir.glob("*.result.zip")):
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(run_dir)


def select_score_json(run_dir: Path):
    unzip_result_archives(run_dir)
    score_files = sorted(run_dir.glob("*_scores_rank_001_*.json"))
    return score_files[0] if score_files else None


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def find_rank1_pdb(run_dir: Path):
    relaxed = sorted(run_dir.glob("*_relaxed_rank_001_*.pdb"))
    if relaxed:
        return relaxed[0]
    unrelaxed = sorted(run_dir.glob("*_unrelaxed_rank_001_*.pdb"))
    return unrelaxed[0] if unrelaxed else None


def pdb_chain_count(pdb_path: Path):
    chains = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if len(line) < 22:
                continue
            chain_id = line[21].strip()
            if chain_id:
                chains.add(chain_id)
    return len(chains), sorted(chains)


def inspect_run(run_dir: Path):
    result = {
        "run_name": run_dir.name,
        "multimer_ok": False,
        "issues": [],
        "score_json": "",
        "ptm": None,
        "iptm": None,
        "pdb": "",
        "pdb_chains": [],
        "fasta": "",
        "fasta_records": 0,
        "fasta_has_colon_sep": False,
    }

    fasta_files = sorted(run_dir.glob("*.fasta"))
    if not fasta_files:
        result["issues"].append("missing_fasta")
    else:
        fasta = fasta_files[0]
        result["fasta"] = str(fasta)
        try:
            records = parse_fasta(fasta)
            result["fasta_records"] = len(records)
            if len(records) == 1 and records:
                seq = records[0][1]
                result["fasta_has_colon_sep"] = ":" in seq
                if not result["fasta_has_colon_sep"]:
                    result["issues"].append("fasta_not_multimer_syntax")
            else:
                result["issues"].append("fasta_not_single_record")
        except Exception:
            result["issues"].append("fasta_parse_error")

    score_json = select_score_json(run_dir)
    if score_json is None:
        result["issues"].append("missing_score_json")
    else:
        result["score_json"] = str(score_json)
        try:
            scores = read_json(score_json)
            result["ptm"] = scores.get("ptm")
            result["iptm"] = scores.get("iptm")
            if result["iptm"] is None:
                result["issues"].append("missing_iptm")
        except Exception:
            result["issues"].append("score_json_parse_error")

    log_path = run_dir / "log.txt"
    if log_path.exists():
        with open(log_path, "r") as f:
            log_text = f.read().lower()
            if "single chain prediction" in log_text or "calculating extra pTM is not supported" in log_text:
                result["issues"].append("log_single_chain_warning")

    pdb = find_rank1_pdb(run_dir)
    if pdb is None:
        result["issues"].append("missing_rank1_pdb")
    else:
        result["pdb"] = str(pdb)
        try:
            n_chains, chains = pdb_chain_count(pdb)
            result["pdb_chains"] = chains
            if n_chains < 2:
                result["issues"].append("pdb_single_chain")
        except Exception:
            result["issues"].append("pdb_chain_count_error")

    result["multimer_ok"] = (
        result["issues"] == []
        and result["fasta_records"] == 1
        and result["fasta_has_colon_sep"]
        and result["iptm"] is not None
    )

    return result


def write_sample_multimer_fasta(output_path: Path, conotoxin_seq: str = "CKGKGAKCSRLMYDCCTGSCRSGKCG", receptor_seq: str = None):
    receptor_seq = receptor_seq or (
        "MSPGFSWLRLRLLLAIVCIVCVVFFILEDPIFPWVIFNIVGSFVNVAELVLKLRFRIMRILRIFREKHQRE"
        "IAEMTLSIKDTVVVQTSGMQKIGAGGSKEQEAEEAVVESIEKIVLNTSFYYRNILAHAKVPSEMFLEGTAE"
        "DDVEAIQSQIISVERAGFIEVASVVKGDNIGFEKDSLQASLDEFIEAKELSEEKLLGDEVGDEALSAGGSG"
        "FRYEVLSILSITKFHFGVALCLGEYHGTVFIILGSFGSAYTGDFNAFALQQFQFKGIMYVVMGFFAVSYLL"
        "FIGWATGSQAMGFDIAPHHVLRFLEFVQAMRHGDMYNWEIIVLLGIFFVLIIVGAGVHSLISNSLQAGSRI"
    )
    complex_seq = f"{receptor_seq}:{conotoxin_seq}"
    output_path = Path(output_path)
    output_path.write_text(
        ">Cav2.2_pore|sample_conotoxin Multimer test input\n"
        f"{complex_seq}\n",
        encoding="utf-8",
    )
    return output_path, complex_seq.count(":")


def validate_fasta_format(fasta_path: Path):
    result = {
        "path": str(fasta_path),
        "is_valid": False,
        "issues": [],
    }
    try:
        records = parse_fasta(fasta_path)
        if len(records) != 1:
            result["issues"].append("not_single_record")
        else:
            seq = records[0][1]
            if ":" not in seq:
                result["issues"].append("missing_colon_separator")
            else:
                parts = seq.split(":")
                if len(parts) != 2 or any(len(p) == 0 for p in parts):
                    result["issues"].append("invalid_chain_separator")
        result["is_valid"] = len(result["issues"]) == 0
    except Exception as e:
        result["issues"].append(f"parse_error: {type(e).__name__}")
    return result


def run_docking_validation(docking_dir: Path, run_filter: str | None = None, limit: int | None = None):
    results = []

    runs = [p for p in sorted(docking_dir.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if run_filter:
        runs = [r for r in runs if run_filter in r.name]
    if limit is not None:
        runs = runs[:limit]

    for run_dir in runs:
        results.append(inspect_run(run_dir))

    return results


def print_validation_report(results):
    total = len(results)
    statuses = Counter("ok" if r["multimer_ok"] else "bad" for r in results)
    issues = Counter()
    for r in results:
        for issue in r["issues"]:
            issues[issue] += 1

    print(f"\nDocking validation checked: {total} run(s)")
    print(f"Multimer-valid: {statuses['ok']} | Invalid/missing: {statuses['bad']}")
    if issues:
        print("Issue counts:")
        for issue, count in sorted(issues.items(), key=lambda x: x[0]):
            print(f"  - {issue}: {count}")

    for r in results:
        status = "OK" if r["multimer_ok"] else "BAD"
        if r["multimer_ok"]:
            print(f"[{status}] {r['run_name']}  ipTM={r['iptm']}  pTM={r['ptm']}  PDB-chains={','.join(r['pdb_chains'])}")
        else:
            print(f"[{status}] {r['run_name']}  issues={','.join(r['issues'])}  score_file={r['score_json']}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Validate ColabFold docking outputs are multimer predictions")
    parser.add_argument(
        "--docking-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "omega_conotoxin" / "docking_validation"),
        help="Path to docking_validation directory",
    )
    parser.add_argument("--run-filter", default=None, help="Substring to filter run directories")
    parser.add_argument("--limit", type=int, default=None, help="Only inspect first N matching runs")
    parser.add_argument(
        "--sample-fasta",
        default=None,
        help="Write a sample multimer FASTA to this path and validate format (no run directory scan)",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.sample_fasta:
        sample_path = Path(args.sample_fasta).resolve()
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        _, colon_count = write_sample_multimer_fasta(sample_path)
        check = validate_fasta_format(sample_path)
        print(f"\nSample multimer FASTA written: {sample_path}")
        print(f"Colon separators: {colon_count}")
        print(f"Format valid: {'YES' if check['is_valid'] else 'NO'}")
        if check["issues"]:
            print(f"Issues: {', '.join(check['issues'])}")
            return 1
        return 0

    docking_dir = Path(args.docking_dir).resolve()
    if not docking_dir.exists():
        print(f"ERROR: docking directory does not exist: {docking_dir}")
        return 1

    results = run_docking_validation(docking_dir, args.run_filter, args.limit)
    print_validation_report(results)

    any_bad = any(not r["multimer_ok"] for r in results)
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
