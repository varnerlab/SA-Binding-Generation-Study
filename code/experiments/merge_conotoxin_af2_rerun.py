#!/usr/bin/env python3
"""
Merge a partial AF2 rerun (sa_strong + sa_full only, from ColabFold_AF2_Conotoxin.ipynb)
with the existing, unaffected "stored" rows into a complete three-group raw/summary CSV
pair, without ever rerunning ColabFold on "stored".

Usage:
    python3 experiments/merge_conotoxin_af2_rerun.py <path-to-downloaded-raw-csv>

The downloaded CSV is exactly what the notebook's cell 12 writes
(conotoxin_af2_validation_raw.csv): columns name,source,plddt,tmscore,predictor, with
source in {sa_strong, sa_full}, 50 rows each. This script does not trust that file blindly:
it re-validates schema, row counts, name uniqueness, name identity against the af2_input
FASTAs actually used, and score finiteness before touching anything on disk. Nothing in
code/data/af2_results is overwritten until every check has passed.
"""
import csv
import math
import shutil
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
AF2_INPUT_DIR = CODE_DIR / "data" / "af2_input" / "omega_conotoxin"
AF2_RESULTS_DIR = CODE_DIR / "data" / "af2_results" / "omega_conotoxin"
EXISTING_RAW = AF2_RESULTS_DIR / "conotoxin_af2_validation_raw_corrected.csv"
OUT_RAW = AF2_RESULTS_DIR / "conotoxin_af2_validation_raw_corrected.csv"
OUT_SUMMARY = AF2_RESULTS_DIR / "conotoxin_af2_validation_summary_corrected.csv"

RERUN_CATEGORIES = ("sa_strong", "sa_full")
REQUIRED_COLUMNS = ["name", "source", "plddt", "tmscore", "predictor"]


def read_fasta_names(path):
    names = set()
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                names.add(line[1:].strip())
    return names


def read_csv_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(f"{path}: expected columns {REQUIRED_COLUMNS}, got {reader.fieldnames}")
        return list(reader)


def validate_finite(rows, label):
    for row in rows:
        for col in ("plddt", "tmscore"):
            try:
                val = float(row[col])
            except ValueError:
                raise ValueError(f"{label}: {row['name']}.{col} = {row[col]!r} is not a number")
            if not math.isfinite(val):
                raise ValueError(f"{label}: {row['name']}.{col} = {val} is not finite")


def validate_rerun_rows(rows):
    if len(rows) != 100:
        raise ValueError(f"Downloaded raw CSV has {len(rows)} rows, expected exactly 100 (50+50)")

    by_cat = {}
    for row in rows:
        if row["source"] not in RERUN_CATEGORIES:
            raise ValueError(f"Unexpected source {row['source']!r} in downloaded raw CSV; "
                              f"expected only {RERUN_CATEGORIES}")
        if row["predictor"] != "AF2":
            raise ValueError(f"{row['name']}: predictor is {row['predictor']!r}, expected 'AF2'")
        by_cat.setdefault(row["source"], []).append(row)

    for cat in RERUN_CATEGORIES:
        cat_rows = by_cat.get(cat, [])
        if len(cat_rows) != 50:
            raise ValueError(f"{cat}: {len(cat_rows)} rows, expected exactly 50")
        names = [r["name"] for r in cat_rows]
        if len(set(names)) != 50:
            raise ValueError(f"{cat}: duplicate names in downloaded raw CSV")

        fasta_path = AF2_INPUT_DIR / f"{cat}.fasta"
        expected_names = read_fasta_names(fasta_path)
        if set(names) != expected_names:
            missing = expected_names - set(names)
            extra = set(names) - expected_names
            raise ValueError(
                f"{cat}: downloaded names don't match {fasta_path}. "
                f"Missing: {sorted(missing)}. Unexpected: {sorted(extra)}."
            )

    validate_finite(rows, "downloaded raw CSV")


def load_existing_stored_rows():
    if not EXISTING_RAW.exists():
        raise FileNotFoundError(f"Missing existing AF2 results: {EXISTING_RAW}")
    existing_rows = read_csv_rows(EXISTING_RAW)
    stored_rows = [r for r in existing_rows if r["source"] == "stored"]
    if len(stored_rows) != 50:
        raise ValueError(f"Expected exactly 50 'stored' rows in {EXISTING_RAW}, found {len(stored_rows)}")
    names = [r["name"] for r in stored_rows]
    if len(set(names)) != 50:
        raise ValueError(f"{EXISTING_RAW}: duplicate 'stored' names")
    for row in stored_rows:
        if row["predictor"] != "AF2":
            raise ValueError(f"stored/{row['name']}: predictor is {row['predictor']!r}, expected 'AF2'")
    validate_finite(stored_rows, "existing stored rows")
    return stored_rows


def compute_summary(rows):
    summary = []
    for source in ("stored", "sa_strong", "sa_full"):
        subset = [r for r in rows if r["source"] == source]
        if not subset:
            continue
        plddts = [float(r["plddt"]) for r in subset]
        tms = [float(r["tmscore"]) for r in subset]
        n = len(subset)
        plddt_mean = sum(plddts) / n
        tm_mean = sum(tms) / n
        # Population SD (ddof=0), matching np.std(...) as used in the notebook's own
        # summary cell — must match convention, not just be "a" standard deviation.
        plddt_std = math.sqrt(sum((x - plddt_mean) ** 2 for x in plddts) / n)
        tm_std = math.sqrt(sum((x - tm_mean) ** 2 for x in tms) / n)
        summary.append({
            "source": source, "n": n,
            "plddt_mean": f"{plddt_mean:.1f}", "plddt_std": f"{plddt_std:.1f}",
            "tmscore_mean": f"{tm_mean:.4f}", "tmscore_std": f"{tm_std:.4f}",
        })
    return summary


def main(downloaded_raw_path):
    downloaded_raw_path = Path(downloaded_raw_path)
    rerun_rows = read_csv_rows(downloaded_raw_path)
    validate_rerun_rows(rerun_rows)
    stored_rows = load_existing_stored_rows()

    merged_rows = stored_rows + \
        [r for r in rerun_rows if r["source"] == "sa_strong"] + \
        [r for r in rerun_rows if r["source"] == "sa_full"]
    if len(merged_rows) != 150:
        raise ValueError(f"Merged row count is {len(merged_rows)}, expected 150")

    summary_rows = compute_summary(merged_rows)
    if len(summary_rows) != 3:
        raise ValueError(f"Summary has {len(summary_rows)} sources, expected 3")

    # Write to temp files first; only replace the committed files once both are known-good.
    tmp_raw = OUT_RAW.with_suffix(".tmp.csv")
    tmp_summary = OUT_SUMMARY.with_suffix(".tmp.csv")

    with open(tmp_raw, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(merged_rows)

    with open(tmp_summary, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "n", "plddt_mean", "plddt_std", "tmscore_mean", "tmscore_std"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Re-read what was just written and re-validate before committing, to catch any
    # writer-side bug rather than trusting the in-memory data we just wrote.
    reloaded = read_csv_rows(tmp_raw)
    if len(reloaded) != 150:
        raise ValueError("Post-write validation failed: raw CSV round-trip row count mismatch")
    validate_finite(reloaded, "post-write raw CSV")

    shutil.move(str(tmp_raw), str(OUT_RAW))
    shutil.move(str(tmp_summary), str(OUT_SUMMARY))

    print(f"Merged {len(merged_rows)} rows (50 stored + 50 sa_strong + 50 sa_full) -> {OUT_RAW}")
    print(f"Summary -> {OUT_SUMMARY}")
    for row in summary_rows:
        print(f"  {row['source']:<12} n={row['n']:>3}  "
              f"pLDDT={row['plddt_mean']}+-{row['plddt_std']}  "
              f"TM={row['tmscore_mean']}+-{row['tmscore_std']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <path-to-downloaded-raw-csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
