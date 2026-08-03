#!/usr/bin/env python3
"""Score matched rho=500 SA and profile-HMM Kunitz libraries with ESM2-650M."""

import csv
import statistics
from pathlib import Path

from score_esm2_perplexity import ESM2Scorer, parse_fasta


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
DATA_DIR = CODE_DIR / "data" / "profile_hmm_conditioning"
INPUTS = {
    "SA_multiplicity_rho500": DATA_DIR / "kunitz_sa_rho500_rep1.fasta",
    "HMM_weighted_rho500": DATA_DIR / "kunitz_rho500_rep1.fasta",
}
OUTPUT_RAW = DATA_DIR / "kunitz_rho500_matched_esm2_raw.csv"
OUTPUT_SUMMARY = DATA_DIR / "kunitz_rho500_matched_esm2_summary.csv"
MAX_SEQUENCES = 50


def main():
    scorer = ESM2Scorer()
    results = []
    for source, path in INPUTS.items():
        sequences = parse_fasta(path)[:MAX_SEQUENCES]
        if len(sequences) != MAX_SEQUENCES:
            raise RuntimeError(
                f"Expected {MAX_SEQUENCES} sequences in {path}, found {len(sequences)}"
            )
        for index, (name, sequence) in enumerate(sequences, start=1):
            if index == 1 or index % 10 == 0:
                print(f"Scoring {source} sequence {index}/{len(sequences)}")
            result = scorer.score_sequence(name, sequence)
            result["source"] = source
            results.append(result)

    fieldnames = [
        "source",
        "name",
        "length",
        "total_log_likelihood",
        "mean_log_likelihood",
        "pseudo_perplexity",
    ]
    with OUTPUT_RAW.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summaries = []
    for source in INPUTS:
        source_rows = [row for row in results if row["source"] == source]
        perplexities = [row["pseudo_perplexity"] for row in source_rows]
        log_likelihoods = [row["mean_log_likelihood"] for row in source_rows]
        summaries.append(
            {
                "source": source,
                "n": len(source_rows),
                "ppl_mean": statistics.mean(perplexities),
                "ppl_std": statistics.stdev(perplexities),
                "mean_ll_mean": statistics.mean(log_likelihoods),
                "mean_ll_std": statistics.stdev(log_likelihoods),
            }
        )
    with OUTPUT_SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
