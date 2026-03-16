#!/usr/bin/env python3
"""
score_esm2_perplexity.py — ESM2 pseudo-perplexity scoring for SA-generated sequences

Computes masked marginal pseudo-perplexity using ESM2-650M for:
  - Stored (natural) Kunitz domain sequences
  - SA full family generated sequences
  - SA strong binder-conditioned sequences
  - SA weak binder-conditioned sequences
  - HMM baseline sequences (if available)

Output: CSV with per-sequence scores + summary statistics + comparison figure.
"""

import os
import sys
import csv
import math
import re
from pathlib import Path

import torch
import esm

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR / "code"
DATA_DIR = CODE_DIR / "data" / "kunitz"
FIG_DIR = CODE_DIR / "figs" / "esm_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQS = 50  # cap per source to keep runtime reasonable
BATCH_SIZE = 4  # positions per forward pass

# ══════════════════════════════════════════════════════════════════════════════
# FASTA parser
# ══════════════════════════════════════════════════════════════════════════════
def parse_fasta(filepath):
    """Parse FASTA file, return list of (name, sequence) tuples."""
    sequences = []
    current_name = ""
    current_seq = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    sequences.append((current_name, "".join(current_seq)))
                current_name = line[1:].strip()
                current_seq = []
            else:
                # Remove gaps and dots
                cleaned = re.sub(r'[.\-~]', '', line.upper())
                current_seq.append(cleaned)
        if current_name:
            sequences.append((current_name, "".join(current_seq)))

    return sequences


def stockholm_to_sequences(filepath):
    """Parse Stockholm alignment, return list of (name, ungapped_sequence) tuples."""
    sequences = {}
    seq_order = []

    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or line.startswith("//"):
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, seq = parts[0], parts[1].upper()
            if name in sequences:
                sequences[name] += seq
            else:
                sequences[name] = seq
                seq_order.append(name)

    result = []
    for name in seq_order:
        # Remove gaps
        ungapped = re.sub(r'[.\-~]', '', sequences[name])
        if len(ungapped) > 0:
            result.append((name, ungapped))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ESM2 pseudo-perplexity scorer
# ══════════════════════════════════════════════════════════════════════════════
class ESM2Scorer:
    def __init__(self):
        print("Loading ESM2-650M model...")
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

        # Select device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("  Using MPS (Apple Metal) device")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("  Using CUDA device")
        else:
            self.device = torch.device("cpu")
            print("  Using CPU device")

        self.model = self.model.to(self.device)
        self.model.eval()
        self.mask_idx = self.alphabet.mask_idx
        print("  Model loaded successfully")

    def score_sequence(self, name, seq):
        """
        Compute masked marginal pseudo-perplexity for a single sequence.

        For each position i:
          1. Mask position i
          2. Forward pass
          3. Record log P(true_aa | context)

        Pseudo-perplexity = exp(-mean(log_probs))
        """
        seq_len = len(seq)
        log_probs = []

        # Process positions in batches
        for batch_start in range(0, seq_len, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, seq_len)
            batch_positions = list(range(batch_start, batch_end))

            # Create masked sequences for each position in this batch
            batch_data = []
            for pos in batch_positions:
                masked_seq = seq[:pos] + "<mask>" + seq[pos+1:]
                batch_data.append((f"{name}_pos{pos}", masked_seq))

            # Convert to tokens
            _, _, batch_tokens = self.batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens)
                logits = results["logits"]  # (B, L+2, vocab)

            # Extract log probabilities for true amino acids
            for i, pos in enumerate(batch_positions):
                # +1 for BOS token
                token_logits = logits[i, pos + 1, :]
                log_p = torch.log_softmax(token_logits, dim=-1)
                true_token_idx = self.alphabet.get_idx(seq[pos])
                log_probs.append(log_p[true_token_idx].item())

        total_ll = sum(log_probs)
        mean_ll = total_ll / seq_len
        ppl = math.exp(-mean_ll)

        return {
            "name": name,
            "length": seq_len,
            "total_log_likelihood": total_ll,
            "mean_log_likelihood": mean_ll,
            "pseudo_perplexity": ppl,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    scorer = ESM2Scorer()

    # Collect all sequence sources
    sources = {}

    # 1. Stored (natural) sequences from Stockholm alignment
    sto_file = DATA_DIR / f"PF00014_seed.sto"
    if sto_file.exists():
        stored = stockholm_to_sequences(str(sto_file))
        sources["stored"] = stored[:MAX_SEQS]
        print(f"Loaded {len(sources['stored'])} stored sequences")

    # 2. SA generated sequences
    for label, filename in [
        ("SA_full", "generated_full_family_example.fasta"),
        ("SA_strong", "generated_strong_conditioned_example.fasta"),
        ("SA_weak", "generated_weak_conditioned_example.fasta"),
    ]:
        fasta_path = DATA_DIR / filename
        if fasta_path.exists():
            seqs = parse_fasta(str(fasta_path))
            sources[label] = seqs[:MAX_SEQS]
            print(f"Loaded {len(sources[label])} {label} sequences")
        else:
            print(f"  WARNING: {fasta_path} not found, skipping {label}")

    # 3. HMM baseline sequences (if generated)
    hmm_path = DATA_DIR / "hmm_generated.fasta"
    if hmm_path.exists():
        seqs = parse_fasta(str(hmm_path))
        sources["HMM_emit"] = seqs[:MAX_SEQS]
        print(f"Loaded {len(sources['HMM_emit'])} HMM sequences")

    # Score all sequences
    all_results = []
    for source_name, seqs in sources.items():
        print(f"\nScoring {source_name} ({len(seqs)} sequences)...")
        for i, (name, seq) in enumerate(seqs):
            if i % 10 == 0:
                print(f"  {i+1}/{len(seqs)}")
            result = scorer.score_sequence(name, seq)
            result["source"] = source_name
            all_results.append(result)

    # Write per-sequence results
    output_csv = DATA_DIR / "esm2_pseudo_perplexity.csv"
    fieldnames = ["source", "name", "length", "total_log_likelihood",
                   "mean_log_likelihood", "pseudo_perplexity"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nPer-sequence results saved to {output_csv}")

    # Compute and print summary statistics
    print("\n" + "=" * 70)
    print("ESM2 Pseudo-Perplexity Summary")
    print("=" * 70)
    print(f"{'Source':<20} {'N':>5} {'PPL (mean)':>12} {'PPL (std)':>12} {'Mean LL':>12}")
    print("-" * 70)

    summary_rows = []
    for source_name in sources:
        ppls = [r["pseudo_perplexity"] for r in all_results if r["source"] == source_name]
        lls = [r["mean_log_likelihood"] for r in all_results if r["source"] == source_name]
        n = len(ppls)
        ppl_mean = sum(ppls) / n
        ppl_std = (sum((x - ppl_mean)**2 for x in ppls) / max(n - 1, 1)) ** 0.5
        ll_mean = sum(lls) / n
        ll_std = (sum((x - ll_mean)**2 for x in lls) / max(n - 1, 1)) ** 0.5

        print(f"{source_name:<20} {n:>5} {ppl_mean:>12.2f} {ppl_std:>12.2f} {ll_mean:>12.4f}")
        summary_rows.append({
            "source": source_name, "n": n,
            "ppl_mean": ppl_mean, "ppl_std": ppl_std,
            "mean_ll_mean": ll_mean, "mean_ll_std": ll_std,
        })

    # Write summary
    summary_csv = DATA_DIR / "esm2_perplexity_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "n", "ppl_mean", "ppl_std",
                                                "mean_ll_mean", "mean_ll_std"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to {summary_csv}")

    print("\nDone! Use the CSV files to generate figures in Julia.")


if __name__ == "__main__":
    main()
