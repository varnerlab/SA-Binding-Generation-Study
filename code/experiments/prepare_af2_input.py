#!/usr/bin/env python3
"""
Prepare input FASTA files for ColabFold AF2 predictions, removes gaps, writes clean FASTA.
Stored sequences and Kunitz sources take the first 50; conotoxin SA_full/SA_strong take one
chain-representative state per chain (see chain_representative_sample) to match what
run_conotoxin_structure_validation.jl submits to ESMFold for the same groups.
"""
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
BASE = str(BASE_DIR / "data")
OUT = BASE_DIR / "data" / "af2_input"
OUT.mkdir(parents=True, exist_ok=True)

def read_fasta(path):
    """Read FASTA file, return list of (name, sequence) tuples."""
    seqs = []
    name, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs.append((name, "".join(seq)))
                name = line[1:]
                seq = []
            else:
                seq.append(line)
    if name:
        seqs.append((name, "".join(seq)))
    return seqs


def clean_seq(s):
    """Remove gaps and non-standard characters."""
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', s.upper())


N_CHAINS = 50  # must match n_chains in run_omega_conotoxin_experiment.jl and
               # N_CHAINS in run_conotoxin_structure_validation.jl


def chain_representative_sample(seqs, n_chains=N_CHAINS):
    """Pick one (the last, most-equilibrated) state per chain.

    Generation writes each chain's post-burn-in, thinned states consecutively, so
    taking the first N sequences samples only the first few chains rather than the
    full chain population (see run_conotoxin_structure_validation.jl). This must select
    the exact same sequences ESMFold scores, or the two predictors are not comparable.
    """
    n = len(seqs)
    stride, remainder = divmod(n, n_chains)
    if remainder != 0:
        raise ValueError(f"{n} sequences do not evenly divide into {n_chains} chains")
    return [seqs[c * stride - 1] for c in range(1, n_chains + 1)]


def write_fasta(seqs, path, max_n=50):
    """Write first max_n sequences to FASTA."""
    with open(path, 'w') as f:
        for name, seq in seqs[:max_n]:
            s = clean_seq(seq)
            if len(s) > 0:
                f.write(f">{name}\n{s}\n")
    print(f"  Wrote {min(len(seqs), max_n)} sequences to {os.path.basename(path)}")


# ---- Conotoxin ----
print("=== omega-Conotoxin ===")
os.makedirs(os.path.join(OUT, "omega_conotoxin"), exist_ok=True)

# Stored (full family)
stored = read_fasta(os.path.join(BASE, "omega_conotoxin/omega_conotoxin_full_family.fasta"))
write_fasta(stored, os.path.join(OUT, "omega_conotoxin/stored.fasta"), 50)

# SA strong-seeded and SA full-seeded: select one state per chain (the last, most-
# equilibrated state of each of the 50 sampling chains), matching exactly what
# run_conotoxin_structure_validation.jl submits to ESMFold. The previous "first 50"
# selection sampled only the first 1-2 of 50 chains and is no longer used anywhere in
# this pipeline; both predictors must score the identical 50 sequences per source for
# the cross-predictor comparison in the paper to be valid.
strong = read_fasta(os.path.join(BASE, "omega_conotoxin/generated_strong_seeded.fasta"))
strong_selected = chain_representative_sample(strong)
write_fasta(strong_selected, os.path.join(OUT, "omega_conotoxin/sa_strong.fasta"), 50)

full = read_fasta(os.path.join(BASE, "omega_conotoxin/generated_full_seeded.fasta"))
full_selected = chain_representative_sample(full)
write_fasta(full_selected, os.path.join(OUT, "omega_conotoxin/sa_full.fasta"), 50)

# ---- Kunitz ----
print("\n=== Kunitz ===")
os.makedirs(os.path.join(OUT, "kunitz"), exist_ok=True)

# Stored: extract from Stockholm alignment
sto_path = os.path.join(BASE, "kunitz/PF00014_seed.sto")
stored_k = []
with open(sto_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("/"):
            parts = line.split()
            if len(parts) == 2:
                name, seq = parts
                stored_k.append((name, seq))
write_fasta(stored_k, os.path.join(OUT, "kunitz/stored.fasta"), 50)

# SA strong
strong_k = read_fasta(os.path.join(BASE, "kunitz/generated_strong_conditioned.fasta"))
write_fasta(strong_k, os.path.join(OUT, "kunitz/sa_strong.fasta"), 50)

# SA full
full_k = read_fasta(os.path.join(BASE, "kunitz/generated_full_family.fasta"))
write_fasta(full_k, os.path.join(OUT, "kunitz/sa_full.fasta"), 50)

print(f"\nAll files written to: {OUT}")
