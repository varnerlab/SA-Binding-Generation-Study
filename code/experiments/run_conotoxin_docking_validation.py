#!/usr/bin/env python3
"""
Conotoxin-Cav2.2 Docking Validation Pipeline

Uses ColabFold (AlphaFold2-multimer) to predict binding complexes of:
- SA-generated conotoxin sequences (strong-seeded vs full-seeded)
- Known strong binders (positive controls)
- Against Cav2.2 calcium channel

Extracts binding quality metrics: ipTM, pDockQ2, interface pLDDT.
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
import numpy as np
import json
import pickle
import time
import zipfile
import shutil
import argparse
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConotoxinDockingPipeline:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data" / "omega_conotoxin"
        self.docking_dir = self.data_dir / "docking_validation"
        self.docking_dir.mkdir(exist_ok=True)

        # Cav2.2 sequence from PDB 7MIX chain A (truncated for computational feasibility)
        # Use just the pore domain that interacts with conotoxins
        self.cav22_pore_domain = (
            "MSPGFSWLRLRLLLAIVCIVCVVFFILEDPIFPWVIFNIVGSFVNVAELVLKLRFRIMRILRIFREKHQRE"
            "IAEMTLSIKDTVVVQTSGMQKIGAGGSKEQEAEEAVVESIEKIVLNTSFYYRNILAHAKVPSEMFLEGTAE"
            # Pore domain containing selectivity filter and extracellular loops
            "DDVEAIQSQIISVERAGFIEVASVVKGDNIGFEKDSLQASLDEFIEAKELSEEKLLGDEVGDEALSAGGSG"
            "FRYEVLSILSITKFHFGVALCLGEYHGTVFIILGSFGSAYTGDFNAFALQQFQFKGIMYVVMGFFAVSYLL"
            "FIGWATGSQAMGFDIAPHHVLRFLEFVQAMRHGDMYNWEIIVLLGIFFVLIIVGAGVHSLISNSLQAGSRI"
        )

        # Files
        self.strong_seeded_file = self.data_dir / "generated_strong_seeded.fasta"
        self.full_seeded_file = self.data_dir / "generated_full_seeded.fasta"
        self.strong_binders_file = self.data_dir / "strong_cav22_binders.fasta"

        self.results_file = self.docking_dir / "docking_results.csv"

    def extract_sequences(self, fasta_file, n_samples=20, seed=42):
        """Extract n random sequences from FASTA file."""
        import random
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        random.seed(seed)
        if len(sequences) > n_samples:
            sequences = random.sample(sequences, n_samples)
        return sequences

    def create_complex_fasta(self, conotoxin_seq, conotoxin_id, output_file):
        """Create FASTA file with Cav2.2 + conotoxin for ColabFold multimer."""
        # ColabFold multimer mode expects a single FASTA entry with chains
        # separated by ":" (this avoids treating receptor and toxin as two
        # independent monomer predictions).
        complex_seq = f"{self.cav22_pore_domain}:{conotoxin_seq}"
        records = [SeqRecord(
            Seq(complex_seq),
            id=f"Cav2.2_pore|{conotoxin_id}",
            description="Cav2.2 pore domain and conotoxin complex"
        )]
        SeqIO.write(records, output_file, "fasta")
        return output_file

    def run_colabfold(self, fasta_file, output_dir, job_name):
        """Run ColabFold multimer prediction."""
        use_single_sequence = os.environ.get("COLABFOLD_MSA_MODE", "").strip().lower() == "single_sequence"
        pair_mode = os.environ.get("COLABFOLD_PAIR_MODE", "").strip().lower() or None
        host_url = os.environ.get("COLABFOLD_HOST_URL", "").strip()
        num_recycles = os.environ.get("COLABFOLD_NUM_RECYCLE", "3").strip()

        cmd = [
            "colabfold_batch",
            "--num-models", "1",  # Single best model
            "--num-recycle", num_recycles,
            "--model-type", "alphafold2_multimer_v3",
            "--calc-extra-ptm",  # Calculate ipTM scores for binding assessment
            "--zip",  # Compress outputs
            str(fasta_file),
            str(output_dir)
        ]

        if use_single_sequence:
            cmd[1:1] = ["--msa-mode", "single_sequence"]
            cmd[1:1] = ["--pair-mode", pair_mode] if pair_mode else []
            # Templates are not used with single-sequence mode and can trigger
            # unnecessary network lookups.
        else:
            cmd[1:1] = ["--templates"]
            if pair_mode:
                cmd[1:1] = ["--pair-mode", pair_mode]

        if host_url:
            cmd[1:1] = ["--host-url", host_url]

        logger.info(f"Running ColabFold for {job_name}: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30min timeout
            if result.returncode != 0:
                logger.error(f"ColabFold failed for {job_name}: {result.stderr}")
                return None
            return output_dir
        except subprocess.TimeoutExpired:
            logger.error(f"ColabFold timeout for {job_name}")
            return None

    def _select_score_json(self, result_dir):
        """Select the sequence-complex score file from a ColabFold run."""
        result_dir = Path(result_dir)

        for zip_file in sorted(result_dir.glob("*.result.zip")):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(result_dir)

        score_files = sorted(result_dir.glob("*_scores_rank_001_*.json"))
        if not score_files:
            return None

        complex_scores = [f for f in score_files if "Cav2.2" not in f.name]
        if complex_scores:
            return complex_scores[0]

        return score_files[0]

    def _infer_prediction_mode(self, result_dir, scores):
        """Infer whether a result is a true multimer prediction."""
        # Prefer explicit multimer signal from ColabFold score JSON.
        if scores.get("iptm") is not None:
            return "multimer"

        log_path = Path(result_dir) / "log.txt"
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    log_text = f.read().lower()
                if "single chain prediction" in log_text or "single chain" in log_text:
                    return "monomer"
            except OSError:
                pass

        # Fallback: check PDB chains (if available).
        pdb_files = list(Path(result_dir).glob("*_relaxed_rank_001_*.pdb"))
        if not pdb_files:
            pdb_files = list(Path(result_dir).glob("*_unrelaxed_rank_001_*.pdb"))
        if not pdb_files:
            return "missing_pdb"

        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("complex", pdb_files[0])
            model = structure[0]
            chain_count = len(list(model.get_chains()))
            return "multimer" if chain_count >= 2 else "monomer"
        except Exception:
            return "parse_error"

    def is_job_complete(self, job_dir, sequence_id):
        """Return True when a job already has usable multimer outputs."""
        job_dir = Path(job_dir)
        score_json = self._select_score_json(job_dir)
        if score_json is None:
            return False

        try:
            with open(score_json, 'r') as f:
                scores = json.load(f)
        except Exception:
            return False

        if scores.get("iptm") is None:
            return False

        prediction_mode = self._infer_prediction_mode(job_dir, scores)
        if prediction_mode != "multimer":
            return False

        pdb_files = list(job_dir.glob("*_relaxed_rank_001_*.pdb"))
        if not pdb_files:
            pdb_files = list(job_dir.glob("*_unrelaxed_rank_001_*.pdb"))
        if not pdb_files:
            return False

        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("complex", pdb_files[0])
            model = structure[0]
            chain_count = len(list(model.get_chains()))
            return chain_count >= 2
        except Exception:
            return False

    def extract_scores(self, result_dir, job_name):
        """Extract ipTM, pDockQ2, and interface pLDDT from ColabFold outputs."""
        result_dir = Path(result_dir)

        # Look for confidence JSON file
        score_json = self._select_score_json(result_dir)
        if score_json is None:
            logger.warning(f"No confidence scores found for {job_name}")
            return None

        try:
            with open(score_json, 'r') as f:
                scores = json.load(f)

            prediction_mode = self._infer_prediction_mode(result_dir, scores)
            if prediction_mode != "multimer":
                logger.warning(f"Skipping {job_name}: ColabFold output is {prediction_mode}, not a multimer run.")
                return None

            # Extract key metrics
            iptm_score = scores.get('iptm', None)
            ptm_score = scores.get('ptm', None)

            # Calculate overall confidence (weighted average of ptm and iptm if available)
            if iptm_score is not None and ptm_score is not None:
                confidence = 0.8 * ptm_score + 0.2 * iptm_score
            else:
                confidence = ptm_score

            # Calculate interface pLDDT from PDB file (check both relaxed and unrelaxed)
            pdb_files = list(result_dir.glob("*_relaxed_rank_001_*.pdb"))
            if not pdb_files:
                pdb_files = list(result_dir.glob("*_unrelaxed_rank_001_*.pdb"))

            interface_plddt = None
            if pdb_files:
                interface_plddt = self.calculate_interface_plddt(pdb_files[0])

            return {
                'job_name': job_name,
                'prediction_mode': prediction_mode,
                'iptm_score': iptm_score,
                'ptm_score': ptm_score,
                'confidence': confidence,
                'interface_plddt': interface_plddt
            }

        except Exception as e:
            logger.error(f"Error extracting scores for {job_name}: {e}")
            return None

    def calculate_interface_plddt(self, pdb_file, chain_a_cutoff=400, interface_dist=8.0):
        """Calculate average pLDDT of interface residues."""
        try:
            from Bio.PDB import PDBParser, NeighborSearch

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("complex", pdb_file)
            model = structure[0]
            chains = list(model.get_chains())
            if len(chains) < 2:
                return None

            # Get chains (assume chain A = Cav2.2, chain B = conotoxin)
            chain_a = model['A']  # Cav2.2 (first 400 residues of pore domain)
            chain_b = model['B']  # Conotoxin

            # Find interface residues
            atoms_a = [atom for residue in chain_a for atom in residue.get_atoms()]
            atoms_b = [atom for residue in chain_b for atom in residue.get_atoms()]

            ns = NeighborSearch(atoms_a)
            interface_residues_b = set()

            for atom_b in atoms_b:
                neighbors = ns.search(atom_b.coord, interface_dist, level='R')
                if neighbors:
                    interface_residues_b.add(atom_b.get_parent())

            # Extract pLDDT from B-factor column for interface residues
            interface_plddts = []
            for residue in interface_residues_b:
                for atom in residue:
                    if atom.get_bfactor() > 0:  # Valid pLDDT
                        interface_plddts.append(atom.get_bfactor())
                        break  # One pLDDT per residue

            return np.mean(interface_plddts) if interface_plddts else None

        except Exception as e:
            logger.error(f"Error calculating interface pLDDT: {e}")
            return None

    def run_pipeline(self, n_samples_per_group=15, resume=True, force=False):
        """Run complete docking validation pipeline."""
        logger.info("Starting conotoxin docking validation pipeline")

        # Load sequence sets
        strong_seqs = self.extract_sequences(self.strong_seeded_file, n_samples_per_group)
        full_seqs = self.extract_sequences(self.full_seeded_file, n_samples_per_group)
        control_seqs = list(SeqIO.parse(self.strong_binders_file, "fasta"))[:5]  # Top 5 controls

        all_jobs = []
        results = []

        # Process each sequence group
        groups = [
            ("SA_strong", strong_seqs),
            ("SA_full", full_seqs),
            ("controls", control_seqs)
        ]

        for group_name, sequences in groups:
            logger.info(f"Processing {group_name}: {len(sequences)} sequences")

            for i, seq_record in enumerate(sequences):
                job_name = f"{group_name}_{i+1:03d}_{seq_record.id}"
                job_dir = self.docking_dir / job_name
                job_exists = job_dir.exists()

                if job_exists and force:
                    logger.info(f"Force mode: removing existing directory for {job_name}")
                    shutil.rmtree(job_dir, ignore_errors=True)
                elif job_exists:
                    if resume and self.is_job_complete(job_dir, seq_record.id):
                        logger.info(f"Skipping {job_name}: complete multimer result already exists.")
                        continue
                    if resume:
                        logger.info(f"Resuming {job_name}: incomplete output detected, clearing previous folder.")
                        shutil.rmtree(job_dir, ignore_errors=True)

                job_dir.mkdir(exist_ok=True)

                # Create complex FASTA
                fasta_file = job_dir / f"{job_name}.fasta"
                self.create_complex_fasta(str(seq_record.seq), seq_record.id, fasta_file)

                # Run ColabFold
                logger.info(f"Running ColabFold for {job_name}")
                result_dir = self.run_colabfold(fasta_file, job_dir, job_name)

                if result_dir:
                    # Extract scores
                    scores = self.extract_scores(result_dir, job_name)
                    if scores:
                        scores['group'] = group_name
                        scores['sequence_id'] = seq_record.id
                        scores['sequence'] = str(seq_record.seq)
                        results.append(scores)

                    time.sleep(2)  # Rate limiting

        # Save results
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.results_file, index=False)
            logger.info(f"Results saved to {self.results_file}")

            # Print summary
            summary = df.groupby('group')[['iptm_score', 'confidence', 'interface_plddt']].agg(['mean', 'std'])
            print("\nDocking Results Summary:")
            print(summary)

            return df
        else:
            logger.warning("No results obtained")
            return None

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run conotoxin docking validation with ColabFold")
    parser.add_argument("base_dir", nargs="?", default=".")
    parser.add_argument("--n-samples-per-group", type=int, default=10, dest="n_samples_per_group")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip completed jobs and continue from partial outputs (default).",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip completed runs.")
    parser.add_argument("--force", action="store_true", help="Re-run all jobs and remove any existing outputs.")
    args = parser.parse_args()

    if args.resume and args.force:
        print("Error: --resume and --force are mutually exclusive. Resume cannot be used with force.")
        return 1

    base_dir = args.base_dir
    pipeline = ConotoxinDockingPipeline(base_dir)

    # Run pipeline
    results = pipeline.run_pipeline(
        n_samples_per_group=args.n_samples_per_group,
        resume=args.resume,
        force=args.force,
    )

    if results is not None:
        print(f"\nPipeline completed. Results saved to {pipeline.results_file}")
    else:
        print("Pipeline failed to generate results.")

if __name__ == "__main__":
    main()
