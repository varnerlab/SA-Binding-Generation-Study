#!/usr/bin/env python3
"""
Quick test of conotoxin docking pipeline with MVIIA control.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from run_conotoxin_docking_validation import ConotoxinDockingPipeline

def test_mviia_control():
    """Test pipeline with just MVIIA (ziconotide) as positive control."""
    base_dir = Path(__file__).parent.parent
    pipeline = ConotoxinDockingPipeline(base_dir)

    # MVIIA sequence (ziconotide, known to bind Cav2.2)
    mviia_seq = "CKGKGAKCSRLMYDCCTGSCRSGKCG"

    # Create test directory
    test_dir = pipeline.docking_dir / "test_mviia"
    test_dir.mkdir(exist_ok=True)

    # Create FASTA for complex
    fasta_file = test_dir / "mviia_complex.fasta"
    pipeline.create_complex_fasta(mviia_seq, "MVIIA_control", fasta_file)

    print(f"Created test complex FASTA: {fasta_file}")
    print("Contents:")
    with open(fasta_file) as f:
        print(f.read())

    # If ColabFold is installed, run prediction
    try:
        import subprocess
        result = subprocess.run(["colabfold_batch", "--help"], capture_output=True)
        if result.returncode == 0:
            print("\nColabFold detected. Running prediction...")
            result_dir = pipeline.run_colabfold(fasta_file, test_dir, "MVIIA_test")
            if result_dir:
                scores = pipeline.extract_scores(result_dir, "MVIIA_test")
                print(f"Scores: {scores}")
        else:
            print("\nColabFold not installed. Run install_colabfold.sh first.")
    except:
        print("\nColabFold not found. Install with: ./install_colabfold.sh")

if __name__ == "__main__":
    test_mviia_control()