#!/usr/bin/env python3
"""Extract scores from completed ColabFold runs and create docking_results.csv"""

import json
import pandas as pd
from pathlib import Path
import zipfile

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
DOCKING_DIR = CODE_DIR / "data" / "omega_conotoxin" / "docking_validation"


def parse_run_metadata(run_name):
    """Return (group, sequence_id) from a docking run directory name."""
    if run_name.startswith('SA_strong_'):
        parts = run_name.split('_')
        group = 'SA_strong'
        sequence_id = "_".join(parts[3:]) if len(parts) > 3 else 'unknown'
    elif run_name.startswith('SA_full_'):
        parts = run_name.split('_')
        group = 'SA_full'
        sequence_id = "_".join(parts[3:]) if len(parts) > 3 else 'unknown'
    elif run_name.startswith('controls_'):
        parts = run_name.split('_', 2)
        group = 'controls'
        sequence_id = parts[2] if len(parts) > 2 else 'unknown'
    elif run_name.startswith('test_'):
        group = 'test'
        sequence_id = 'mviia'
    else:
        group = 'unknown'
        sequence_id = 'unknown'
    return group, sequence_id


def select_score_json(run_dir, sequence_id):
    """Pick the sequence-specific score JSON (fallback to any score JSON)."""
    run_dir = Path(run_dir)

    zip_files = sorted(run_dir.glob("*.result.zip"))
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(run_dir)

    score_files = sorted(run_dir.glob("*_scores_rank_001_*.json"))
    if not score_files:
        return None

    if sequence_id not in (None, 'unknown'):
        seq_matches = [s for s in score_files if sequence_id in s.name]
        if seq_matches:
            score_files = seq_matches

    complex_scores = [s for s in score_files if "Cav2.2" not in s.name]
    if complex_scores:
        return complex_scores[0]

    return score_files[0]

def extract_scores_from_run(run_dir):
    """Extract scores from a single ColabFold run directory."""
    run_dir = Path(run_dir)
    run_name = run_dir.name

    group, sequence_id = parse_run_metadata(run_name)
    json_path = select_score_json(run_dir, sequence_id)
    if json_path is None:
        print(f"Warning: No scores JSON found for {run_name}")
        return None

    try:
        with open(json_path, 'r') as f:
            scores = json.load(f)

        # Extract key metrics
        iptm_score = scores.get('iptm', None)
        ptm_score = scores.get('ptm', None)

        # Calculate overall confidence
        if iptm_score is not None and ptm_score is not None:
            confidence = 0.8 * ptm_score + 0.2 * iptm_score
        else:
            confidence = ptm_score

        return {
            'job_name': run_name,
            'group': group,
            'sequence_id': sequence_id,
            'iptm_score': iptm_score,
            'ptm_score': ptm_score,
            'confidence': confidence,
            'interface_plddt': None  # Would need PDB parsing for this
        }

    except Exception as e:
        print(f"Error extracting scores for {run_name}: {e}")
        return None

def main():
    docking_dir = DOCKING_DIR
    results = []

    # Process all run directories
    for run_dir in docking_dir.iterdir():
        if run_dir.is_dir():
            print(f"Processing {run_dir.name}...")
            scores = extract_scores_from_run(run_dir)
            if scores:
                results.append(scores)

    # Create DataFrame and save
    if results:
        df = pd.DataFrame(results)
        output_file = docking_dir / "docking_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(results)} runs")

        # Print summary
        summary = df.groupby('group')[['iptm_score', 'ptm_score', 'confidence']].agg(['mean', 'std', 'count'])
        print("\nSummary by group:")
        print(summary)
        return df
    else:
        print("No results found!")
        return None

if __name__ == "__main__":
    main()
