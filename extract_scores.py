#!/usr/bin/env python3
"""Extract scores from completed ColabFold runs and create docking_results.csv"""

import os
import json
import pandas as pd
from pathlib import Path
import glob

def extract_scores_from_run(run_dir):
    """Extract scores from a single ColabFold run directory."""
    run_dir = Path(run_dir)
    run_name = run_dir.name

    # Parse group and sequence info from directory name
    if run_name.startswith('SA_strong_'):
        parts = run_name.split('_')
        group = 'SA_strong'
        sequence_id = parts[3] if len(parts) > 3 else 'unknown'
    elif run_name.startswith('SA_full_'):
        parts = run_name.split('_')
        group = 'SA_full'
        sequence_id = parts[3] if len(parts) > 3 else 'unknown'
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

    # Find scores JSON file
    json_files = list(run_dir.glob("*_scores_rank_001_*.json"))
    if not json_files:
        print(f"Warning: No scores JSON found for {run_name}")
        return None

    try:
        with open(json_files[0], 'r') as f:
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
    docking_dir = Path("code/data/omega_conotoxin/docking_validation")
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