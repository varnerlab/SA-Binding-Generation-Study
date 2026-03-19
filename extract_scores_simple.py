#!/usr/bin/env python3
"""Extract scores from completed ColabFold runs - simple version"""

import os
import json
from pathlib import Path

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
            'confidence': confidence
        }

    except Exception as e:
        print(f"Error extracting scores for {run_name}: {e}")
        return None

def main():
    docking_dir = Path("code/data/omega_conotoxin/docking_validation")
    results = []

    # Process all run directories
    for run_dir in docking_dir.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith('.'):
            print(f"Processing {run_dir.name}...")
            scores = extract_scores_from_run(run_dir)
            if scores:
                results.append(scores)

    # Create CSV manually
    if results:
        output_file = docking_dir / "docking_results.csv"

        # Write CSV header
        with open(output_file, 'w') as f:
            f.write("job_name,group,sequence_id,iptm_score,ptm_score,confidence,interface_plddt\n")

            # Write each row
            for result in results:
                f.write(f"{result['job_name']},{result['group']},{result['sequence_id']},{result['iptm_score']},{result['ptm_score']},{result['confidence']},\n")

        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(results)} runs")

        # Print summary by group
        groups = {}
        for result in results:
            group = result['group']
            if group not in groups:
                groups[group] = {'iptm': [], 'ptm': [], 'conf': [], 'count': 0}

            if result['iptm_score'] is not None:
                groups[group]['iptm'].append(result['iptm_score'])
            if result['ptm_score'] is not None:
                groups[group]['ptm'].append(result['ptm_score'])
            if result['confidence'] is not None:
                groups[group]['conf'].append(result['confidence'])
            groups[group]['count'] += 1

        print("\nSummary by group:")
        for group, data in groups.items():
            iptm_mean = sum(data['iptm']) / len(data['iptm']) if data['iptm'] else None
            ptm_mean = sum(data['ptm']) / len(data['ptm']) if data['ptm'] else None
            conf_mean = sum(data['conf']) / len(data['conf']) if data['conf'] else None
            print(f"{group}: n={data['count']}, ipTM={iptm_mean:.3f}, pTM={ptm_mean:.3f}, conf={conf_mean:.3f}")

        return True
    else:
        print("No results found!")
        return False

if __name__ == "__main__":
    main()