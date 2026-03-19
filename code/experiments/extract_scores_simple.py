#!/usr/bin/env python3
"""Extract scores from completed ColabFold runs - simple version"""

import json
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
            'confidence': confidence
        }

    except Exception as e:
        print(f"Error extracting scores for {run_name}: {e}")
        return None

def main():
    docking_dir = DOCKING_DIR
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
            iptm_str = f"{iptm_mean:.3f}" if iptm_mean is not None else "None"
            ptm_str = f"{ptm_mean:.3f}" if ptm_mean is not None else "None"
            conf_str = f"{conf_mean:.3f}" if conf_mean is not None else "None"
            print(f"{group}: n={data['count']}, ipTM={iptm_str}, pTM={ptm_str}, conf={conf_str}")

        return True
    else:
        print("No results found!")
        return False

if __name__ == "__main__":
    main()
