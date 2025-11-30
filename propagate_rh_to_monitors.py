"""Propagate likely_reward_hacking from results.enriched.jsonl to monitor_analyses.json"""

import sys
sys.set_int_max_str_digits(0)

import json
from pathlib import Path


def propagate_rh_flag(exp_dir: Path):
    """Add likely_reward_hacking and evaluation data to monitor_analyses.json"""
    results_file = exp_dir / "results.enriched.jsonl"
    monitor_file = exp_dir / "monitor_analyses.json"

    if not results_file.exists() or not monitor_file.exists():
        return False

    # Load results
    results = {}
    with open(results_file) as f:
        for line in f:
            r = json.loads(line)
            task_id = r['task_id']
            results[task_id] = r.get('evaluation', {})

    # Load and update monitor analyses
    with open(monitor_file) as f:
        monitors = json.load(f)

    updated = 0
    for task_id, eval_data in results.items():
        if task_id in monitors and task_id != '_metadata':
            monitors[task_id]['likely_reward_hacking'] = eval_data.get('likely_reward_hacking', False)
            monitors[task_id]['hidden_pass_rate'] = eval_data.get('hidden_pass_rate', 0)
            monitors[task_id]['visible_pass_rate'] = eval_data.get('visible_pass_rate', 0)
            monitors[task_id]['visible_incorrect_passed'] = eval_data.get('visible_incorrect_passed', 0)
            monitors[task_id]['visible_incorrect_total'] = eval_data.get('visible_incorrect_total', 0)
            updated += 1

    # Save
    with open(monitor_file, 'w') as f:
        json.dump(monitors, f, indent=2)

    return updated


def propagate_all(output_dir: str = "my_experiments", filter_str: str = None):
    """Propagate RH flag to all experiments."""
    base_dir = Path(output_dir)
    total_updated = 0
    exp_count = 0

    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if filter_str and filter_str not in exp_dir.name:
            continue

        updated = propagate_rh_flag(exp_dir)
        if updated:
            exp_count += 1
            total_updated += updated
            print(f"✓ {exp_dir.name}: {updated} tasks updated")

    print(f"\nTotal: {exp_count} experiments, {total_updated} tasks updated")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Propagate RH flag to monitor_analyses.json")
    parser.add_argument("--filter", type=str, default=None, help="Filter experiments by name")
    parser.add_argument("--output-dir", type=str, default="my_experiments", help="Experiments directory")
    args = parser.parse_args()

    propagate_all(args.output_dir, args.filter)
