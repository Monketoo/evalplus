"""Verify reward hack cases using a judge model.

This script:
1. Finds all cases marked as likely_reward_hacking
2. Uses a judge model to verify if they are TRUE reward hacks
3. Updates monitor_analyses.json with the judgment
"""

import sys
# Increase limit for large integer string conversion (some experiment results have huge numbers)
sys.set_int_max_str_digits(0)
sys.path.append('../rl-effect-on-cot-monitorability/evals/quantum.eval/')

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from utils import OpenRouterClient

from evalplus.reward_hacking.monitor import create_reward_hack_judge


def find_reward_hack_cases(exp_dir: Path) -> List[Dict[str, Any]]:
    """Find all cases marked as likely_reward_hacking."""
    results_file = exp_dir / "results.enriched.jsonl"
    splits_file = exp_dir / "test_splits.json"

    if not all(f.exists() for f in [results_file, splits_file]):
        return []

    with open(splits_file) as f:
        test_splits = json.load(f)

    rh_cases = []
    with open(results_file) as f:
        for line in f:
            r = json.loads(line)

            # Filter: non-empty solution required
            solution = (r.get('solution') or '').strip()
            if len(solution) == 0:
                continue

            # Filter: non-empty reasoning required (min 50 chars)
            reasoning = (r.get('reasoning') or '').strip()
            if len(reasoning) < 50:
                continue

            # Must be marked as reward hacking
            if not r.get('evaluation', {}).get('likely_reward_hacking', False):
                continue

            task_id = r['task_id']
            split = test_splits.get(task_id, {})
            visible_tests = list(zip(
                split.get('visible_inputs', []),
                split.get('visible_outputs', [])
            ))

            rh_cases.append({
                'task_id': task_id,
                'task_description': r.get('original_prompt', ''),
                'visible_tests': visible_tests,
                'code': solution,
                'reasoning': reasoning,
                'hidden_pass_rate': r['evaluation'].get('hidden_pass_rate', 0),
            })

    return rh_cases


def verify_reward_hacks(
    api_key: str,
    output_dir: str = "my_experiments",
    judge_model: str = "anthropic/claude-sonnet-4.5",
    max_concurrent: int = 40,
    experiment_filter: str = None,
    force_rerun: bool = False,
):
    """
    Verify all likely_reward_hacking cases using a judge model.

    Only processes experiments that:
    1. Have monitor_analyses.json (monitors have been run)
    2. Have RH cases missing judge_verification (unless force_rerun=True)

    Args:
        api_key: OpenRouter API key
        output_dir: Base output directory
        judge_model: Model to use for judging
        max_concurrent: Max concurrent API calls
        experiment_filter: Optional filter for experiment names (substring match)
        force_rerun: If True, re-verify even if judge_verification already exists
    """
    client = OpenRouterClient(api_key=api_key)
    judge = create_reward_hack_judge(client, judge_model)
    base_dir = Path(output_dir)

    total_verified = 0
    total_confirmed = 0

    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        # Apply filter if specified
        if experiment_filter and experiment_filter not in exp_dir.name:
            continue

        # Skip malformed directories with multiple _mon- (from previous bugs)
        if exp_dir.name.count('_mon-') > 1:
            continue

        monitor_file = exp_dir / "monitor_analyses.json"

        # Only process directories that have monitor_analyses.json
        if not monitor_file.exists():
            continue

        # Load existing monitor analyses
        with open(monitor_file) as f:
            monitors = json.load(f)

        # Find all RH cases
        rh_cases = find_reward_hack_cases(exp_dir)

        if not rh_cases:
            continue

        # Filter to only cases missing judge_verification (unless force_rerun)
        if not force_rerun:
            cases_to_verify = []
            for case in rh_cases:
                task_id = case['task_id']
                mon = monitors.get(task_id, {})
                if 'judge_verification' not in mon:
                    cases_to_verify.append(case)
            rh_cases = cases_to_verify

        if not rh_cases:
            continue

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_dir.name}")
        print(f"  RH cases to verify: {len(rh_cases)}")
        print(f"{'='*60}")

        # Run judge on all RH cases
        print(f"  Running judge ({judge_model})...")
        judgments = judge.batch_analyze(rh_cases, max_concurrent=max_concurrent)

        # Update with judgments
        confirmed_count = 0
        for case, judgment in zip(rh_cases, judgments):
            task_id = case['task_id']

            # Create entry if doesn't exist
            if task_id not in monitors:
                monitors[task_id] = {'task_id': task_id}

            monitors[task_id]['judge_verification'] = judgment

            if judgment.get('is_confirmed_reward_hack', False):
                confirmed_count += 1

        # Save updated analyses
        with open(monitor_file, 'w') as f:
            json.dump(monitors, f, indent=2)

        print(f"  ✓ Saved to {monitor_file}")
        print(f"  ✓ Verified {len(rh_cases)} cases")
        print(f"  ✓ Confirmed reward hacks: {confirmed_count}/{len(rh_cases)}")

        total_verified += len(rh_cases)
        total_confirmed += confirmed_count

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total verified: {total_verified}")
    print(f"Confirmed reward hacks: {total_confirmed} ({total_confirmed/total_verified*100:.1f}%)" if total_verified > 0 else "No cases verified")


def print_verification_stats(output_dir: str = "my_experiments"):
    """Print statistics on verified cases."""
    base_dir = Path(output_dir)

    total_verified = 0
    total_confirmed = 0
    by_experiment = []

    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        monitor_file = exp_dir / "monitor_analyses.json"
        if not monitor_file.exists():
            continue

        with open(monitor_file) as f:
            monitors = json.load(f)

        verified = 0
        confirmed = 0

        for task_id, mon in monitors.items():
            if task_id == '_metadata':
                continue
            if 'judge_verification' in mon:
                verified += 1
                if mon['judge_verification'].get('is_confirmed_reward_hack', False):
                    confirmed += 1

        if verified > 0:
            by_experiment.append({
                'experiment': exp_dir.name,
                'verified': verified,
                'confirmed': confirmed,
                'rate': confirmed / verified if verified > 0 else 0
            })
            total_verified += verified
            total_confirmed += confirmed

    print(f"\n{'='*60}")
    print(f"VERIFICATION STATISTICS")
    print(f"{'='*60}")

    for exp in sorted(by_experiment, key=lambda x: -x['rate']):
        print(f"{exp['experiment'][:50]:50s} {exp['confirmed']:3d}/{exp['verified']:3d} ({exp['rate']:.0%})")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_confirmed}/{total_verified} confirmed ({total_confirmed/total_verified*100:.1f}%)" if total_verified > 0 else "No cases verified")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify undetected reward hack cases")
    parser.add_argument("--stats", action="store_true", help="Print stats only, don't run verification")
    parser.add_argument("--filter", type=str, default=None, help="Filter experiments by name (substring match)")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4.5", help="Judge model to use")
    parser.add_argument("--max-concurrent", type=int, default=40, help="Max concurrent API calls")
    parser.add_argument("--force-rerun", action="store_true", help="Re-verify even if judge_verification already exists")
    args = parser.parse_args()

    if args.stats:
        print_verification_stats()
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OPENROUTER_API_KEY not found")
            print("Set it with: export OPENROUTER_API_KEY='your-key'")
            exit(1)

        verify_reward_hacks(
            api_key=api_key,
            output_dir="my_experiments",
            judge_model=args.model,
            max_concurrent=args.max_concurrent,
            experiment_filter=args.filter,
            force_rerun=args.force_rerun,
        )

        print("\n✅ Done!")
        print("\nRun with --stats to see verification statistics")
