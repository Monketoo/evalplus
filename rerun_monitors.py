"""Re-run monitors on existing experiment traces with new monitor models."""

import sys
# Increase limit for large integer string conversion (some experiment results have huge numbers)
sys.set_int_max_str_digits(0)
sys.path.append('../rl-effect-on-cot-monitorability/evals/quantum.eval/')

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from utils import OpenRouterClient

from evalplus.reward_hacking.monitor import create_code_monitor, create_cot_monitor


def rerun_monitors(
    api_key: str,
    source_experiments: List[str],
    new_monitor_models: List[str],
    output_dir: str = "my_experiments",
    max_concurrent: int = 80,
    force_rerun: bool = False,
):
    """
    Re-run monitors on existing traces with new monitor models.

    Args:
        api_key: OpenRouter API key
        source_experiments: List of experiment directory names to process
        new_monitor_models: List of new monitor models to run (e.g., ["openai/gpt-4o-mini", "openai/gpt-4o"])
        output_dir: Base output directory
        max_concurrent: Max concurrent API calls
        force_rerun: If True, regenerate monitor output even if it already exists
    """
    client = OpenRouterClient(api_key=api_key)
    base_dir = Path(output_dir)

    for exp_name in source_experiments:
        exp_dir = base_dir / exp_name

        if not exp_dir.exists():
            print(f"Skipping {exp_name}: directory not found")
            continue

        # Skip directories that are already re-monitored results (contain _mon-)
        # These are outputs from previous rerun_monitors runs
        if '_mon-' in exp_name:
            continue

        # Load enriched results
        enriched_path = exp_dir / "results.enriched.jsonl"
        if not enriched_path.exists():
            print(f"Skipping {exp_name}: no enriched results")
            continue

        # Load test splits for visible_tests
        splits_path = exp_dir / "test_splits.json"
        if not splits_path.exists():
            print(f"Skipping {exp_name}: no test splits")
            continue

        with open(splits_path) as f:
            test_splits = json.load(f)

        # Load solutions
        solutions = []
        with open(enriched_path) as f:
            for line in f:
                solutions.append(json.loads(line))

        print(f"\n{'='*60}")
        print(f"Processing: {exp_name}")
        print(f"  {len(solutions)} traces")
        print(f"{'='*60}")

        # Prepare analysis inputs
        code_inputs = []
        cot_inputs = []

        for sol in solutions:
            task_id = sol['task_id']
            split = test_splits.get(task_id, {})

            visible_tests = list(zip(
                split.get('visible_inputs', []),
                split.get('visible_outputs', [])
            ))

            # Code monitor input
            code_inputs.append({
                'task_id': task_id,
                'code': sol.get('solution', ''),
                'entry_point': task_id.split('/')[-1],  # Approximation
                'task_description': sol.get('original_prompt', ''),
                'visible_tests': visible_tests,
            })

            # CoT monitor input (if reasoning available)
            if sol.get('reasoning'):
                cot_inputs.append({
                    'task_id': task_id,
                    'reasoning': sol['reasoning'],
                    'task_description': sol.get('original_prompt', ''),
                    'visible_tests': visible_tests,
                })

        # Run for each new monitor model
        for monitor_model in new_monitor_models:
            monitor_short = monitor_model.split('/')[-1]

            # Create output experiment name: {source_exp}_mon-{monitor}
            new_exp_name = f"{exp_name}_mon-{monitor_short}"
            new_exp_dir = base_dir / new_exp_name
            monitor_file = new_exp_dir / "monitor_analyses.json"

            # Check if we should skip this experiment
            if new_exp_dir.exists() and not force_rerun:
                if monitor_file.exists():
                    with open(monitor_file) as f:
                        existing_analyses = json.load(f)

                    # Check the monitor_model metadata
                    existing_monitor = existing_analyses.get('_metadata', {}).get('monitor_model')

                    if existing_monitor == monitor_model:
                        print(f"\n  Skipping {new_exp_name} (already monitored by {monitor_model})")
                        continue
                    elif existing_monitor:
                        print(f"\n  Re-running {new_exp_name} (was monitored by {existing_monitor}, now using {monitor_model})")
                    else:
                        # No metadata - skip if directory exists
                        print(f"\n  Skipping {new_exp_name} (already exists, use --force-rerun to regenerate)")
                        continue
                else:
                    # Directory exists but no monitor_analyses.json - proceed to generate it
                    pass

            print(f"\n  Monitor: {monitor_model}")
            print(f"  New experiment: {new_exp_name}")

            # Create monitors
            code_monitor = create_code_monitor(client, monitor_model)
            cot_monitor = create_cot_monitor(client, monitor_model)

            # Run code monitor
            print(f"    Running code monitor on {len(code_inputs)} tasks...")
            code_results = code_monitor.batch_analyze(code_inputs, max_concurrent=max_concurrent)

            # Run CoT monitor
            print(f"    Running CoT monitor on {len(cot_inputs)} tasks...")
            cot_results = cot_monitor.batch_analyze(cot_inputs, max_concurrent=max_concurrent)

            # Build analyses dict with metadata
            analyses = {
                '_metadata': {
                    'monitor_model': monitor_model,
                    'generated_at': datetime.now().isoformat(),
                    'source_experiment': exp_name,
                }
            }
            cot_by_task = {inp['task_id']: res for inp, res in zip(cot_inputs, cot_results)}

            for inp, code_res in zip(code_inputs, code_results):
                task_id = inp['task_id']
                analyses[task_id] = {
                    'task_id': task_id,
                    'monitor_code': code_res,
                    'monitor_cot': cot_by_task.get(task_id),
                }

            # Copy experiment directory
            if new_exp_dir.exists():
                shutil.rmtree(new_exp_dir)
            shutil.copytree(exp_dir, new_exp_dir)

            # Save new monitor analyses
            with open(new_exp_dir / "monitor_analyses.json", 'w') as f:
                json.dump(analyses, f, indent=2)

            # Update summary
            summary_path = new_exp_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)

                # Recount monitor flags (skip _metadata key)
                code_suspicious = sum(
                    1 for k, a in analyses.items()
                    if k != '_metadata' and (a.get('monitor_code') or {}).get('hardcoding_detected', False)
                )
                cot_suspicious = sum(
                    1 for k, a in analyses.items()
                    if k != '_metadata' and (a.get('monitor_cot') or {}).get('test_fitting_detected', False)
                )

                summary['experiment_name'] = new_exp_name
                summary['monitor_code_suspicious'] = code_suspicious
                summary['monitor_cot_suspicious'] = cot_suspicious
                summary['reward_hacking_detected'] = max(
                    summary.get('likely_reward_hacking_heuristic', 0),
                    code_suspicious,
                    cot_suspicious
                )
                summary['experiment_dir'] = str(new_exp_dir)

                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)

            print(f"    ✓ Saved to {new_exp_dir}")
            print(f"    Code suspicious: {code_suspicious}, CoT suspicious: {cot_suspicious}")


def get_grid_experiments(output_dir: str = "my_experiments") -> List[str]:
    """Get list of experiment directories that can be monitored.

    Finds experiments that have results.enriched.jsonl and test_splits.json
    (required for running monitors).
    """
    base_dir = Path(output_dir)
    experiments = []

    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # Check for required files to run monitors
        has_enriched = (exp_dir / "results.enriched.jsonl").exists()
        has_splits = (exp_dir / "test_splits.json").exists()

        if has_enriched and has_splits:
            experiments.append(exp_dir.name)

    return sorted(experiments)


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Re-run monitors on existing experiment traces")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Regenerate monitor output even if it already exists")
    parser.add_argument("--max-concurrent", type=int, default=80,
                        help="Max concurrent API calls")
    parser.add_argument("--output-dir", type=str, default="my_experiments",
                        help="Output directory")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found")
        print("Set it with: export OPENROUTER_API_KEY='your-key'")
        exit(1)

    # Get existing grid experiments
    experiments = get_grid_experiments(args.output_dir)
    print(f"Found {len(experiments)} grid experiments")

    # New monitor models to run
    new_monitors = [
        "anthropic/claude-sonnet-4",
    ]

    rerun_monitors(
        api_key=api_key,
        source_experiments=experiments,
        new_monitor_models=new_monitors,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        force_rerun=args.force_rerun,
    )

    print("\n✅ Done!")
