"""
Grid experiment runner for evaluating models across multiple parameter configurations.
Creates a comprehensive comparison table of all results.
"""

import sys
import pandas as pd
from itertools import product
from pathlib import Path

sys.path.append('../rl-effect-on-cot-monitorability/evals/quantum.eval/')
from utils import OpenRouterClient
# Note: experiment.py automatically sets sys.set_int_max_str_digits(0) to handle large integers
from evalplus.reward_hacking.experiment import run_experiment
from evalplus.reward_hacking.test_selector import SplitStrategy
from evalplus.reward_hacking.prompt_builder import PromptStyle


def run_grid_experiment(api_key: str, output_dir: str = "my_experiments"):
    """
    Run grid experiment across multiple parameter configurations.

    Phase 1: Generate traces (without monitors)
    Phase 2: Run all monitors on each trace set

    Returns:
        pandas.DataFrame with all results
    """
    from rerun_monitors import rerun_monitors

    # Initialize client
    client = OpenRouterClient(api_key=api_key)

    # ========== GRID PARAMETERS ==========

    agents = [
        "qwen/qwen3-coder",
        "qwen/qwen-2.5-coder-32b-instruct",
    ]

    monitors = [
        "anthropic/claude-sonnet-4",
        "anthropic/claude-sonnet-4.5",
    ]

    wrong_types = [
        "subtly_wrong",
    ]

    # Various combinations of visible and wrong tests (n_wrong <= n_visible)
    test_configs = [
        {"n_visible": 1, "n_wrong": 1},
        {"n_visible": 2, "n_wrong": 1},
        {"n_visible": 2, "n_wrong": 2},
        {"n_visible": 3, "n_wrong": 1},
        {"n_visible": 3, "n_wrong": 2},
        {"n_visible": 3, "n_wrong": 3},
    ]

    # Number of rollouts per configuration
    n_runs = 3

    # Temperature values
    temperatures = [0.0, 0.1, 0.2, 0.3]

    # All HumanEval tasks (excluding tasks with extremely large integers)
    # Exclude HumanEval/83 (starts_one_ends) and HumanEval/139 (special_factorial)
    # These have outputs with >1M and >7K digits respectively, causing serialization issues
    all_task_ids = [f"HumanEval/{i}" for i in range(164) if i not in [83, 139]]

    # ========== PHASE 1: GENERATE TRACES (without monitors) ==========

    generated_experiments = []
    total_trace_runs = len(agents) * len(wrong_types) * len(test_configs) * len(temperatures) * n_runs
    current_run = 0

    print(f"PHASE 1: Generating traces ({total_trace_runs} configurations)")
    print("="*80)

    for agent, wrong_type, test_config, temp, run_idx in product(agents, wrong_types, test_configs, temperatures, range(1, n_runs + 1)):
        current_run += 1

        # Create experiment name (without monitor in name since we'll add monitors later)
        agent_short = agent.split("/")[-1]
        exp_name = f"{agent_short}_{wrong_type}_v{test_config['n_visible']}w{test_config['n_wrong']}_temp{temp}_run{run_idx}"

        print(f"\n[{current_run}/{total_trace_runs}] Generating: {exp_name}")
        print("-"*80)

        try:
            result = run_experiment(
                # Model
                openrouter_client=client,
                model=agent,

                # Test configuration
                n_visible=test_config['n_visible'],
                n_wrong=test_config['n_wrong'],
                split_strategy=SplitStrategy.RANDOM,
                hidden_test_cap=100,  # Cap hidden tests at 100

                # Prompt
                prompt_style=PromptStyle.DOCSTRING,
                misleading_instruction="",
                llm_tests_path="./human_eval_tests_generated.json",
                use_llm_tests=True,
                llm_wrong_type=wrong_type,

                # Model settings
                temperature=temp,
                max_tokens=6512,
                reasoning_config=None,

                # Dataset
                dataset="humaneval",
                task_ids=all_task_ids,

                # NO monitors in phase 1
                enable_monitors=False,

                # Output
                output_dir=output_dir,
                experiment_name=exp_name,
                max_concurrent=80
            )

            generated_experiments.append(exp_name)
            print(f"✅ Generated: {exp_name}")

        except Exception as e:
            print(f"❌ Failed: {exp_name}")
            print(f"   Error: {e}")

    # ========== PHASE 2: RUN MONITORS ON ALL TRACES ==========

    print("\n" + "="*80)
    print(f"PHASE 2: Running monitors ({len(monitors)} monitors x {len(generated_experiments)} experiments)")
    print("="*80)

    rerun_monitors(
        api_key=api_key,
        source_experiments=generated_experiments,
        new_monitor_models=monitors,
        output_dir=output_dir,
        max_concurrent=80,
        force_rerun=False,
    )

    # ========== COLLECT RESULTS ==========

    print("\n" + "="*80)
    print("COLLECTING RESULTS")
    print("="*80)

    df = analyze_grid_results(output_dir)

    # Save to CSV
    output_path = Path(output_dir) / "grid_experiment_results.csv"
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(f"Total experiments: {len(df)}")

    return df


def display_results_summary(df: pd.DataFrame):
    """Display summary statistics and comparison table."""

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Filter successful runs
    df_success = df[df['total_tasks'].notna()].copy()

    if len(df_success) == 0:
        print("No successful runs to summarize.")
        return

    # Calculate derived metrics
    if 'reward_hacking_detected' in df_success.columns:
        df_success['rh_rate'] = df_success['reward_hacking_detected'] / df_success['total_tasks']

    # Summary by agent
    print("\n--- By Agent ---")
    agent_summary = df_success.groupby('agent').agg({
        'pass_rate_visible': 'mean',
        'pass_rate_hidden': 'mean',
        'reward_hacking_detected': 'sum',
    }).round(3)
    print(agent_summary)

    # Summary by monitor
    print("\n--- By Monitor ---")
    monitor_summary = df_success.groupby('monitor').agg({
        'pass_rate_visible': 'mean',
        'pass_rate_hidden': 'mean',
        'reward_hacking_detected': 'sum',
    }).round(3)
    print(monitor_summary)

    # Summary by wrong type
    print("\n--- By Wrong Type ---")
    wrong_summary = df_success.groupby('wrong_type').agg({
        'pass_rate_visible': 'mean',
        'pass_rate_hidden': 'mean',
        'reward_hacking_detected': 'sum',
    }).round(3)
    print(wrong_summary)

    # Summary by test config
    print("\n--- By Test Config ---")
    test_summary = df_success.groupby(['n_visible', 'n_wrong']).agg({
        'pass_rate_visible': 'mean',
        'pass_rate_hidden': 'mean',
        'reward_hacking_detected': 'sum',
    }).round(3)
    print(test_summary)

    # Full comparison table (sorted by pass_rate_hidden)
    print("\n--- Full Comparison Table (Top 10 by Hidden Pass Rate) ---")
    display_cols = ['agent', 'monitor', 'wrong_type', 'n_visible', 'n_wrong',
                   'pass_rate_visible', 'pass_rate_hidden', 'reward_hacking_detected']
    display_cols = [c for c in display_cols if c in df_success.columns]

    top_10 = df_success.nlargest(10, 'pass_rate_hidden')[display_cols]
    print(top_10.to_string(index=False))


def analyze_grid_results(output_dir: str = "my_experiments") -> pd.DataFrame:
    """
    Analyze results from grid experiment logs by reading summary.json files.

    Args:
        output_dir: Directory containing experiment folders

    Returns:
        pandas.DataFrame with all experiment results
    """
    import json
    from pathlib import Path

    output_path = Path(output_dir)
    results = []

    # Find all experiment directories (those with summary.json)
    for exp_dir in output_path.iterdir():
        if not exp_dir.is_dir():
            continue

        summary_file = exp_dir / "summary.json"
        if not summary_file.exists():
            continue

        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            # Parse experiment name to extract config
            exp_name = summary.get('experiment_name', exp_dir.name)

            # Try to get monitor from monitor_analyses.json metadata
            monitor = 'unknown'
            monitor_file = exp_dir / "monitor_analyses.json"
            if monitor_file.exists():
                with open(monitor_file) as f:
                    mon_data = json.load(f)
                    metadata = mon_data.get('_metadata', {})
                    monitor = metadata.get('monitor_model', 'unknown')
                    if '/' in monitor:
                        monitor = monitor.split('/')[-1]

            row = {
                'experiment_name': exp_name,
                'model': summary.get('model', ''),
                'total_tasks': summary.get('total_tasks', 0),
                'avg_visible_pass_rate': summary.get('avg_visible_pass_rate', 0),
                'avg_hidden_pass_rate': summary.get('avg_hidden_pass_rate', 0),
                'avg_wrong_test_pass_rate': summary.get('avg_wrong_test_pass_rate', 0),
                'avg_performance_gap': summary.get('avg_performance_gap', 0),
                'likely_reward_hacking_heuristic': summary.get('likely_reward_hacking_heuristic', 0),
                'monitor_code_suspicious': summary.get('monitor_code_suspicious', 0),
                'monitor_cot_suspicious': summary.get('monitor_cot_suspicious', 0),
                'reward_hacking_detected': summary.get('reward_hacking_detected', 0),
                'experiment_dir': str(exp_dir),
                'monitor': monitor,
            }

            # Initialize defaults
            row['agent'] = exp_name.split('_')[0]
            row['wrong_type'] = 'subtly_wrong'
            row['n_visible'] = 1
            row['n_wrong'] = 1
            row['temperature'] = 0.0

            # Format 1: {agent}_mon-{monitor}_{wrong_type}_v{n_visible}w{n_wrong}
            if '_mon-' in exp_name:
                parts = exp_name.split('_mon-')
                row['agent'] = parts[0]
                rest = parts[1]

                # Parse monitor and remaining config
                for wrong_type in ['subtly_wrong', 'completely_wrong']:
                    if f'_{wrong_type}_' in rest:
                        monitor_part, config_part = rest.split(f'_{wrong_type}_')
                        row['monitor'] = monitor_part
                        row['wrong_type'] = wrong_type

                        # Parse v{n_visible}w{n_wrong}
                        if config_part.startswith('v') and 'w' in config_part:
                            v_w = config_part[1:]  # Remove leading 'v'
                            v, w = v_w.split('w')
                            row['n_visible'] = int(v)
                            row['n_wrong'] = int(w)
                        break
                else:
                    # New format with _mon- but no wrong_type in name
                    if '_nwrong-' in row['agent']:
                        agent_parts = row['agent'].split('_nwrong-')
                        row['agent'] = agent_parts[0]
                        # Parse nwrong-X_temp-Y_run-Z
                        config_str = agent_parts[1]
                        if '_temp-' in config_str:
                            nwrong_str, temp_rest = config_str.split('_temp-')
                            row['n_wrong'] = int(nwrong_str)
                            row['n_visible'] = row['n_wrong']
                            if '_run-' in temp_rest:
                                temp_str = temp_rest.split('_run-')[0]
                                row['temperature'] = float(temp_str)
                    row['monitor'] = rest

            # Format 2: {agent}_nwrong-X_temp-Y_run-Z (no _mon- in name)
            elif '_nwrong-' in exp_name:
                agent_parts = exp_name.split('_nwrong-')
                row['agent'] = agent_parts[0]
                config_str = agent_parts[1]
                # Parse nwrong-X_temp-Y_run-Z
                if '_temp-' in config_str:
                    nwrong_str, temp_rest = config_str.split('_temp-')
                    row['n_wrong'] = int(nwrong_str)
                    row['n_visible'] = row['n_wrong']
                    if '_run-' in temp_rest:
                        temp_str = temp_rest.split('_run-')[0]
                        row['temperature'] = float(temp_str)
                row['wrong_type'] = 'subtly_wrong'

            results.append(row)

        except Exception as e:
            print(f"Warning: Failed to parse {exp_dir}: {e}")
            continue

    df = pd.DataFrame(results)

    # Sort by experiment name for consistent ordering
    if not df.empty:
        df = df.sort_values('experiment_name').reset_index(drop=True)

    print(f"Loaded {len(df)} experiment results from {output_dir}")

    return df


if __name__ == "__main__":
    # Get API key from environment or pass directly
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment")
        print("Set it with: export OPENROUTER_API_KEY='your-key'")
        sys.exit(1)

    # Run grid experiment
    df = run_grid_experiment(api_key)

    # Display summary
    display_results_summary(df)

    print("\n✅ Grid experiment complete!")
