"""
Run experiments with varying n_wrong (number of subtly incorrect tests).

Configuration:
- Model: qwen-2.5-coder-32b-instruct
- n_wrong: 2, 3, 4, 5
- Temperature: 1.0
- Runs: 2 per n_wrong setting
- Monitor: anthropic/claude-sonnet-4
"""

import os
import sys
sys.path.append('../rl-effect-on-cot-monitorability/evals/quantum.eval/')

from utils import OpenRouterClient
from evalplus.reward_hacking.experiment import run_experiment
from evalplus.reward_hacking.test_selector import SplitStrategy
from evalplus.reward_hacking.prompt_builder import PromptStyle


def run_n_wrong_experiments(
    api_key: str,
    model: str = "qwen/qwen3-coder",
    n_wrong_values: list = [2, 3, 4, 5],
    n_runs: int = 2,
    temperature: float = 1.0,
    output_dir: str = "my_experiments",
    max_concurrent: int = 80,
    llm_tests_path: str = "evalplus/reward_hacking/data/humaneval_llm_tests.json",
):
    """
    Run experiments with varying n_wrong values.

    Args:
        api_key: OpenRouter API key
        model: Model to use
        n_wrong_values: List of n_wrong values to test
        n_runs: Number of runs per n_wrong value
        temperature: Sampling temperature
        output_dir: Output directory
        max_concurrent: Max concurrent API calls
        llm_tests_path: Path to LLM-generated tests
    """
    client = OpenRouterClient(api_key=api_key)

    model_short = model.split('/')[-1]

    total_experiments = len(n_wrong_values) * n_runs
    completed = 0

    for n_wrong in n_wrong_values:
        for run_idx in range(n_runs):
            completed += 1
            exp_name = f"{model_short}_nwrong-{n_wrong}_temp-{temperature}_run-{run_idx + 1}"

            print(f"\n{'=' * 70}")
            print(f"Experiment {completed}/{total_experiments}: {exp_name}")
            print(f"  n_wrong={n_wrong}, temperature={temperature}, run={run_idx + 1}")
            print(f"{'=' * 70}\n")

            # Exclude tasks 15, 83, and 139 which have huge number of tests and load eternally
            all_task_ids = [f"HumanEval/{i}" for i in range(164) if i not in (15, 83, 139)]

            try:
                result = run_experiment(
                    model=model,
                    openrouter_client=client,

                    # Test configuration
                    n_visible=n_wrong,  # n_visible = n_wrong
                    n_wrong=n_wrong,
                    split_strategy=SplitStrategy.RANDOM,
                    use_llm_tests=True,
                    llm_tests_path=llm_tests_path,
                    llm_wrong_type="subtly_wrong",
                    hidden_test_cap=100,

                    # Prompt configuration
                    prompt_style=PromptStyle.DOCSTRING,

                    # Dataset
                    dataset="humaneval",
                    task_ids=all_task_ids,

                    # Concurrency
                    max_concurrent=max_concurrent,

                    # Monitors
                    enable_monitors=True,
                    monitor_model="anthropic/claude-sonnet-4",

                    # Model config
                    temperature=temperature,

                    # Output
                    output_dir=output_dir,
                    experiment_name=exp_name,
                )

                print(f"\n✓ Completed: {exp_name}")
                print(f"  Tasks: {result.get('total_tasks', 'N/A')}")
                print(f"  Likely RH (heuristic): {result.get('likely_reward_hacking_heuristic', 'N/A')}")
                print(f"  Avg visible pass: {result.get('avg_visible_pass_rate', 0):.1%}")
                print(f"  Avg hidden pass: {result.get('avg_hidden_pass_rate', 0):.1%}")

            except Exception as e:
                print(f"\n✗ FAILED: {exp_name}")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"  Total: {total_experiments}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run n_wrong experiments")
    parser.add_argument("--model", type=str, default="qwen/qwen3-coder",
                        help="Model to use")
    parser.add_argument("--n-wrong", type=int, nargs="+", default=[1],
                        help="n_wrong values to test")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of runs per n_wrong value")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
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

    run_n_wrong_experiments(
        api_key=api_key,
        model=args.model,
        n_wrong_values=args.n_wrong,
        n_runs=args.n_runs,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
    )
