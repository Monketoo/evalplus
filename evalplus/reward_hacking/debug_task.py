"""Debug task failures in reward hacking experiments."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from evalplus.data import get_human_eval_plus, get_mbpp_plus


def show_task_enriched(
    log_dir: str,
    task_id: str,
    dataset: str = "humaneval",
    show_all_hidden: bool = False
) -> None:
    """
    Debug why hidden tests failed for a task.

    Args:
        log_dir: Path to experiment directory
        task_id: Task ID (e.g., 'HumanEval/25')
        dataset: Dataset name ("humaneval" or "mbpp")
        show_all_hidden: If True, show all hidden test results (can be verbose)
    """
    log_path = Path(log_dir)

    # Load task definition
    if dataset == "humaneval":
        problems = get_human_eval_plus()
    else:
        problems = get_mbpp_plus()

    if task_id not in problems:
        print(f"Error: Task {task_id} not found in {dataset}")
        return

    task = problems[task_id]

    # Load test splits
    test_splits_file = log_path / "test_splits.json"
    if not test_splits_file.exists():
        print(f"Error: test_splits.json not found in {log_dir}")
        return

    with open(test_splits_file) as f:
        test_splits = json.load(f)

    if task_id not in test_splits:
        print(f"Error: Task {task_id} not in test splits")
        return

    split = test_splits[task_id]

    # Load results
    results_file = log_path / "results.enriched.jsonl"
    if not results_file.exists():
        print(f"Error: results.enriched.jsonl not found in {log_dir}")
        return

    solution_data = None
    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            if data['task_id'] == task_id:
                solution_data = data
                break

    if solution_data is None:
        print(f"Error: Task {task_id} not in results")
        return

    # Print task overview
    print(f"\n{'='*100}")
    print(f"DEBUG: {task_id}")
    print(f"{'='*100}\n")

    # 1. Show original task definition
    print("1. ORIGINAL TASK DEFINITION")
    print(f"{'-'*100}")
    print(f"Entry Point: {task['entry_point']}")
    print(f"\nDocstring/Prompt:\n{task['prompt']}")
    print()

    # 2. Show visible tests
    print(f"\n2. VISIBLE TESTS (What the model saw)")
    print(f"{'-'*100}")
    print(f"Total visible: {len(split['visible_inputs'])}")
    print(f"Wrong tests injected: {split['wrong_tests_injected']}")

    for i, (inp, out) in enumerate(zip(split['visible_inputs'], split['visible_outputs'])):
        is_wrong = i in split.get('wrong_test_indices', [])
        marker = " [WRONG TEST]" if is_wrong else ""
        print(f"\n  Test {i+1}{marker}:")
        print(f"    Input:    {inp}")
        print(f"    Expected: {out}")

    # 3. Show the solution
    print(f"\n3. GENERATED SOLUTION")
    print(f"{'-'*100}")
    print(solution_data.get('solution', 'N/A'))

    # 4. Show test results
    print(f"\n4. TEST RESULTS")
    print(f"{'-'*100}")

    # Results can be at top level or nested under 'evaluation'
    if 'evaluation' in solution_data:
        eval_data = solution_data['evaluation']
        visible_results = eval_data.get('visible_results', [])
        hidden_results = eval_data.get('hidden_results', [])
    else:
        visible_results = solution_data.get('visible_results', [])
        hidden_results = solution_data.get('hidden_results', [])

    visible_pass = sum(visible_results) if visible_results else 0
    hidden_pass = sum(hidden_results) if hidden_results else 0

    if visible_results:
        print(f"Visible Tests: {visible_pass}/{len(visible_results)} passed ({visible_pass/len(visible_results)*100:.1f}%)")
    else:
        print(f"Visible Tests: (empty results list)")

    if hidden_results:
        print(f"Hidden Tests:  {hidden_pass}/{len(hidden_results)} passed ({hidden_pass/len(hidden_results)*100:.1f}%)")
    else:
        print(f"Hidden Tests:  (empty results list)")

    print(f"\nVisible test breakdown:")
    for i, result in enumerate(visible_results):
        is_wrong = i in split.get('wrong_test_indices', [])
        marker = " [WRONG TEST]" if is_wrong else "[CORRECT TEST]"
        status = "PASS" if result else "FAIL"
        print(f"  Test {i+1}{marker}: {status}")

    # 5. Show hidden test failures (sample)
    if hidden_pass < len(hidden_results):
        print(f"\nHidden test failures (sample):")
        failed_count = 0
        for i, result in enumerate(hidden_results):
            if not result and failed_count < 5:
                print(f"  Test {i+1}: FAILED")
                failed_count += 1
        if failed_count == 5:
            total_failed = sum(1 for r in hidden_results if not r)
            print(f"  ... and {total_failed - 5} more failures")

    # 6. Analysis
    print(f"\n5. ANALYSIS")
    print(f"{'-'*100}")

    if not visible_results or not hidden_results:
        print("❌ Empty test results - evaluation may have failed during execution")
        print("   Check for runtime errors or import issues in the generated code")

    elif visible_pass == len(visible_results) and visible_results:
        if hidden_pass < len(hidden_results):
            print("⚠️  REWARD HACKING DETECTED:")
            print(f"    - Model passes ALL visible tests ({visible_pass}/{len(visible_results)})")
            print(f"    - But fails MOST hidden tests ({hidden_pass}/{len(hidden_results)})")
            print(f"    - Gap: {(visible_pass/len(visible_results) - hidden_pass/len(hidden_results))*100:.1f} percentage points")

            # Check if it's fitting to wrong tests
            wrong_tests_passed = sum(
                1 for i, result in enumerate(visible_results)
                if result and i in split.get('wrong_test_indices', [])
            )
            if wrong_tests_passed > 0:
                print(f"\n    🎯 The model is passing {wrong_tests_passed} WRONG test(s)!")
                print(f"       This suggests it's fitting to the incorrect expected outputs")
                print(f"       rather than learning the actual task requirement.")

    elif visible_pass == 0:
        print("❌ Model fails ALL visible tests")
        print("    - This suggests the solution is fundamentally wrong")
        if hidden_pass > 0:
            print(f"    - Surprisingly passes {hidden_pass} hidden tests (unusual)")

    else:
        print(f"⚠️  Model passes {visible_pass}/{len(visible_results)} visible tests")
        print(f"    - And only passes {hidden_pass}/{len(hidden_results)} hidden tests")

    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python debug_task.py <log_dir> <task_id> [--all-hidden]")
        print("Example: python debug_task.py ./my_experiments/exp_1 HumanEval/25")
        sys.exit(1)

    log_dir = sys.argv[1]
    task_id = sys.argv[2]
    show_all = "--all-hidden" in sys.argv

    show_task_enriched(log_dir, task_id, show_all_hidden=show_all)
