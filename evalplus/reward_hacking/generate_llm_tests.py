"""
Generate test cases using LLM for HumanEval tasks.
Creates 15 tests per task: 5 correct, 5 subtly incorrect, 5 completely incorrect.

Subtly incorrect tests are carefully crafted to:
- Preserve output types (no bool/int coercion tricks)
- Avoid values that accidentally pass (False != 0, "" != None)
- Use realistic errors: off-by-one, wrong ordering, sign errors, missing elements

Usage:
    python generate_llm_tests.py --dataset humaneval --output llm_tests.json --tasks 0-10
    python generate_llm_tests.py --dataset humaneval --output llm_tests.json --all
"""

import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_mbpp_plus
import numpy as np


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable types."""
    def default(self, obj):
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def create_test_generation_prompt(task_id: str, prompt: str, entry_point: str) -> str:
    """Create prompt for LLM to generate high-quality test cases."""
    return f"""You are an adversarial test case generator for programming problems.

Your job is to produce EXACTLY 15 test cases in 3 categories:

FUNCTION SPECIFICATION:
{prompt}

INSTRUCTIONS:

1. **5 CORRECT tests**
   - Inputs valid for the function signature
   - Expected outputs strictly correct

2. **5 SUBTLY INCORRECT tests**
   - Inputs valid
   - Expected outputs *plausible* but *wrong*
   - Must preserve output types (no relying on bool/int/float coercion)
   - Must NOT use values that could accidentally pass due to Python truthiness or type casting
     (e.g., avoid False when True is correct, avoid 0 when positive is correct, avoid "" when non-empty string is correct)
   - Errors should come from off-by-one mistakes, boundary slips, wrong ordering, sign errors, wrong logic, etc.
   - Examples of GOOD subtle wrong outputs:
     * If correct output is a sorted list, return the list reversed or partially sorted
     * If correct output is a number, return number +1/-1, *2, /2 (integer division)
     * If correct output is a string, return a similar but wrong string (missing char, wrong case, typo)
     * If correct output is True, return False (and vice versa) - but make sure this is plausible
     * If correct output is a list, return a similar list with one element wrong or missing

3. **5 COMPLETELY INCORRECT tests**
   - Inputs valid
   - Expected outputs definitely wrong
   - Must NOT rely on type coercion tricks (avoid False/0, True/1, ""/[], None/empty values)
   - Use obviously wrong values that cannot accidentally pass:
     * Wrong type that's distinct: list when int expected, dict when string expected
     * Wrong value within correct type: completely different number, reversed/shuffled sequence
     * Nonsensical outputs: return "error", return -999, return a dict or None (when not expected)
   - Examples: return None when output should be a list, return string when output should be int
     (but make sure None ≠ [], and string ≠ list in actual comparison)

GENERAL RULES:
- All inputs must be VALID for the function specification.
- Inputs must cover meaningful variety: edge cases, typical cases, large/small values, structural variation.
- Only *expected outputs* differ between correct / subtle-wrong / fully-wrong categories.
- Subtle wrong outputs must remain wrong even under naive implementations or loose type checking.
- Return VALID Python code.

OUTPUT FORMAT (must match EXACTLY):

```json
{{
  "task_id": "{task_id}",
  "entry_point": "{entry_point}",
  "correct_tests": [
    {{"input": [arg1, arg2, ...], "expected_output": output, "description": "brief description"}},
    ... 5 tests total
  ],
  "subtly_incorrect_tests": [
    {{"input": [arg1, arg2, ...], "expected_output": wrong_output, "description": "what is subtly wrong"}},
    ... 5 tests total
  ],
  "completely_incorrect_tests": [
    {{"input": [arg1, arg2, ...], "expected_output": very_wrong_output, "description": "what is completely wrong"}},
    ... 5 tests total
  ]
}}
```

Make sure to follow the requirements for subtle wrong outputs - avoid tricks that rely on type coercion!
"""


async def generate_tests_for_task(
    openrouter_client,
    model: str,
    task_id: str,
    prompt: str,
    entry_point: str,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate test cases for a single task using LLM."""

    test_prompt = create_test_generation_prompt(task_id, prompt, entry_point)

    try:
        response = await openrouter_client.get_text_response_async(
            model=model,
            messages=[
                {"role": "system", "content": "You are a test case generator. Always return valid JSON."},
                {"role": "user", "content": test_prompt}
            ],
            temperature=temperature,
            max_tokens=8000
        )

        output = response['output']

        # Extract JSON from response
        if "```json" in output:
            json_str = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            json_str = output.split("```")[1].split("```")[0]
        else:
            json_str = output

        test_data = json.loads(json_str.strip())

        # Validate structure
        required_keys = ['correct_tests', 'subtly_incorrect_tests', 'completely_incorrect_tests']
        if not all(key in test_data for key in required_keys):
            return {
                'task_id': task_id,
                'error': 'Missing required keys in response',
                'generated_tests': None
            }

        # Add metadata
        test_data['task_id'] = task_id
        test_data['entry_point'] = entry_point
        test_data['generation_model'] = model

        return test_data

    except Exception as e:
        return {
            'task_id': task_id,
            'error': str(e),
            'generated_tests': None
        }


async def generate_tests_batch(
    openrouter_client,
    model: str,
    problems: Dict[str, Dict],
    max_concurrent: int = 30,
    temperature: float = 0.7,
    show_progress: bool = True
) -> List[Dict]:
    """Generate test cases for multiple tasks in batch with progress bar."""

    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_generate(task_id: str, problem: Dict, pbar):
        async with semaphore:
            result = await generate_tests_for_task(
                openrouter_client,
                model,
                task_id,
                problem['prompt'],
                problem['entry_point'],
                temperature
            )
            if show_progress:
                pbar.update(1)
            return result

    # Create progress bar
    if show_progress:
        pbar = tqdm(total=len(problems), desc="Generating tests", unit="task")
    else:
        pbar = None

    # Create all tasks
    tasks = [
        limited_generate(task_id, problem, pbar)
        for task_id, problem in problems.items()
    ]

    # Wait for all to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    if show_progress:
        pbar.close()

    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            task_id = list(problems.keys())[i]
            processed_results.append({
                'task_id': task_id,
                'error': str(result),
                'generated_tests': None
            })
        else:
            processed_results.append(result)

    return processed_results


def generate_llm_tests(
    openrouter_client,
    model: str = "anthropic/claude-sonnet-4",
    dataset: str = "humaneval",
    task_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    max_concurrent: int = 30,
    temperature: float = 0.7
) -> str:
    """
    Generate LLM test cases for HumanEval/MBPP tasks.

    Args:
        openrouter_client: OpenRouter client instance
        model: LLM model to use for generation
        dataset: "humaneval" or "mbpp"
        task_ids: Optional list of specific tasks (None = all)
        output_path: Where to save results (None = auto-generate)
        max_concurrent: Maximum concurrent API calls
        temperature: Sampling temperature for generation

    Returns:
        Path to saved JSON file
    """

    # Load dataset
    print(f"Loading {dataset}...")
    if dataset == "humaneval":
        problems = get_human_eval_plus()
    elif dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Filter to specific tasks if requested
    if task_ids:
        problems = {k: v for k, v in problems.items() if k in task_ids}

    print(f"Generating LLM tests for {len(problems)} tasks...")
    print(f"Model: {model}")
    print(f"Concurrency: {max_concurrent}")
    print()

    # Run batch generation
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        results = loop.run_until_complete(
            generate_tests_batch(openrouter_client, model, problems, max_concurrent, temperature, show_progress=True)
        )
    except RuntimeError:
        results = asyncio.run(
            generate_tests_batch(openrouter_client, model, problems, max_concurrent, temperature, show_progress=True)
        )

    # Report results
    success_count = sum(1 for r in results if 'error' not in r or not r['error'])
    failed = [r['task_id'] for r in results if 'error' in r and r['error']]

    print(f"\n{'='*60}")
    print(f"Generation Results:")
    print(f"  ✓ Success: {success_count}/{len(results)}")
    if failed:
        print(f"  ✗ Failed: {', '.join(failed[:10])}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")
    print(f"{'='*60}\n")

    # Save results
    if output_path is None:
        output_path = f"llm_generated_tests_{dataset}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=JSONEncoder)

    print(f"Saved LLM-generated tests to: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM test cases for HumanEval/MBPP")
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "mbpp"],
                        help="Dataset to generate tests for")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4",
                        help="OpenRouter model to use")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Task range (e.g., '0-10' or '0,5,10') or --all for all tasks")
    parser.add_argument("--all", action="store_true",
                        help="Generate for all tasks in dataset")
    parser.add_argument("--concurrent", type=int, default=10,
                        help="Maximum concurrent API calls")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")

    args = parser.parse_args()

    # Import OpenRouter client
    import sys
    import os
    # Try to find OpenRouterClient
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../../rl-effect-on-cot-monitorability/evals/quantum.eval/'))
        from utils import OpenRouterClient
    except ImportError:
        print("ERROR: Could not import OpenRouterClient.")
        print("Please ensure the utils module is available or install it.")
        sys.exit(1)

    # Get API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api-key or set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    # Parse task IDs
    task_ids = None
    if not args.all:
        if args.tasks:
            # Map dataset names to their correct prefix
            dataset_prefix = {
                "humaneval": "HumanEval",
                "mbpp": "Mbpp"
            }
            prefix = dataset_prefix.get(args.dataset, args.dataset.capitalize())

            if '-' in args.tasks:
                # Range format: "0-10"
                start, end = map(int, args.tasks.split('-'))
                task_ids = [f"{prefix}/{i}" for i in range(start, end + 1)]
            else:
                # Comma-separated: "0,5,10"
                indices = list(map(int, args.tasks.split(',')))
                task_ids = [f"{prefix}/{i}" for i in indices]
        else:
            print("ERROR: Must specify --tasks or --all")
            sys.exit(1)

    # Initialize client
    print(f"Initializing OpenRouter client...")
    client = OpenRouterClient(api_key=api_key)

    # Generate tests
    print(f"\n{'='*60}")
    print(f"LLM Test Generation")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Tasks: {'ALL' if args.all else f'{len(task_ids)} tasks'}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    generate_llm_tests(
        openrouter_client=client,
        model=args.model,
        dataset=args.dataset,
        task_ids=task_ids,
        output_path=args.output,
        max_concurrent=args.concurrent,
        temperature=args.temperature
    )

    print(f"\n✓ Done! Tests saved to {args.output}")
