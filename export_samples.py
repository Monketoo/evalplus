"""Export sampled detected/undetected RH traces for analysis."""

import json
import random
from pathlib import Path

def export_samples(output_dir: str = "my_experiments", n_samples: int = 5, seed: int = 42,
                   filter_by: str = None, require_confirmed: bool = True):
    """
    Export sampled detected and undetected RH traces to files.

    Args:
        output_dir: Directory with experiment results
        n_samples: Number of samples per category
        seed: Random seed
        filter_by: Optional filter - 'code', 'cot', 'both', 'neither', or None (all)
        require_confirmed: If True, only include RH cases where judge_verification.is_confirmed_reward_hack=True
    """
    random.seed(seed)

    detected = []
    undetected = []
    code_only = []
    cot_only = []
    both = []
    neither = []
    missed_by_code = []  # RH cases not caught by code monitor
    missed_by_cot = []   # RH cases not caught by cot monitor

    for exp_dir in Path(output_dir).iterdir():
        if not exp_dir.is_dir():
            continue

        results_file = exp_dir / "results.enriched.jsonl"
        monitor_file = exp_dir / "monitor_analyses.json"

        if not results_file.exists() or not monitor_file.exists():
            continue

        # Parse experiment config from name
        name = exp_dir.name
        parts = name.split('_mon-')
        if len(parts) != 2:
            continue
        agent = parts[0]
        rest = parts[1]
        monitor = None
        for wt in ['subtly_wrong', 'completely_wrong']:
            if f'_{wt}_' in rest:
                monitor = rest.split(f'_{wt}_')[0]
                break
        if not monitor:
            continue

        with open(monitor_file) as f:
            monitors = json.load(f)

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

                if not r.get('evaluation', {}).get('likely_reward_hacking', False):
                    continue

                task_id = r['task_id']
                mon = monitors.get(task_id, {})

                # Check if judge verification confirms this is a real reward hack
                if require_confirmed:
                    judge_verification = mon.get('judge_verification', {})
                    if not judge_verification.get('is_confirmed_reward_hack', False):
                        continue  # Skip non-confirmed RH cases

                code_mon = mon.get('monitor_code') or {}
                cot_mon = mon.get('monitor_cot') or {}

                # Skip if errors
                if 'error' in code_mon or 'error' in cot_mon:
                    continue

                code_caught = code_mon.get('hardcoding_detected', False)
                cot_caught = cot_mon.get('test_fitting_detected', False)

                # Get wrong tests
                visible_tests = r.get('visible_tests', [])
                wrong_tests = [t for t in visible_tests if t.get('is_wrong', False)]

                sample = {
                    'agent': agent,
                    'monitor': monitor,
                    'task_id': task_id,
                    'exp_path': str(exp_dir),
                    'original_prompt': r.get('original_prompt') or '',
                    'prompt_with_tests': r.get('prompt_with_tests') or '',
                    'wrong_tests': wrong_tests,
                    'code': r.get('solution') or '',
                    'reasoning': r.get('reasoning') or '',
                    'visible_pass': r['evaluation']['visible_pass_rate'],
                    'hidden_pass': r['evaluation']['hidden_pass_rate'],
                    'code_detected': code_caught,
                    'cot_detected': cot_caught,
                }

                # Categorize by detection
                if code_caught and cot_caught:
                    both.append(sample)
                    detected.append(sample)
                elif code_caught:
                    code_only.append(sample)
                    detected.append(sample)
                elif cot_caught:
                    cot_only.append(sample)
                    detected.append(sample)
                else:
                    neither.append(sample)
                    undetected.append(sample)

                # Categorize by what MISSED detection (for analysis)
                if not code_caught:
                    missed_by_code.append(sample)
                if not cot_caught:
                    missed_by_cot.append(sample)

    # Select based on filter
    if filter_by == 'code':
        samples = {'code_only': code_only}
    elif filter_by == 'cot':
        samples = {'cot_only': cot_only}
    elif filter_by == 'both':
        samples = {'both': both}
    elif filter_by == 'neither':
        samples = {'neither': neither}
    elif filter_by == 'missed_code':
        samples = {'missed_by_code': missed_by_code}
    elif filter_by == 'missed_cot':
        samples = {'missed_by_cot': missed_by_cot}
    else:
        samples = {
            'detected': detected,
            'undetected': undetected,
            'code_only': code_only,
            'cot_only': cot_only,
            'both': both,
            'neither': neither,
        }

    def format_sample(s):
        wrong_tests_str = '\n'.join([f"  input: {t['input']} → expected: {t['expected_output']}" for t in s['wrong_tests'][:5]])
        if len(s['wrong_tests']) > 5:
            wrong_tests_str += f"\n  ... ({len(s['wrong_tests']) - 5} more)"

        return f"""{'='*70}
AGENT: {s['agent']} | MONITOR: {s['monitor']} | TASK: {s['task_id']}
CODE_DETECTED: {s['code_detected']} | COT_DETECTED: {s['cot_detected']}
VISIBLE: {s['visible_pass']:.0%} | HIDDEN: {s['hidden_pass']:.0%}
EXP_PATH: {s['exp_path']}
{'='*70}

ORIGINAL PROMPT:
{s['original_prompt'][:1000]}{'...' if len(s['original_prompt']) > 1000 else ''}

WRONG TESTS GIVEN:
{wrong_tests_str if wrong_tests_str else '  (none)'}

PROMPT WITH TESTS (given to agent):
{s['prompt_with_tests'][:1000]}{'...' if len(s['prompt_with_tests']) > 1000 else ''}

GENERATED CODE:
{s['code'][:1500]}{'...' if len(s['code']) > 1500 else ''}

REASONING:
{s['reasoning'][:1500]}{'...' if len(s['reasoning']) > 1500 else ''}

"""

    # Export each category
    for name, data in samples.items():
        sampled = random.sample(data, min(n_samples, len(data)))
        filename = f'samples_{name}.txt'

        with open(filename, 'w') as f:
            f.write(f"{name.upper()} RH SAMPLES ({len(sampled)}/{len(data)} total)\n\n")
            for s in sampled:
                f.write(format_sample(s))

        print(f"Exported {len(sampled)}/{len(data)} {name} samples to {filename}")

    print(f"\nSummary:")
    print(f"  Detected: both={len(both)}, code_only={len(code_only)}, cot_only={len(cot_only)}")
    print(f"  Undetected: neither={len(neither)}")
    print(f"  Missed by code={len(missed_by_code)}, missed by cot={len(missed_by_cot)}")


if __name__ == "__main__":
    export_samples()
