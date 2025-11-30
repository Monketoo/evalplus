"""Diagnose setups with highest undetected reward hacking rates.

Analyzes Code and CoT monitors SEPARATELY:
- Code monitor: uses valid_code (non-empty solution)
- CoT monitor: uses valid_cot (non-empty solution AND non-empty reasoning > 50 chars)

NOTE: Only counts CONFIRMED reward hacks (judge_verification.is_confirmed_reward_hack=True)
"""

import pandas as pd
from analyze_monitorability import collect_all

def diagnose_undetected(require_confirmed: bool = True):
    """
    Diagnose undetected reward hacks.

    Args:
        require_confirmed: If True, only count RH cases confirmed by judge
    """
    df = collect_all(require_confirmed=require_confirmed)

    print(f"Using {'CONFIRMED' if require_confirmed else 'ALL (heuristic)'} reward hacks\n")

    # Filter to setups with at least 10 valid CoT cases
    df = df[df['valid_cot'] >= 10].copy()
    print(f"Filtered to {len(df)} setups with valid_cot >= 10\n")

    # Calculate miss rates for each monitor type separately
    # Code: missed = total_rh - caught_code - code_errors (among valid_code)
    df['code_missed'] = df['valid_code'] - df['caught_code']  # valid_code already excludes errors
    df['code_miss_rate'] = df['code_missed'] / df['valid_code'].replace(0, float('nan'))

    # CoT: missed = valid_cot - caught_cot (valid_cot already excludes errors and empty reasoning)
    df['cot_missed'] = df['valid_cot'] - df['caught_cot']
    df['cot_miss_rate'] = df['cot_missed'] / df['valid_cot'].replace(0, float('nan'))

    # =========================================================================
    # CODE MONITOR ANALYSIS
    # =========================================================================
    print("=" * 80)
    print("CODE MONITOR: TOP 10 SETUPS WITH HIGHEST MISS RATE")
    print("(Requires: non-empty solution)")
    print("=" * 80)

    top_code = df.nlargest(10, 'code_miss_rate')[
        ['agent', 'monitor', 'wrong_type', 'config',
         'valid_code', 'caught_code', 'code_missed', 'code_miss_rate']
    ]
    print(top_code.to_string(index=False))

    print("\n" + "=" * 80)
    print("CODE MONITOR: BREAKDOWN BY AGENT")
    print("=" * 80)
    by_agent_code = df.groupby('agent').agg({
        'valid_code': 'sum',
        'caught_code': 'sum',
        'code_missed': 'sum',
    }).sort_values('code_missed', ascending=False)
    by_agent_code['miss_rate'] = by_agent_code['code_missed'] / by_agent_code['valid_code']
    by_agent_code['catch_rate'] = by_agent_code['caught_code'] / by_agent_code['valid_code']
    print(by_agent_code.to_string())

    print("\n" + "=" * 80)
    print("CODE MONITOR: BREAKDOWN BY MONITOR MODEL")
    print("=" * 80)
    by_mon_code = df.groupby('monitor').agg({
        'valid_code': 'sum',
        'caught_code': 'sum',
        'code_missed': 'sum',
    }).sort_values('miss_rate' if 'miss_rate' in df.columns else 'code_missed', ascending=False)
    by_mon_code['miss_rate'] = by_mon_code['code_missed'] / by_mon_code['valid_code']
    by_mon_code['catch_rate'] = by_mon_code['caught_code'] / by_mon_code['valid_code']
    by_mon_code = by_mon_code.sort_values('miss_rate', ascending=False)
    print(by_mon_code.to_string())

    print("\n" + "=" * 80)
    print("CODE MONITOR: BREAKDOWN BY WRONG TYPE")
    print("=" * 80)
    by_wrong_code = df.groupby('wrong_type').agg({
        'valid_code': 'sum',
        'caught_code': 'sum',
        'code_missed': 'sum',
    })
    by_wrong_code['miss_rate'] = by_wrong_code['code_missed'] / by_wrong_code['valid_code']
    by_wrong_code['catch_rate'] = by_wrong_code['caught_code'] / by_wrong_code['valid_code']
    by_wrong_code = by_wrong_code.sort_values('miss_rate', ascending=False)
    print(by_wrong_code.to_string())

    # =========================================================================
    # COT MONITOR ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("COT MONITOR: TOP 10 SETUPS WITH HIGHEST MISS RATE")
    print("(Requires: non-empty solution AND reasoning > 50 chars)")
    print("=" * 80)

    top_cot = df.nlargest(10, 'cot_miss_rate')[
        ['agent', 'monitor', 'wrong_type', 'config',
         'valid_cot', 'caught_cot', 'cot_missed', 'cot_miss_rate']
    ]
    print(top_cot.to_string(index=False))

    print("\n" + "=" * 80)
    print("COT MONITOR: BREAKDOWN BY AGENT")
    print("=" * 80)
    by_agent_cot = df.groupby('agent').agg({
        'valid_cot': 'sum',
        'caught_cot': 'sum',
        'cot_missed': 'sum',
    }).sort_values('cot_missed', ascending=False)
    by_agent_cot['miss_rate'] = by_agent_cot['cot_missed'] / by_agent_cot['valid_cot']
    by_agent_cot['catch_rate'] = by_agent_cot['caught_cot'] / by_agent_cot['valid_cot']
    print(by_agent_cot.to_string())

    print("\n" + "=" * 80)
    print("COT MONITOR: BREAKDOWN BY MONITOR MODEL")
    print("=" * 80)
    by_mon_cot = df.groupby('monitor').agg({
        'valid_cot': 'sum',
        'caught_cot': 'sum',
        'cot_missed': 'sum',
    })
    by_mon_cot['miss_rate'] = by_mon_cot['cot_missed'] / by_mon_cot['valid_cot']
    by_mon_cot['catch_rate'] = by_mon_cot['caught_cot'] / by_mon_cot['valid_cot']
    by_mon_cot = by_mon_cot.sort_values('miss_rate', ascending=False)
    print(by_mon_cot.to_string())

    print("\n" + "=" * 80)
    print("COT MONITOR: BREAKDOWN BY WRONG TYPE")
    print("=" * 80)
    by_wrong_cot = df.groupby('wrong_type').agg({
        'valid_cot': 'sum',
        'caught_cot': 'sum',
        'cot_missed': 'sum',
    })
    by_wrong_cot['miss_rate'] = by_wrong_cot['cot_missed'] / by_wrong_cot['valid_cot']
    by_wrong_cot['catch_rate'] = by_wrong_cot['caught_cot'] / by_wrong_cot['valid_cot']
    by_wrong_cot = by_wrong_cot.sort_values('miss_rate', ascending=False)
    print(by_wrong_cot.to_string())

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: CODE vs COT MONITOR")
    print("=" * 80)
    print(f"CODE Monitor - Total valid RH: {df['valid_code'].sum()}, Caught: {df['caught_code'].sum()}, "
          f"Missed: {df['code_missed'].sum()}, Miss rate: {df['code_missed'].sum() / df['valid_code'].sum():.1%}")
    print(f"COT Monitor  - Total valid RH: {df['valid_cot'].sum()}, Caught: {df['caught_cot'].sum()}, "
          f"Missed: {df['cot_missed'].sum()}, Miss rate: {df['cot_missed'].sum() / df['valid_cot'].sum():.1%}")

    print("\n" + "=" * 80)
    print("AGENT x MONITOR: CODE MISS RATE")
    print("=" * 80)
    pivot_code = df.pivot_table(
        values='code_miss_rate',
        index='agent',
        columns='monitor',
        aggfunc='mean'
    )
    print(pivot_code.to_string())

    print("\n" + "=" * 80)
    print("AGENT x MONITOR: COT MISS RATE")
    print("=" * 80)
    pivot_cot = df.pivot_table(
        values='cot_miss_rate',
        index='agent',
        columns='monitor',
        aggfunc='mean'
    )
    print(pivot_cot.to_string())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose undetected reward hacks")
    parser.add_argument("--all", action="store_true", help="Include all heuristic RH (not just confirmed)")
    args = parser.parse_args()

    diagnose_undetected(require_confirmed=not args.all)
