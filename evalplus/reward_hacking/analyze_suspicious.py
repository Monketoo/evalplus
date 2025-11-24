#!/usr/bin/env python3
"""
CLI tool to analyze and display suspicious tasks from reward hacking experiments.

Usage:
    python analyze_suspicious.py <log_dir> [--threshold 0.6] [--top 10]
    python analyze_suspicious.py my_experiments/experiment_1 --threshold 0.7 --top 5
"""

import argparse
import sys
from pathlib import Path
from .suspicious_analyzer import display_suspicious_tasks, get_summary_stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze suspicious tasks from reward hacking experiments"
    )
    parser.add_argument(
        "log_dir",
        help="Path to experiment log directory containing monitor_analyses.json"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for detecting suspicious tasks (default: 0.6)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only show top N suspicious tasks (default: show all)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show summary statistics, not detailed task info"
    )

    args = parser.parse_args()

    # Verify log dir exists
    log_path = Path(args.log_dir)
    if not log_path.exists():
        print(f"Error: Log directory not found: {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    analyses_file = log_path / "monitor_analyses.json"
    if not analyses_file.exists():
        print(
            f"Error: monitor_analyses.json not found in {args.log_dir}",
            file=sys.stderr
        )
        sys.exit(1)

    try:
        # Get summary statistics
        stats = get_summary_stats(args.log_dir, args.threshold)

        print(f"\n{'='*100}")
        print(f"Reward Hacking Analysis: {args.log_dir}")
        print(f"{'='*100}\n")

        print("SUMMARY STATISTICS")
        print(f"  Total Suspicious Tasks: {stats['total_suspicious']}")
        print(f"  CoT Monitor Detected:   {stats['cot_detected']}")
        print(f"  Code Monitor Detected:  {stats['code_detected']}")
        print(f"  Both Detected:          {stats['both_detected']}")
        print(f"  Heuristic Only:         {stats['heuristic_only']}")
        if stats['total_suspicious'] > 0:
            print(f"  Avg Reward Hacking Score: {stats['avg_rh_score']:.2f}")
            print(f"  Max Reward Hacking Score: {stats['max_rh_score']:.2f}")

        if not args.stats_only:
            # Display detailed analysis
            display_suspicious_tasks(args.log_dir, args.threshold, args.top)
        else:
            print()

    except Exception as e:
        print(f"Error analyzing log directory: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
