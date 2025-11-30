"""Visualization script for grid experiment results."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Model params mapping
MODEL_PARAMS = {
    'qwen-2.5-coder-32b-instruct': 32,
    'qwen2.5-coder-7b-instruct': 7,
    'qwen3-coder': 35,  # 35B active params
}


def extract_model_name(agent_or_exp_name: str, model_col: str = None) -> str:
    """Extract base model name from agent column or experiment name.

    Handles both old format (e.g., 'qwen2.5-coder-7b-instruct') and
    new format (e.g., 'qwen-2.5-coder-32b-instruct_nwrong-1_temp-1.0_run-1').
    """
    # If we have a model column value, extract from it
    if model_col and '/' in str(model_col):
        # e.g., 'qwen/qwen-2.5-coder-32b-instruct' -> 'qwen-2.5-coder-32b-instruct'
        return model_col.split('/')[-1]

    name = str(agent_or_exp_name)

    # Check if it's already a known model name
    for model in MODEL_PARAMS.keys():
        if name == model:
            return model

    # Try to extract model name from experiment name patterns
    # Pattern: {model}_nwrong-{n}_temp-{t}_run-{r}[_mon-{monitor}]
    for model in MODEL_PARAMS.keys():
        if name.startswith(model + '_') or name.startswith(model + '_nwrong'):
            return model

    # Fallback: try to match by prefix
    if 'qwen-2.5-coder-32b' in name or 'qwen/qwen-2.5-coder-32b' in name:
        return 'qwen-2.5-coder-32b-instruct'
    if 'qwen2.5-coder-7b' in name or 'qwen/qwen2.5-coder-7b' in name:
        return 'qwen2.5-coder-7b-instruct'
    if 'qwen3-coder' in name or 'qwen/qwen3-coder' in name:
        return 'qwen3-coder'

    return name  # Return as-is if no match


def count_confirmed_rh(exp_dir: Path) -> int:
    """Count confirmed reward hacks in an experiment."""
    monitor_file = exp_dir / "monitor_analyses.json"
    if not monitor_file.exists():
        return 0

    with open(monitor_file) as f:
        monitors = json.load(f)

    confirmed = 0
    for task_id, mon in monitors.items():
        judge_verification = mon.get('judge_verification', {})
        if judge_verification.get('is_confirmed_reward_hack', False):
            confirmed += 1
    return confirmed


def load_and_prepare(csv_path: str = "exp_results.csv", experiments_dir: str = "my_experiments") -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)

    # Normalize agent column - extract base model name from agent or model column
    df['agent'] = df.apply(
        lambda row: extract_model_name(
            row.get('agent', '') or row.get('experiment_name', ''),
            row.get('model', '')
        ),
        axis=1
    )
    df['params_b'] = df['agent'].map(MODEL_PARAMS)

    # Count confirmed RH from monitor_analyses.json for each experiment
    confirmed_rh = {}
    for exp_dir in Path(experiments_dir).iterdir():
        if exp_dir.is_dir():
            confirmed_rh[exp_dir.name] = count_confirmed_rh(exp_dir)

    # Map confirmed RH counts to dataframe (use experiment_name if available)
    df['confirmed_rh'] = df.apply(
        lambda row: confirmed_rh.get(row.get('experiment_name', ''), 0),
        axis=1
    )

    # Use confirmed RH for rate calculation (fallback to heuristic if no confirmed data)
    df['rh_rate'] = df.apply(
        lambda row: row['confirmed_rh'] / row['total_tasks'] if row['confirmed_rh'] > 0
        else row['likely_reward_hacking_heuristic'] / row['total_tasks'],
        axis=1
    )
    df['monitor_flag_rate'] = df['reward_hacking_detected'] / df['total_tasks']
    return df

def plot_rh_by_model(df: pd.DataFrame, show: bool = True):
    """Reward hacking rate by model size."""
    fig, ax = plt.subplots(figsize=(8, 6))

    agg = df.groupby(['agent', 'params_b'])['rh_rate'].mean().reset_index()
    agg = agg.sort_values('params_b')

    bars = ax.bar(agg['agent'], agg['rh_rate'], color=['#2ecc71', '#3498db', '#9b59b6'])
    ax.set_ylabel('Reward Hacking Rate')
    ax.set_title('Reward Hacking by Model')
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([f"{a}\n({int(p)}B)" for a, p in zip(agg['agent'], agg['params_b'])], rotation=0)

    for bar, rate in zip(bars, agg['rh_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, max(agg['rh_rate'].max() * 1.2, 0.15))
    plt.tight_layout()
    if not show:
        plt.close(fig)
    return fig

def plot_rh_heatmap(df: pd.DataFrame, show: bool = True):
    """Heatmap: agent x wrong_type x test_config."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, wrong_type in zip(axes, ['subtly_wrong', 'completely_wrong']):
        subset = df[df['wrong_type'] == wrong_type]
        pivot = subset.pivot_table(
            values='rh_rate',
            index='agent',
            columns=['n_visible', 'n_wrong'],
            aggfunc='mean'
        )
        pivot.columns = [f'v{int(v)}w{int(w)}' for v, w in pivot.columns]

        sns.heatmap(pivot, annot=True, fmt='.1%', cmap='Reds', ax=ax, vmin=0, vmax=0.3)
        ax.set_title(f'RH Rate: {wrong_type}')

    plt.tight_layout()
    if not show:
        plt.close(fig)
    return fig

def plot_pass_rates(df: pd.DataFrame, show: bool = True):
    """Visible vs Hidden pass rates."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'qwen2.5-coder-7b-instruct': '#2ecc71',
              'qwen-2.5-coder-32b-instruct': '#3498db',
              'qwen3-coder': '#9b59b6'}

    for agent in df['agent'].unique():
        subset = df[df['agent'] == agent]
        ax.scatter(subset['avg_visible_pass_rate'], subset['avg_hidden_pass_rate'],
                   label=f"{agent} ({MODEL_PARAMS.get(agent, '?')}B)",
                   alpha=0.7, s=80, c=colors.get(agent, 'gray'))

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Visible Pass Rate')
    ax.set_ylabel('Hidden Pass Rate')
    ax.set_title('Visible vs Hidden Test Performance')
    ax.legend()
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if not show:
        plt.close(fig)
    return fig

def plot_monitor_comparison(df: pd.DataFrame, show: bool = True):
    """Monitor detection vs heuristic detection."""
    fig, ax = plt.subplots(figsize=(8, 6))

    agg = df.groupby('agent').agg({
        'likely_reward_hacking_heuristic': 'sum',
        'monitor_code_suspicious': 'sum',
        'monitor_cot_suspicious': 'sum',
    }).reset_index()

    x = range(len(agg))
    w = 0.25

    ax.bar([i - w for i in x], agg['likely_reward_hacking_heuristic'], w, label='Heuristic RH', color='#e74c3c')
    ax.bar([i for i in x], agg['monitor_code_suspicious'], w, label='Code Monitor', color='#3498db')
    ax.bar([i + w for i in x], agg['monitor_cot_suspicious'], w, label='CoT Monitor', color='#2ecc71')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n({MODEL_PARAMS.get(a, '?')}B)" for a in agg['agent']])
    ax.set_ylabel('Count')
    ax.set_title('Detection Methods Comparison')
    ax.legend()

    plt.tight_layout()
    if not show:
        plt.close(fig)
    return fig

def plot_performance_gap(df: pd.DataFrame, show: bool = True):
    """Performance gap distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for agent in sorted(df['agent'].unique()):
        subset = df[df['agent'] == agent]
        ax.hist(subset['avg_performance_gap'], bins=15, alpha=0.5,
                label=f"{agent} ({MODEL_PARAMS.get(agent, '?')}B)")

    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Performance Gap (visible - hidden)')
    ax.set_ylabel('Count')
    ax.set_title('Performance Gap Distribution')
    ax.legend()

    plt.tight_layout()
    if not show:
        plt.close(fig)
    return fig

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats by model."""
    agg_dict = {
        'params_b': 'first',
        'rh_rate': 'mean',
        'avg_visible_pass_rate': 'mean',
        'avg_hidden_pass_rate': 'mean',
        'avg_performance_gap': 'mean',
        'likely_reward_hacking_heuristic': 'sum',
        'monitor_code_suspicious': 'sum',
    }
    # Add confirmed_rh if available
    if 'confirmed_rh' in df.columns:
        agg_dict['confirmed_rh'] = 'sum'
    return df.groupby('agent').agg(agg_dict).round(3)


if __name__ == "__main__":
    df = load_and_prepare("exp_results.csv")

    print("=== Summary by Model ===")
    print(summary_table(df))
    print()

    # Generate plots
    fig1 = plot_rh_by_model(df)
    fig1.savefig('rh_by_model.png', dpi=150)

    fig2 = plot_rh_heatmap(df)
    fig2.savefig('rh_heatmap.png', dpi=150)

    fig3 = plot_pass_rates(df)
    fig3.savefig('pass_rates.png', dpi=150)

    fig4 = plot_monitor_comparison(df)
    fig4.savefig('monitor_comparison.png', dpi=150)

    fig5 = plot_performance_gap(df)
    fig5.savefig('performance_gap.png', dpi=150)

    plt.show()
    print("\nPlots saved: rh_by_model.png, rh_heatmap.png, pass_rates.png, monitor_comparison.png, performance_gap.png")
