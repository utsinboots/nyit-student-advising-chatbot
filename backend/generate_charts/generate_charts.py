"""
Chart Generation Script for AI Academic Advisor Chatbot
Focuses on: Latency, Cost, Route Accuracy
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
BACKEND_DIR = Path(__file__).resolve().parents[1]  
RESULTS_DIR = BACKEND_DIR / "benchmark_tests" / "results"
CHARTS_DIR = Path(__file__).resolve().parent / "charts_multi_enhanced"
CHARTS_DIR.mkdir(exist_ok=True)
# Style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']


def load_latest_results():
    """Load the most recent benchmark results"""
    result_files = list(RESULTS_DIR.glob("benchmark_*.json"))

    if not result_files:
        print("No benchmark results found.")
        print("Run: python benchmark.py")
        return None

    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file.name}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return results


def create_latency_chart(results):
    """Bar chart: Average latency by route"""
    df = pd.DataFrame([r for r in results if 'status' not in r])

    if df.empty:
        print("No data for latency chart")
        return

    stats = df.groupby('route_used')['latency_ms'].agg(['mean', 'std', 'count'])

    fig, ax = plt.subplots(figsize=(10, 6))

    routes = stats.index
    means = stats['mean']
    stds = stats['std'].fillna(0)

    bars = ax.bar(
        routes, means, yerr=stds, capsize=5,
        color=COLORS[:len(routes)], alpha=0.8,
        edgecolor='black', linewidth=1.5
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height,
            f'{height:.0f}ms',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_xlabel('Route', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Response Latency by Route', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = CHARTS_DIR / 'latency_by_route.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def create_cost_chart(results):
    """Bar chart: Total cost by route"""
    df = pd.DataFrame([r for r in results if 'status' not in r])

    if df.empty:
        print("No data for cost chart")
        return

    cost_stats = df.groupby('route_used')['cost'].agg(['sum', 'mean', 'count'])

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    routes = cost_stats.index
    totals = cost_stats['sum']

    bars = ax.bar(
        routes, totals,
        color=COLORS[:len(routes)], alpha=0.8,
        edgecolor='black', linewidth=1.5
    )

    for bar in bars:
        height = bar.get_height()
        label_y = height if height > 0 else 0.00001
        label_txt = f'${height:.4f}' if height > 0 else '$0'
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, label_y,
            label_txt,
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_xlabel('Route', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontsize=13, fontweight='bold')
    ax.set_title('API Cost by Route', fontsize=15, fontweight='bold', pad=20)
    ax.set_yscale('log' if totals.max() > totals.min() * 100 else 'linear')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = CHARTS_DIR / 'cost_by_route.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def create_route_distribution(results):
    """Pie chart: Query distribution across routes"""
    df = pd.DataFrame([r for r in results if 'status' not in r])

    if df.empty:
        print("No data for distribution chart")
        return

    route_counts = df['route_used'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        route_counts.values,
        labels=route_counts.index,
        autopct='%1.1f%%',
        colors=COLORS[:len(route_counts)],
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')

    ax.set_title('Query Distribution by Route', fontsize=15, fontweight='bold', pad=20)

    legend_labels = [f'{route}: {count} queries' for route, count in route_counts.items()]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)

    plt.tight_layout()
    save_path = CHARTS_DIR / 'route_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def create_route_accuracy_chart(results):
    """Bar chart: Route prediction accuracy"""
    df = pd.DataFrame([r for r in results if 'status' not in r])

    if df.empty:
        print("No data for accuracy chart")
        return

    accuracy_data = []
    for expected in df['expected_route'].unique():
        subset = df[df['expected_route'] == expected]
        matches = subset['route_matched'].sum()
        total = len(subset)
        accuracy = (matches / total) * 100
        accuracy_data.append({
            'route': expected,
            'accuracy': accuracy,
            'matched': matches,
            'total': total
        })

    acc_df = pd.DataFrame(accuracy_data).sort_values('route')

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        acc_df['route'], acc_df['accuracy'],
        color=COLORS[:len(acc_df)], alpha=0.8,
        edgecolor='black', linewidth=1.5
    )

    for i, bar in enumerate(bars):
        height = bar.get_height()
        matched = acc_df.iloc[i]['matched']
        total = acc_df.iloc[i]['total']
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height,
            f'{height:.0f}%\n({matched}/{total})',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    ax.set_xlabel('Expected Route', fontsize=13, fontweight='bold')
    ax.set_ylabel('Routing Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Route Prediction Accuracy', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Accuracy')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_path = CHARTS_DIR / 'route_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def create_latency_vs_cost_scatter(results):
    """Scatter plot: Cost vs Latency trade-off"""
    df = pd.DataFrame([r for r in results if 'status' not in r])

    if df.empty:
        print("No data for scatter plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    routes = df['route_used'].unique()
    route_colors = {route: COLORS[i % len(COLORS)] for i, route in enumerate(sorted(routes))}

    for route in sorted(routes):
        route_data = df[df['route_used'] == route]
        ax.scatter(
            route_data['latency_ms'],
            route_data['cost'],
            label=route,
            color=route_colors[route],
            alpha=0.7,
            s=120,
            edgecolors='black',
            linewidth=1
        )

    ax.set_xlabel('Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cost ($)', fontsize=13, fontweight='bold')
    ax.set_title('Cost vs Latency Trade-off', fontsize=15, fontweight='bold', pad=20)
    ax.legend(title='Route', title_fontsize=11, fontsize=11)
    ax.grid(alpha=0.3)

    mid_latency = df['latency_ms'].median()
    mid_cost = df['cost'].median()
    ax.axvline(x=mid_latency, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=mid_cost, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    save_path = CHARTS_DIR / 'cost_vs_latency.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def main():
    """Generate charts only (no report)."""
    print("\nGenerating Benchmark Visualizations")
    print("=" * 70)

    results = load_latest_results()
    if not results:
        return

    print(f"\nAnalyzing {len(results)} results...\n")

    create_latency_chart(results)
    create_cost_chart(results)
    create_route_distribution(results)
    create_route_accuracy_chart(results)
    create_latency_vs_cost_scatter(results)

    print("\n" + "=" * 70)
    print("All visualizations generated!")
    print(f"Charts saved to: {CHARTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
