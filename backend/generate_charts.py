"""
Chart Generation Script for AI Academic Advisor Chatbot
Focuses on: Latency, Cost, Route Accuracy
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
RESULTS_DIR = Path("results")
CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

def load_latest_results():
    """Load the most recent benchmark results"""
    result_files = list(RESULTS_DIR.glob("benchmark_*.json"))
    
    if not result_files:
        print("❌ No benchmark results found!")
        print("   Run 'python benchmark_v2.py' first")
        return None
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    
    print(f"📂 Loading: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results

def create_latency_chart(results):
    """Bar chart: Average latency by route"""
    df = pd.DataFrame([r for r in results if 'status' not in r])
    
    if df.empty:
        print("⚠️  No data for latency chart")
        return
    
    stats = df.groupby('route_used')['latency_ms'].agg(['mean', 'std', 'count'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    routes = stats.index
    means = stats['mean']
    stds = stats['std'].fillna(0)
    
    bars = ax.bar(routes, means, yerr=stds, capsize=5, 
                   color=COLORS[:len(routes)], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}ms',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Route', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Response Latency by Route', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = CHARTS_DIR / 'latency_by_route.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")

def create_cost_chart(results):
    """Bar chart: Total cost by route"""
    df = pd.DataFrame([r for r in results if 'status' not in r])
    
    if df.empty:
        print("⚠️  No data for cost chart")
        return
    
    cost_stats = df.groupby('route_used')['cost'].agg(['sum', 'mean', 'count'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    routes = cost_stats.index
    totals = cost_stats['sum']
    
    bars = ax.bar(routes, totals, color=COLORS[:len(routes)], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 0.00001,
                    '$0',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Route', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontsize=13, fontweight='bold')
    ax.set_title('API Cost by Route', fontsize=15, fontweight='bold', pad=20)
    ax.set_yscale('log' if totals.max() > totals.min() * 100 else 'linear')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = CHARTS_DIR / 'cost_by_route.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")

def create_route_distribution(results):
    """Pie chart: Query distribution across routes"""
    df = pd.DataFrame([r for r in results if 'status' not in r])
    
    if df.empty:
        print("⚠️  No data for distribution chart")
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
    
    # Legend with counts
    legend_labels = [f'{route}: {count} queries' 
                    for route, count in route_counts.items()]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
    
    plt.tight_layout()
    save_path = CHARTS_DIR / 'route_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")

def create_route_accuracy_chart(results):
    """Bar chart: Route prediction accuracy"""
    df = pd.DataFrame([r for r in results if 'status' not in r])
    
    if df.empty:
        print("⚠️  No data for accuracy chart")
        return
    
    # Calculate accuracy by expected route
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
    
    bars = ax.bar(acc_df['route'], acc_df['accuracy'],
                   color=COLORS[:len(acc_df)], alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        matched = acc_df.iloc[i]['matched']
        total = acc_df.iloc[i]['total']
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%\n({matched}/{total})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
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
    
    print(f"✓ Saved: {save_path}")

def create_latency_vs_cost_scatter(results):
    """Scatter plot: Cost vs Latency trade-off"""
    df = pd.DataFrame([r for r in results if 'status' not in r])
    
    if df.empty:
        print("⚠️  No data for scatter plot")
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
    
    # Add quadrant labels
    mid_latency = df['latency_ms'].median()
    mid_cost = df['cost'].median()
    ax.axvline(x=mid_latency, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=mid_cost, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = CHARTS_DIR / 'cost_vs_latency.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")

def generate_report(results):
    """Generate markdown report with key findings"""
    df = pd.DataFrame([r for r in results if 'status' not in r])
    
    if df.empty:
        print("⚠️  No data for report")
        return
    
    total = len(results)
    successful = len(df)
    failed = total - successful
    
    # Calculate metrics
    route_matches = df['route_matched'].sum()
    route_accuracy = (route_matches / successful) * 100
    
    avg_latency = df['latency_ms'].mean()
    min_latency = df['latency_ms'].min()
    max_latency = df['latency_ms'].max()
    
    total_cost = df['cost'].sum()
    
    # Route statistics
    route_stats = df.groupby('route_used').agg({
        'latency_ms': ['count', 'mean', 'std', 'min', 'max'],
        'cost': ['sum', 'mean'],
        'confidence': 'mean',
        'route_matched': 'sum'
    })
    
    # Generate report
    report = f"""# AI Academic Advisor Chatbot - Benchmark Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Metric | Value |
|--------|------:|
| **Total Queries** | {total} |
| **Successful** | {successful} |
| **Failed** | {failed} |
| **Route Accuracy** | **{route_accuracy:.1f}%** |
| **Avg Latency** | {avg_latency:.0f}ms |
| **Total Cost** | ${total_cost:.6f} |

---

## Performance Metrics

### Latency Analysis
- **Average:** {avg_latency:.0f}ms
- **Minimum:** {min_latency:.0f}ms
- **Maximum:** {max_latency:.0f}ms
- **Range:** {max_latency - min_latency:.0f}ms

### Cost Analysis
- **Total Cost:** ${total_cost:.6f}
- **LLM Queries:** {len(df[df['route_used'] == 'llm'])}
"""
    
    llm_queries = df[df['route_used'] == 'llm']
    if len(llm_queries) > 0:
        avg_llm_cost = llm_queries['cost'].mean()
        report += f"- **Avg Cost per LLM Query:** ${avg_llm_cost:.6f}\n"
    
    report += f"""
---

## Route Performance

| Route | Queries | % | Avg Latency | Std Dev | Min | Max | Total Cost | Avg Cost | Avg Confidence |
|-------|---------|---|-------------|---------|-----|-----|------------|----------|----------------|
"""
    
    for route in route_stats.index:
        stats = route_stats.loc[route]
        count = int(stats['latency_ms']['count'])
        pct = (count / successful) * 100
        avg_lat = stats['latency_ms']['mean']
        std_lat = stats['latency_ms']['std']
        min_lat = stats['latency_ms']['min']
        max_lat = stats['latency_ms']['max']
        total_cost_route = stats['cost']['sum']
        avg_cost = stats['cost']['mean']
        avg_conf = stats['confidence']['mean']
        
        report += f"| **{route}** | {count} | {pct:.1f}% | {avg_lat:.0f}ms | ±{std_lat:.0f}ms | {min_lat:.0f}ms | {max_lat:.0f}ms | ${total_cost_route:.6f} | ${avg_cost:.6f} | {avg_conf:.2f} |\n"
    
    report += f"""
---

## Visualizations

### Latency Performance
![Latency by Route](charts/latency_by_route.png)

### Cost Analysis
![Cost by Route](charts/cost_by_route.png)

### Route Distribution
![Route Distribution](charts/route_distribution.png)

### Routing Accuracy
![Route Accuracy](charts/route_accuracy.png)

### Performance Trade-off
![Cost vs Latency](charts/cost_vs_latency.png)

---

## Key Findings

1. **Routing Accuracy:** {route_accuracy:.1f}% of queries were correctly routed
   - {route_matches} out of {successful} queries matched expected routes
   
2. **Performance:** Average response time of {avg_latency:.0f}ms
   - Fastest route: {route_stats['latency_ms']['mean'].idxmin()} ({route_stats.loc[route_stats['latency_ms']['mean'].idxmin(), 'latency_ms']['mean']:.0f}ms)
   - Slowest route: {route_stats['latency_ms']['mean'].idxmax()} ({route_stats.loc[route_stats['latency_ms']['mean'].idxmax(), 'latency_ms']['mean']:.0f}ms)

3. **Cost Efficiency:** Total benchmark cost of ${total_cost:.6f}
   - Zero-cost routes (rule_based, rag) handled {len(df[df['cost'] == 0])} queries
   - LLM route used for {len(llm_queries)} complex queries

---

## Recommendations

"""
    
    if route_accuracy >= 80:
        report += "✅ **Excellent routing accuracy!** The intent classifier is performing well.\n\n"
    elif route_accuracy >= 60:
        report += "⚠️ **Good routing accuracy** with room for improvement. Consider refining intent patterns.\n\n"
    else:
        report += "❌ **Routing accuracy needs improvement.** Review intent classifier logic and thresholds.\n\n"
    
    report += f"""
### System Optimization:
1. **Rule-based routing** provides fastest response ({route_stats.loc['rule_based', 'latency_ms']['mean'] if 'rule_based' in route_stats.index else 'N/A':.0f}ms) at zero cost
2. **RAG routing** balances accuracy and cost for policy questions
3. **LLM routing** for complex queries justifies higher cost with better quality

### Next Steps:
- Monitor production usage patterns
- Fine-tune intent thresholds based on user feedback
- Implement caching for frequently asked questions
- Consider response quality metrics in future evaluations

---

## Technical Details

**Test Configuration:**
- Total test queries: {total}
- Query categories: Simple FAQ, Policy/Procedure, Complex Planning
- Routes: rule_based, rag, llm
- Metrics tracked: Latency, Cost, Accuracy

**Evaluation Focus:**
- ✅ Latency (response speed)
- ✅ Cost (API usage)
- ✅ Route Accuracy (correct routing decisions)

---

*Report generated by benchmark system v2.0*
"""
    
    report_path = CHARTS_DIR.parent / 'benchmark_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Saved: {report_path}")

def main():
    """Generate all charts and report"""
    
    print("\n📊 Generating Benchmark Visualizations")
    print("=" * 70)
    
    results = load_latest_results()
    
    if not results:
        return
    
    print(f"\nAnalyzing {len(results)} results...\n")
    
    # Generate all charts
    create_latency_chart(results)
    create_cost_chart(results)
    create_route_distribution(results)
    create_route_accuracy_chart(results)
    create_latency_vs_cost_scatter(results)
    
    # Generate report
    generate_report(results)
    
    print("\n" + "=" * 70)
    print("✅ All visualizations generated!")
    print(f"📁 Charts: {CHARTS_DIR}/")
    print(f"📄 Report: benchmark_report.md")
    print("=" * 70)

if __name__ == "__main__":
    main()