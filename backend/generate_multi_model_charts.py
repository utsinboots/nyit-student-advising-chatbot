"""
Multi-Model Comparison Chart Generator
Creates professional comparison charts for multi-model benchmarks
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
from datetime import datetime

# Configuration
RESULTS_DIR = Path("results")
CHARTS_DIR = Path("charts_multi_model")
CHARTS_DIR.mkdir(exist_ok=True)

# Professional color scheme for each provider
PROVIDER_COLORS = {
    'openai': '#10A37F',      # OpenAI green
    'anthropic': '#D97757',   # Anthropic orange
    'groq': '#6366F1',        # Groq purple/blue
}

MODEL_COLORS = {
    'gpt-4o-mini': '#10A37F',
    'gpt-4o': '#0E8A6D',
    'claude-sonnet-4-20250514': '#D97757',
    'claude-3-5-sonnet-20241022': '#C66647',
    'llama-3.3-70b-versatile': '#6366F1',
    'mixtral-8x7b-32768': '#5558E3',
}

def load_latest_multi_model_results():
    """Load the most recent multi-model benchmark results"""
    result_files = list(RESULTS_DIR.glob("multi_model_benchmark_*.json"))
    
    if not result_files:
        print("❌ No multi-model benchmark results found!")
        print("   Run 'python benchmark_multi_model_verified.py' first")
        return None
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading: {latest_file.name}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

def extract_model_stats(results):
    """Extract statistics for each model"""
    model_stats = {}
    
    for query_result in results:
        for model_name, result in query_result['models'].items():
            if result.get('success'):
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'latencies': [],
                        'costs': [],
                        'tokens_in': [],
                        'tokens_out': [],
                        'total_tokens': [],
                        'answer_lengths': [],
                        'display_name': result.get('model_info', model_name),
                        'provider': 'openai' if 'gpt' in model_name else 
                                   'anthropic' if 'claude' in model_name else 
                                   'groq' if 'llama' in model_name or 'mixtral' in model_name else 'unknown'
                    }
                
                stats = model_stats[model_name]
                stats['latencies'].append(result['latency_ms'])
                stats['costs'].append(result['cost'])
                stats['tokens_in'].append(result['input_tokens'])
                stats['tokens_out'].append(result['output_tokens'])
                stats['total_tokens'].append(result['total_tokens'])
                stats['answer_lengths'].append(result['answer_length'])
    
    return model_stats

def generate_latency_comparison(model_stats):
    """Chart 1: Latency comparison across models"""
    plt.figure(figsize=(12, 7))
    
    models = list(model_stats.keys())
    display_names = [model_stats[m]['display_name'] for m in models]
    avg_latencies = [np.mean(model_stats[m]['latencies']) for m in models]
    colors = [MODEL_COLORS.get(m, '#666666') for m in models]
    
    bars = plt.bar(range(len(models)), avg_latencies, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_latencies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:.0f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Average Response Time (ms)', fontsize=12, fontweight='bold')
    plt.title('Response Latency Comparison Across Models', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(models)), display_names, rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add speedup annotations
    if len(avg_latencies) > 1:
        fastest_idx = np.argmin(avg_latencies)
        fastest_time = avg_latencies[fastest_idx]
        
        for i, lat in enumerate(avg_latencies):
            if i != fastest_idx:
                speedup = lat / fastest_time
                plt.text(i, lat/2, f'{speedup:.1f}x slower', ha='center', 
                        fontsize=9, style='italic', color='white', fontweight='bold')
    
    plt.tight_layout()
    chart_path = CHARTS_DIR / 'latency_comparison.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {chart_path}")

def generate_cost_comparison(model_stats):
    """Chart 2: Cost comparison across models"""
    plt.figure(figsize=(12, 7))
    
    models = list(model_stats.keys())
    display_names = [model_stats[m]['display_name'] for m in models]
    avg_costs = [np.mean(model_stats[m]['costs']) for m in models]
    total_costs = [np.sum(model_stats[m]['costs']) for m in models]
    colors = [MODEL_COLORS.get(m, '#666666') for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, avg_costs, width, label='Avg Cost per Query', 
                    color=colors, alpha=0.8, edgecolor='black')
    bars2 = plt.bar(x + width/2, total_costs, width, label='Total Cost (All Queries)', 
                    color=colors, alpha=0.5, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                    f'${height:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Cost (USD)', fontsize=12, fontweight='bold')
    plt.title('API Cost Comparison Across Models', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, display_names, rotation=15, ha='right')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    chart_path = CHARTS_DIR / 'cost_comparison.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {chart_path}")

def generate_cost_per_1k_tokens(model_stats):
    """Chart 3: Cost efficiency (cost per 1000 tokens)"""
    plt.figure(figsize=(12, 7))
    
    models = list(model_stats.keys())
    display_names = [model_stats[m]['display_name'] for m in models]
    
    cost_per_1k = []
    for m in models:
        total_cost = np.sum(model_stats[m]['costs'])
        total_tokens = np.sum(model_stats[m]['total_tokens'])
        cost_per_1k.append((total_cost / total_tokens) * 1000 if total_tokens > 0 else 0)
    
    colors = [MODEL_COLORS.get(m, '#666666') for m in models]
    
    bars = plt.bar(range(len(models)), cost_per_1k, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, cost_per_1k):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
                f'${val:.5f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Cost per 1,000 Tokens (USD)', fontsize=12, fontweight='bold')
    plt.title('Cost Efficiency: Price per 1,000 Tokens', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(models)), display_names, rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    chart_path = CHARTS_DIR / 'cost_per_1k_tokens.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {chart_path}")

def generate_tokens_comparison(model_stats):
    """Chart 4: Token usage comparison"""
    plt.figure(figsize=(14, 7))
    
    models = list(model_stats.keys())
    display_names = [model_stats[m]['display_name'] for m in models]
    
    avg_input = [np.mean(model_stats[m]['tokens_in']) for m in models]
    avg_output = [np.mean(model_stats[m]['tokens_out']) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    colors = [MODEL_COLORS.get(m, '#666666') for m in models]
    
    bars1 = plt.bar(x - width/2, avg_input, width, label='Input Tokens', 
                    color=colors, alpha=0.6, edgecolor='black')
    bars2 = plt.bar(x + width/2, avg_output, width, label='Output Tokens', 
                    color=colors, alpha=0.9, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Average Tokens', fontsize=12, fontweight='bold')
    plt.title('Token Usage Comparison: Input vs Output', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, display_names, rotation=15, ha='right')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    chart_path = CHARTS_DIR / 'tokens_comparison.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {chart_path}")

def generate_scatter_cost_vs_latency(model_stats):
    """Chart 5: Scatter plot - Cost vs Latency trade-off"""
    plt.figure(figsize=(12, 8))
    
    for model_name, stats in model_stats.items():
        avg_latency = np.mean(stats['latencies'])
        avg_cost = np.mean(stats['costs'])
        color = MODEL_COLORS.get(model_name, '#666666')
        
        plt.scatter(avg_latency, avg_cost * 1000, s=300, alpha=0.7, 
                   color=color, edgecolors='black', linewidth=2)
        
        # Add model label
        plt.annotate(stats['display_name'], 
                    (avg_latency, avg_cost * 1000),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))
    
    plt.xlabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('Average Cost per Query (USD × 1000)', fontsize=12, fontweight='bold')
    plt.title('Cost vs Speed Trade-off', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add quadrant labels
    plt.text(0.02, 0.98, 'Fast & Cheap\n(Ideal)', transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    plt.text(0.98, 0.98, 'Slow & Cheap', transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    plt.text(0.02, 0.02, 'Fast & Expensive', transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    plt.text(0.98, 0.02, 'Slow & Expensive\n(Avoid)', transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.tight_layout()
    chart_path = CHARTS_DIR / 'cost_vs_latency_scatter.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {chart_path}")

def generate_answer_quality_proxy(model_stats):
    """Chart 6: Answer length as quality proxy"""
    plt.figure(figsize=(12, 7))
    
    models = list(model_stats.keys())
    display_names = [model_stats[m]['display_name'] for m in models]
    avg_lengths = [np.mean(model_stats[m]['answer_lengths']) for m in models]
    colors = [MODEL_COLORS.get(m, '#666666') for m in models]
    
    bars = plt.bar(range(len(models)), avg_lengths, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, avg_lengths):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(val)} chars', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Average Answer Length (characters)', fontsize=12, fontweight='bold')
    plt.title('Answer Verbosity Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(models)), display_names, rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    chart_path = CHARTS_DIR / 'answer_length_comparison.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {chart_path}")

def generate_performance_radar(model_stats):
    """Chart 7: Radar chart comparing multiple dimensions"""
    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['Speed\n(inverted latency)', 'Cost\nEfficiency', 'Verbosity']
    N = len(categories)
    
    # Compute angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    # Normalize metrics for radar chart (0-1 scale)
    for model_name, stats in model_stats.items():
        # Speed: invert latency and normalize
        avg_latency = np.mean(stats['latencies'])
        speed_score = 1 / avg_latency * 10000  # Normalize
        
        # Cost efficiency: invert cost and normalize
        avg_cost = np.mean(stats['costs'])
        cost_score = 1 / (avg_cost * 1000 + 0.0001)  # Normalize
        
        # Verbosity
        avg_length = np.mean(stats['answer_lengths'])
        verbosity_score = avg_length / 1000  # Normalize
        
        # Scale to 0-1 range
        values = [speed_score, cost_score, verbosity_score]
        max_val = max(values) if max(values) > 0 else 1
        values = [v / max_val for v in values]
        values += values[:1]
        
        color = MODEL_COLORS.get(model_name, '#666666')
        ax.plot(angles, values, 'o-', linewidth=2, label=stats['display_name'], color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Multi-Dimensional Performance Comparison', fontsize=14, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    chart_path = CHARTS_DIR / 'performance_radar.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {chart_path}")

def generate_summary_table_chart(model_stats):
    """Chart 8: Summary comparison table as image"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    models = list(model_stats.keys())
    table_data = []
    
    for model in models:
        stats = model_stats[model]
        row = [
            stats['display_name'],
            f"{np.mean(stats['latencies']):.0f}ms",
            f"${np.mean(stats['costs']):.5f}",
            f"${np.sum(stats['costs']):.5f}",
            f"{int(np.mean(stats['total_tokens']))}",
            f"{int(np.mean(stats['answer_lengths']))}",
        ]
        table_data.append(row)
    
    headers = ['Model', 'Avg Latency', 'Avg Cost', 'Total Cost', 'Avg Tokens', 'Avg Length']
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by provider
    for i, model in enumerate(models):
        color = MODEL_COLORS.get(model, '#666666')
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    plt.title('Multi-Model Benchmark Summary', fontsize=16, fontweight='bold', pad=20)
    
    chart_path = CHARTS_DIR / 'summary_table.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {chart_path}")

def main():
    print("\n" + "="*70)
    print("📊 Multi-Model Comparison Chart Generator")
    print("="*70 + "\n")
    
    results = load_latest_multi_model_results()
    
    if not results:
        return
    
    print(f"Analyzing {len(results)} queries across multiple models...")
    
    model_stats = extract_model_stats(results)
    
    if not model_stats:
        print("❌ No successful model results found!")
        return
    
    print(f"\n✓ Found {len(model_stats)} models with results\n")
    print("Generating comparison charts...\n")
    
    # Generate all charts
    generate_latency_comparison(model_stats)
    generate_cost_comparison(model_stats)
    generate_cost_per_1k_tokens(model_stats)
    generate_tokens_comparison(model_stats)
    generate_scatter_cost_vs_latency(model_stats)
    generate_answer_quality_proxy(model_stats)
    generate_performance_radar(model_stats)
    generate_summary_table_chart(model_stats)
    
    print("\n" + "="*70)
    print(f"✅ All charts saved to: {CHARTS_DIR}/")
    print("="*70)
    
    print("\n📊 Generated Charts:")
    print("  1. latency_comparison.png - Response speed comparison")
    print("  2. cost_comparison.png - API cost comparison")
    print("  3. cost_per_1k_tokens.png - Cost efficiency")
    print("  4. tokens_comparison.png - Token usage patterns")
    print("  5. cost_vs_latency_scatter.png - Trade-off analysis")
    print("  6. answer_length_comparison.png - Verbosity comparison")
    print("  7. performance_radar.png - Multi-dimensional view")
    print("  8. summary_table.png - Complete summary table")
    
    print("\n✨ Use these charts in your capstone presentation!")

if __name__ == "__main__":
    main()