"""
Enhanced Multi-Model Chart Generator
Works with benchmark_enhanced_quality.py output
Generates charts for quality metrics + performance
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from pathlib import Path
import numpy as np
from datetime import datetime

# Configuration
BACKEND_DIR = Path(__file__).resolve().parents[1]  
RESULTS_DIR = BACKEND_DIR / "benchmark_tests" / "results"
CHARTS_DIR = Path(__file__).resolve().parent / "charts_multi_enhanced"
CHARTS_DIR.mkdir(exist_ok=True)

MODEL_COLORS = {
    'gpt-4o-mini': '#10A37F',
    'claude-sonnet-4-20250514': '#D97757',
    'llama-3.3-70b-versatile': '#6366F1',
}

def load_latest_enhanced_results():
    """Load the most recent enhanced benchmark results"""
    result_files = list(RESULTS_DIR.glob("enhanced_benchmark_*.json"))
    
    if not result_files:
        print("   No enhanced benchmark results found!")
        print("   Run 'python benchmark_multi_enhanced.py' first")
        return None
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"  Loading: {latest_file.name}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

def extract_enhanced_stats(results):
    """Extract comprehensive statistics for each model"""
    model_stats = {}
    
    for query_result in results:
        for model_name, result in query_result['models'].items():
            if result.get('success'):
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        # Performance metrics
                        'latencies': [],
                        'costs': [],
                        'tokens_per_second': [],
                        'cost_per_token': [],
                        
                        # Quality metrics
                        'completeness_scores': [],
                        'readability_scores': [],
                        'specificity_scores': [],
                        'length_appropriateness': [],
                        
                        # Content features
                        'has_steps': 0,
                        'has_examples': 0,
                        'has_structure': 0,
                        'has_caveats': 0,
                        'total': 0,
                        
                        # Token stats
                        'tokens_in': [],
                        'tokens_out': [],
                        'word_counts': [],
                        
                        # Display info
                        'display_name': result.get('model_info', model_name),
                    }
                
                stats = model_stats[model_name]
                
                # Performance
                stats['latencies'].append(result['latency_ms'])
                stats['costs'].append(result['cost'])
                stats['tokens_per_second'].append(result.get('tokens_per_second', 0))
                stats['cost_per_token'].append(result.get('cost_per_token', 0))
                
                # Quality
                stats['completeness_scores'].append(result.get('completeness_score', 0))
                stats['readability_scores'].append(result.get('readability_score', 0))
                stats['specificity_scores'].append(result.get('specificity_score', 0))
                stats['length_appropriateness'].append(result.get('length_appropriateness', 0))
                
                # Content features
                if result.get('has_actionable_steps'):
                    stats['has_steps'] += 1
                if result.get('has_examples'):
                    stats['has_examples'] += 1
                if result.get('has_structure'):
                    stats['has_structure'] += 1
                if result.get('has_caveats'):
                    stats['has_caveats'] += 1
                stats['total'] += 1
                
                # Tokens
                stats['tokens_in'].append(result.get('input_tokens', 0))
                stats['tokens_out'].append(result.get('output_tokens', 0))
                stats['word_counts'].append(result.get('word_count', 0))
    
    return model_stats

def chart1_throughput_comparison(model_stats):
    """Tokens per second comparison"""
    plt.figure(figsize=(12, 7))
    
    models = list(model_stats.keys())
    display_names = [model_stats[m]['display_name'] for m in models]
    avg_throughput = [np.mean(model_stats[m]['tokens_per_second']) for m in models]
    colors = [MODEL_COLORS.get(m, '#666666') for m in models]
    
    bars = plt.bar(range(len(models)), avg_throughput, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, avg_throughput):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.1f}\ntok/s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.xlabel('Model', fontsize=13, fontweight='bold')
    plt.ylabel('Tokens per Second', fontsize=13, fontweight='bold')
    plt.title('Generation Speed: Tokens per Second', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(range(len(models)), display_names, rotation=15, ha='right', fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight fastest
    fastest_idx = np.argmax(avg_throughput)
    bars[fastest_idx].set_edgecolor('gold')
    bars[fastest_idx].set_linewidth(4)
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '1_throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 1_throughput_comparison.png")

def chart2_quality_radar(model_stats):
    """Radar chart for quality metrics"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    categories = ['Completeness', 'Readability\n(scaled)', 'Specificity', 
                  'Structure', 'Examples', 'Actionability']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    
    for model_name, stats in model_stats.items():
        # Prepare values (normalize to 0-1)
        values = [
            np.mean(stats['completeness_scores']),
            np.mean(stats['readability_scores']) / 100,  # Scale to 0-1
            np.mean(stats['specificity_scores']),
            stats['has_structure'] / stats['total'],
            stats['has_examples'] / stats['total'],
            stats['has_steps'] / stats['total'],
        ]
        values += values[:1]  # Complete the circle
        
        color = MODEL_COLORS.get(model_name, '#666666')
        ax.plot(angles, values, 'o-', linewidth=2.5, label=stats['display_name'], color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9)
    plt.title('Quality Metrics Comparison', fontsize=15, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '2_quality_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 2_quality_radar.png")

def chart3_cost_efficiency(model_stats):
    """Cost per token - fair comparison"""
    plt.figure(figsize=(12, 7))
    
    models = list(model_stats.keys())
    display_names = [model_stats[m]['display_name'] for m in models]
    avg_cost_per_token = [np.mean(model_stats[m]['cost_per_token']) * 1_000_000 for m in models]  # Per million tokens
    colors = [MODEL_COLORS.get(m, '#666666') for m in models]
    
    bars = plt.bar(range(len(models)), avg_cost_per_token, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, avg_cost_per_token):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'${val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.xlabel('Model', fontsize=13, fontweight='bold')
    plt.ylabel('Cost per Million Tokens (USD)', fontsize=13, fontweight='bold')
    plt.title('Cost Efficiency Comparison', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(range(len(models)), display_names, rotation=15, ha='right', fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight cheapest
    cheapest_idx = np.argmin(avg_cost_per_token)
    bars[cheapest_idx].set_edgecolor('gold')
    bars[cheapest_idx].set_linewidth(4)
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '3_cost_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3_cost_efficiency.png")

def chart4_quality_vs_cost_scatter(model_stats):
    """2D scatter: Quality vs Cost"""
    plt.figure(figsize=(12, 9))
    
    for model_name, stats in model_stats.items():
        # Overall quality score (average of quality metrics)
        quality = np.mean([
            np.mean(stats['completeness_scores']),
            np.mean(stats['readability_scores']) / 100,
            np.mean(stats['specificity_scores']),
        ]) * 100
        
        avg_cost = np.mean(stats['costs']) * 1000  # Convert to smaller units
        color = MODEL_COLORS.get(model_name, '#666666')
        
        plt.scatter(avg_cost, quality, s=400, alpha=0.7, color=color, 
                   edgecolors='black', linewidth=2, zorder=10)
        
        plt.annotate(stats['display_name'], 
                    (avg_cost, quality),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3, edgecolor='black'))
    
    plt.xlabel('Average Cost per Query (USD × 1000)', fontsize=13, fontweight='bold')
    plt.ylabel('Overall Quality Score (%)', fontsize=13, fontweight='bold')
    plt.title('Quality vs Cost Trade-off', fontsize=15, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add quadrant labels
    ax = plt.gca()
    plt.text(0.05, 0.95, '  High Quality\nLow Cost\n(IDEAL)', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5, edgecolor='darkgreen'))
    
    plt.text(0.95, 0.05, '  Low Quality\nHigh Cost\n(AVOID)', 
            transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5, edgecolor='darkred'))
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '4_quality_vs_cost.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 4_quality_vs_cost.png")

def chart5_content_features(model_stats):
    """Stacked bar chart: Content features"""
    plt.figure(figsize=(12, 7))
    
    models = list(model_stats.keys())
    display_names = [model_stats[m]['display_name'] for m in models]
    
    # Calculate percentages
    steps_pct = [stats['has_steps'] / stats['total'] * 100 for stats in model_stats.values()]
    examples_pct = [stats['has_examples'] / stats['total'] * 100 for stats in model_stats.values()]
    structure_pct = [stats['has_structure'] / stats['total'] * 100 for stats in model_stats.values()]
    caveats_pct = [stats['has_caveats'] / stats['total'] * 100 for stats in model_stats.values()]
    
    x = np.arange(len(models))
    width = 0.6
    
    # Create stacked bars
    p1 = plt.bar(x, steps_pct, width, label='Has Action Steps', color='#4CAF50', alpha=0.9)
    p2 = plt.bar(x, examples_pct, width, bottom=steps_pct, label='Has Examples', color='#2196F3', alpha=0.9)
    p3 = plt.bar(x, structure_pct, width, bottom=np.array(steps_pct)+np.array(examples_pct), 
                label='Has Structure', color='#FF9800', alpha=0.9)
    p4 = plt.bar(x, caveats_pct, width, 
                bottom=np.array(steps_pct)+np.array(examples_pct)+np.array(structure_pct),
                label='Has Caveats', color='#9C27B0', alpha=0.9)
    
    plt.xlabel('Model', fontsize=13, fontweight='bold')
    plt.ylabel('Percentage of Responses (%)', fontsize=13, fontweight='bold')
    plt.title('Content Features Analysis', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x, display_names, rotation=15, ha='right', fontsize=11)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.ylim(0, 400)  # Stacked can go over 100%
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '5_content_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 5_content_features.png")

def chart6_performance_summary(model_stats):
    """Multi-metric grouped bar chart with color = model, hatch = metric"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    models = list(model_stats.keys())
    display_names = [model_stats[m]['display_name'] for m in models]
    colors = [MODEL_COLORS.get(m, '#666666') for m in models]
    

    # Left: Performance metrics
    max_latency = max(np.mean(stats['latencies']) for stats in model_stats.values())
    max_throughput = max(np.mean(stats['tokens_per_second']) for stats in model_stats.values())
    
    latency_norm = [
        100 - (np.mean(stats['latencies']) / max_latency * 100)
        for stats in model_stats.values()
    ]
    throughput_norm = [
        np.mean(stats['tokens_per_second']) / max_throughput * 100
        for stats in model_stats.values()
    ]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(
        x - width/2, latency_norm, width,
        color=colors, hatch='//', alpha=0.8,
        label='Speed (lower latency = higher)'
    )
    ax1.bar(
        x + width/2, throughput_norm, width,
        color=colors, hatch='\\\\', alpha=0.9,
        label='Throughput'
    )
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Score (0–100)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Custom legend for performance metrics
    perf_legend = [
        Patch(facecolor='white', edgecolor='black', hatch='//',
              label='Speed (lower latency = higher)'),
        Patch(facecolor='white', edgecolor='black', hatch='\\\\',
              label='Throughput')
    ]
    ax1.legend(handles=perf_legend, loc='upper left')
    

    # Right: Quality metrics
    completeness_scores = [np.mean(stats['completeness_scores']) * 100 for stats in model_stats.values()]
    readability_scores = [np.mean(stats['readability_scores']) for stats in model_stats.values()]
    specificity_scores = [np.mean(stats['specificity_scores']) * 100 for stats in model_stats.values()]
    
    width = 0.25
    ax2.bar(
        x - width, completeness_scores, width,
        color=colors, hatch='//', alpha=0.8,
        label='Completeness'
    )
    ax2.bar(
        x, readability_scores, width,
        color=colors, hatch='..', alpha=0.8,
        label='Readability'
    )
    ax2.bar(
        x + width, specificity_scores, width,
        color=colors, hatch='xx', alpha=0.9,
        label='Specificity'
    )
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Quality Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Custom legend for quality metrics
    quality_legend = [
        Patch(facecolor='white', edgecolor='black', hatch='//', label='Completeness'),
        Patch(facecolor='white', edgecolor='black', hatch='..', label='Readability'),
        Patch(facecolor='white', edgecolor='black', hatch='xx', label='Specificity')
    ]
    ax2.legend(handles=quality_legend, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '6_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 6_performance_summary.png")

def chart7_comprehensive_table(model_stats):
    """Summary table with all key metrics"""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    models = list(model_stats.keys())
    table_data = []
    
    for model in models:
        stats = model_stats[model]
        row = [
            stats['display_name'],
            f"{np.mean(stats['latencies']):.0f}ms",
            f"{np.mean(stats['tokens_per_second']):.1f}",
            f"${np.mean(stats['costs']):.5f}",
            f"{np.mean(stats['completeness_scores']):.0%}",
            f"{np.mean(stats['readability_scores']):.0f}",
            f"{np.mean(stats['specificity_scores']):.0%}",
            f"{stats['has_steps']/stats['total']:.0%}",
        ]
        table_data.append(row)
    
    headers = ['Model', 'Latency', 'Tok/s', 'Cost', 'Complete', 'Readable', 'Specific', 'Steps']
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                    colWidths=[0.20, 0.12, 0.10, 0.12, 0.12, 0.12, 0.12, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    for i, model in enumerate(models):
        color = MODEL_COLORS.get(model, '#666666')
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
            table[(i+1, j)].set_text_props(weight='bold')
    
    plt.title('Comprehensive Model Comparison', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(CHARTS_DIR / '7_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 7_summary_table.png")

def main():
    print("\n" + "="*70)
    print("  Enhanced Multi-Model Chart Generator")
    print("="*70 + "\n")
    
    results = load_latest_enhanced_results()
    
    if not results:
        return
    
    print(f"Analyzing {len(results)} queries with quality metrics...")
    
    model_stats = extract_enhanced_stats(results)
    
    if not model_stats:
        print("  No successful model results found!")
        return
    
    print(f"\nFound {len(model_stats)} models with results\n")
    print("Generating enhanced comparison charts...\n")
    
    # Generate all enhanced charts
    chart1_throughput_comparison(model_stats)
    chart2_quality_radar(model_stats)
    chart3_cost_efficiency(model_stats)
    chart4_quality_vs_cost_scatter(model_stats)
    chart5_content_features(model_stats)
    chart6_performance_summary(model_stats)
    chart7_comprehensive_table(model_stats)
    
    print("\n" + "="*70)
    print(f"  All enhanced charts saved to: {CHARTS_DIR}/")
    print("="*70)
    
    print("\n  Generated Charts:")
    print("  1. throughput_comparison.png - Generation speed (tok/s)")
    print("  2. quality_radar.png - Multi-dimensional quality view")
    print("  3. cost_efficiency.png - Cost per million tokens")
    print("  4. quality_vs_cost.png - Trade-off analysis")
    print("  5. content_features.png - Feature breakdown")
    print("  6. performance_summary.png - Performance + quality side-by-side")
    print("  7. summary_table.png - Complete comparison table")
    
    print("\nKey insights to highlight:")
    
    # Print winners
    fastest_throughput = max(model_stats.items(), 
                            key=lambda x: np.mean(x[1]['tokens_per_second']))
    cheapest = min(model_stats.items(), 
                  key=lambda x: np.mean(x[1]['cost_per_token']))
    highest_quality = max(model_stats.items(),
                         key=lambda x: np.mean(x[1]['completeness_scores']))
    
    print(f" Fastest generation: {fastest_throughput[1]['display_name']}")
    print(f" Most cost-effective: {cheapest[1]['display_name']}")
    print(f" Highest quality: {highest_quality[1]['display_name']}")

if __name__ == "__main__":
    main()