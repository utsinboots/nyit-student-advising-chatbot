"""
Simple Report Generator - Creates markdown report from benchmark results
"""

import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")
OUTPUT_FILE = Path("benchmark_report.md")

def load_latest_results():
    """Load the most recent benchmark results"""
    result_files = list(RESULTS_DIR.glob("benchmark_*.json"))
    
    if not result_files:
        print("No benchmark results found!")
        return None
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file.name}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

def generate_simple_report(results):
    """Generate a clean markdown report"""
    
    # Filter successful results
    successful = [r for r in results if 'status' not in r]
    failed = [r for r in results if 'status' in r]
    
    if not successful:
        return "No successful queries found!"
    
    # Calculate metrics
    total = len(results)
    route_matches = sum(1 for r in successful if r.get('route_matched', False))
    route_accuracy = (route_matches / len(successful)) * 100
    
    avg_latency = sum(r['latency_ms'] for r in successful) / len(successful)
    min_latency = min(r['latency_ms'] for r in successful)
    max_latency = max(r['latency_ms'] for r in successful)
    
    total_cost = sum(r.get('cost', 0) for r in successful)
    
    # Route distribution
    routes = {}
    for r in successful:
        route = r['route_used']
        routes[route] = routes.get(route, 0) + 1
    
    # Generate report
    report = f"""# AI Academic Advisor Chatbot - Benchmark Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Metric | Value |
|--------|------:|
| Total Queries | {total} |
| Successful | {len(successful)} |
| Failed | {len(failed)} |
| Route Accuracy | {route_accuracy:.1f}% |
| Average Latency | {avg_latency:.0f}ms |
| Total Cost | ${total_cost:.6f} |

---

## Route Distribution

"""
    
    for route, count in sorted(routes.items()):
        pct = (count / len(successful)) * 100
        report += f"- **{route}**: {count} queries ({pct:.1f}%)\n"
    
    report += f"""
---

## Performance Metrics

### Latency Analysis
- **Average Response Time:** {avg_latency:.0f}ms
- **Fastest Response:** {min_latency:.0f}ms
- **Slowest Response:** {max_latency:.0f}ms
- **Range:** {max_latency - min_latency:.0f}ms

### Cost Analysis
- **Total Benchmark Cost:** ${total_cost:.6f}
"""
    
    llm_queries = [r for r in successful if r['route_used'] == 'llm']
    if llm_queries:
        avg_llm_cost = sum(r.get('cost', 0) for r in llm_queries) / len(llm_queries)
        report += f"""- **LLM Queries:** {len(llm_queries)}
- **Average Cost per LLM Query:** ${avg_llm_cost:.6f}
"""
    
    report += f"""
### Route Accuracy
- **Correctly Routed:** {route_matches} out of {len(successful)} queries
- **Accuracy Rate:** {route_accuracy:.1f}%

---

## Performance by Route

| Route | Queries | Percentage | Avg Latency | Avg Cost |
|-------|---------|------------|-------------|----------|
"""
    
    for route in sorted(routes.keys()):
        route_data = [r for r in successful if r['route_used'] == route]
        count = len(route_data)
        pct = (count / len(successful)) * 100
        avg_lat = sum(r['latency_ms'] for r in route_data) / count
        avg_cost = sum(r.get('cost', 0) for r in route_data) / count
        
        report += f"| {route} | {count} | {pct:.1f}% | {avg_lat:.0f}ms | ${avg_cost:.6f} |\n"
    
    report += """
---

## Visualizations

The following charts have been generated in the `charts/` folder:

1. **latency_by_route.png** - Shows average response time for each route
2. **cost_by_route.png** - Shows total API costs by route
3. **route_distribution.png** - Pie chart showing query distribution
4. **route_accuracy.png** - Bar chart showing routing accuracy
5. **cost_vs_latency.png** - Scatter plot showing performance trade-offs

---

## Key Findings

"""
    
    # Generate findings based on data
    fastest_route = min(routes.keys(), key=lambda r: sum(d['latency_ms'] for d in successful if d['route_used'] == r) / routes[r])
    slowest_route = max(routes.keys(), key=lambda r: sum(d['latency_ms'] for d in successful if d['route_used'] == r) / routes[r])
    
    report += f"""1. **Performance:** The `{fastest_route}` route is the fastest, while `{slowest_route}` is the slowest
2. **Route Accuracy:** {route_accuracy:.1f}% of queries were routed to their expected module
3. **Cost Efficiency:** Total benchmark cost was ${total_cost:.6f}
4. **Hybrid Advantage:** The system successfully routes queries to appropriate modules

---

## Route Breakdown

### Rule-Based Route
"""
    
    if 'rule_based' in routes:
        rb_queries = [r for r in successful if r['route_used'] == 'rule_based']
        rb_avg_lat = sum(r['latency_ms'] for r in rb_queries) / len(rb_queries)
        report += f"""- Handled {routes['rule_based']} queries ({routes['rule_based']/len(successful)*100:.1f}%)
- Average latency: {rb_avg_lat:.0f}ms
- Cost: $0 (no API calls)
- Best for: Simple FAQs with high-confidence pattern matches
"""
    
    report += "\n### RAG Route\n"
    
    if 'rag' in routes:
        rag_queries = [r for r in successful if r['route_used'] == 'rag']
        rag_avg_lat = sum(r['latency_ms'] for r in rag_queries) / len(rag_queries)
        report += f"""- Handled {routes['rag']} queries ({routes['rag']/len(successful)*100:.1f}%)
- Average latency: {rag_avg_lat:.0f}ms
- Cost: $0 (using local embeddings)
- Best for: Policy and procedure questions requiring context retrieval
"""
    else:
        report += "- No queries routed to RAG in this benchmark\n"
    
    report += "\n### LLM Route\n"
    
    if 'llm' in routes:
        llm_queries = [r for r in successful if r['route_used'] == 'llm']
        llm_avg_lat = sum(r['latency_ms'] for r in llm_queries) / len(llm_queries)
        llm_total_cost = sum(r.get('cost', 0) for r in llm_queries)
        llm_avg_cost = llm_total_cost / len(llm_queries)
        report += f"""- Handled {routes['llm']} queries ({routes['llm']/len(successful)*100:.1f}%)
- Average latency: {llm_avg_lat:.0f}ms
- Total cost: ${llm_total_cost:.6f}
- Average cost per query: ${llm_avg_cost:.6f}
- Best for: Complex reasoning and planning questions
"""
    
    report += """
---

## Recommendations

### System Strengths
"""
    
    if 'rule_based' in routes and routes['rule_based'] > 0:
        report += "- Rule-based routing successfully handles simple FAQs at zero cost and minimal latency\n"
    
    if 'llm' in routes:
        report += "- LLM fallback ensures no query goes unanswered\n"
    
    report += """
### Areas for Improvement
"""
    
    if route_accuracy < 70:
        report += "- Route accuracy could be improved through threshold tuning\n"
    
    if 'rag' not in routes or routes.get('rag', 0) < 5:
        report += "- RAG route underutilized - consider lowering similarity threshold\n"
    
    report += """
### Next Steps
- Continue monitoring production usage patterns
- Fine-tune routing thresholds based on user feedback
- Expand intent database and RAG corpus
- Consider adding response quality metrics

---

## Technical Details

**Test Configuration:**
- Benchmark queries: Loaded from actual JSONL data files
- Intent database: 62 intents
- RAG corpus: 50 documents
- Routing thresholds: Intent >= 0.85, RAG >= 0.70
- LLM model: GPT-4o-mini

**Evaluation Metrics:**
- Latency (response speed in milliseconds)
- Cost (API usage in USD)
- Route Accuracy (percentage of correctly routed queries)

---

*Generated by AI Academic Advisor Chatbot Benchmark System v2.0*
"""
    
    return report

def main():
    print("\n" + "="*70)
    print("Generating Benchmark Report")
    print("="*70 + "\n")
    
    results = load_latest_results()
    
    if not results:
        print("ERROR: No results to process")
        return
    
    print(f"Processing {len(results)} results...")
    
    report = generate_simple_report(results)
    
    # Write report with UTF-8 encoding
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nSuccess! Report saved to: {OUTPUT_FILE}")
    print("="*70)

if __name__ == "__main__":
    main()