"""
Multi-Model Benchmark with Performance and Cost Metrics
Compares GPT-4o-mini, Claude Sonnet 4, and Groq Llama 3.3
"""

import json
import time
import requests
from datetime import datetime
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_TO_TEST = [
    {
        "name": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "pricing": {"input": 0.15/1_000_000, "output": 0.60/1_000_000},
        "provider": "openai",
        "description": "Fast and affordable"
    },
    {
        "name": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4",
        "pricing": {"input": 3.00/1_000_000, "output": 15.00/1_000_000},
        "provider": "anthropic",
        "description": "High quality reasoning",
    },
    {
        "name": "llama-3.3-70b-versatile",
        "display_name": "Groq Llama 3.3",
        "pricing": {"input": 0.59/1_000_000, "output": 0.79/1_000_000},
        "provider": "groq",
        "description": "Ultra-fast inference",
    }]

TEST_QUERIES = [
    "I'm planning to graduate in May 2026. What courses should I take this semester if I have 15 credits left?",
    "Should I choose the thesis or non-thesis track if I want to pursue a PhD later?",
    "I failed a required course. What are my options and how will this affect my graduation timeline?",
    "Compare the benefits of taking summer courses versus regular semester for finishing my degree faster",
    "I want to switch my concentration. What's the process and how will it impact my graduation date?",
]

def test_model_with_query(query: str, model: dict) -> dict:
    
    if model.get('requires_setup') and model['provider'] == 'anthropic':
        pass  
    
    try:
        start_time = time.time()

        response = requests.post(
            f"{API_URL}/chat",
            json={
                "query": query,
                "model": model["name"],
                "include_debug": True
            },
            timeout=60
        )
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            
            # Calculate cost
            input_tokens = data.get('input_tokens', 0) or 0
            output_tokens = data.get('output_tokens', 0) or 0
            
            cost = (input_tokens * model['pricing']['input'] + 
                   output_tokens * model['pricing']['output'])
            
            return {
                'success': True,
                'latency_ms': latency_ms,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'cost': cost,
                'answer': data.get('answer', ''),
                'answer_length': len(data.get('answer', '')),
                'model_info': f"{model['display_name']} ({model['provider']})"
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text[:100]}",
                'latency_ms': latency_ms,
                'skipped': model.get('requires_setup', False)
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'skipped': model.get('requires_setup', False)
        }

def run_multi_model_benchmark():
    
    print("\n" + "="*80)
    print(" MULTI-MODEL LLM COMPARISON BENCHMARK")
    print("="*80)
    print(f"\nTesting {len([m for m in MODELS_TO_TEST if not m.get('skipped')])} models")
    print(f"with {len(TEST_QUERIES)} complex queries\n")
    
    print("Models to test:")
    for model in MODELS_TO_TEST:
        status = "  Requires setup" if model.get('requires_setup') else " Ready"
        print(f"  {status} - {model['display_name']} ({model['provider']})")
        print(f"           {model['description']}")
    
    print("\n" + "="*80 + "\n")
    
    all_results = []
    models_tested = {m['name']: False for m in MODELS_TO_TEST}
    
    for query_idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"Query {query_idx}/{len(TEST_QUERIES)}")
        print(f"{'='*80}")
        print(f"{query[:100]}...")
        print("-" * 80)
        
        query_results = {
            'query': query,
            'query_id': query_idx,
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model in MODELS_TO_TEST:
            print(f"\n Testing {model['display_name']}...", end=" ", flush=True)
            
            result = test_model_with_query(query, model)
            query_results['models'][model['name']] = result
            
            if result['success']:
                models_tested[model['name']] = True
                print(f"Yes")
                print(f"   Latency: {result['latency_ms']:.0f}ms")
                print(f"   Tokens: {result['input_tokens']}in + {result['output_tokens']}out = {result['total_tokens']}")
                print(f"   Cost: ${result['cost']:.6f}")
                print(f"   Answer: {result['answer_length']} chars")
            else:
                if result.get('skipped'):
                    print(f"   Skipped (not configured)")
                else:
                    print(f"  Failed")
                    print(f"   Error: {result.get('error', 'Unknown')[:60]}")
            
            time.sleep(1)
        
        all_results.append(query_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"multi_model_benchmark_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, indent=2, fp=f)
    
    print("\n" + "="*80)
    print(f"  Results saved to: {results_file}")
    print("="*80)

    generate_summary(all_results, models_tested)
    
    return all_results

def generate_summary(results, models_tested):
    """Generate summary comparison"""
    
    print("\n" + "="*80)
    print("  BENCHMARK SUMMARY")
    print("="*80 + "\n")
    
    working_models = [name for name, tested in models_tested.items() if tested]
    
    if not working_models:
        print("  No models successfully tested!")
        print("\nPossible issues:")
        print("  - Claude API not set up (need ANTHROPIC_API_KEY)")
        print("  - OpenAI API issues")
        print("  - Server not running")
        return
    
    print(f" Successfully tested {len(working_models)} models:\n")
    
    model_stats = {}
    
    for query_result in results:
        for model_name, result in query_result['models'].items():
            if result['success']:
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'latencies': [],
                        'costs': [],
                        'tokens': [],
                        'answer_lengths': [],
                        'display_name': result['model_info']
                    }
                
                stats = model_stats[model_name]
                stats['latencies'].append(result['latency_ms'])
                stats['costs'].append(result['cost'])
                stats['tokens'].append(result['total_tokens'])
                stats['answer_lengths'].append(result['answer_length'])
    
    print(f"{'Model':<25} {'Avg Latency':>12} {'Avg Cost':>12} {'Total Cost':>12} {'Avg Tokens':>12}")
    print("-" * 80)
    
    for model_name, stats in model_stats.items():
        display = stats['display_name']
        avg_lat = sum(stats['latencies']) / len(stats['latencies'])
        avg_cost = sum(stats['costs']) / len(stats['costs'])
        total_cost = sum(stats['costs'])
        avg_tokens = sum(stats['tokens']) / len(stats['tokens'])
        
        print(f"{display:<25} {avg_lat:>10.0f}ms ${avg_cost:>10.6f} ${total_cost:>10.6f} {avg_tokens:>10.0f}")
    
    print("\n" + "="*80)
    
    # Winner 
    if len(model_stats) > 1:
        print("\n WINNERS BY CATEGORY:\n")
        
        fastest = min(model_stats.items(), key=lambda x: sum(x[1]['latencies'])/len(x[1]['latencies']))
        cheapest = min(model_stats.items(), key=lambda x: sum(x[1]['costs']))
        most_detailed = max(model_stats.items(), key=lambda x: sum(x[1]['answer_lengths'])/len(x[1]['answer_lengths']))
        
        print(f" Fastest Response: {fastest[1]['display_name']}")
        print(f" Most Cost-Effective: {cheapest[1]['display_name']}")
        print(f" Most Detailed Answers: {most_detailed[1]['display_name']}")
    
    print("\n" + "="*80)
    
    if len(model_stats) == 1:
        print("  Only one model tested. To compare:")
        print("   1. Set up Claude API (see CLAUDE_API_SETUP.md)")
        print("   2. Restart server")
        print("   3. Run benchmark again")
    else:
        print(" Multi-model comparison complete!")
    
    print("\n" + "="*80)

def main():
    try:
        print("\n Starting Multi-Model Benchmark...")
        print("\nNote: This will test multiple AI models with the same queries")
        print("      to compare performance, cost, and quality.\n")
        
        results = run_multi_model_benchmark()
        
    except KeyboardInterrupt:
        print("\n\n Benchmark cancelled by user")
    except Exception as e:
        print(f"\n\n Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()