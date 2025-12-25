"""
Enhanced Multi-Model Benchmark with Quality and Performance Metrics
Adds: Relevance scoring, completeness, factual accuracy, readability
"""

import json
import time
import requests
from datetime import datetime
from pathlib import Path
import re

# Configuration
API_URL = "http://localhost:8000"

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models to test
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
        "requires_setup": True
    },
    {
        "name": "llama-3.3-70b-versatile",
        "display_name": "Groq Llama 3.3",
        "pricing": {"input": 0.59/1_000_000, "output": 0.79/1_000_000},
        "provider": "groq",
        "description": "Ultra-fast inference",
        "requires_setup": True
    },
]

# Enhanced test queries with expected answer characteristics
TEST_QUERIES = [
    {
        "query": "I'm planning to graduate in May 2026. What courses should I take this semester if I have 15 credits left?",
        "category": "Planning",
        "expected_elements": ["thesis", "non-thesis", "elective", "required", "credit"],
        "complexity": "high"
    },
    {
        "query": "Should I choose the thesis or non-thesis track if I want to pursue a PhD later?",
        "category": "Advice",
        "expected_elements": ["research", "thesis", "PhD", "advisor", "publication"],
        "complexity": "medium"
    },
    {
        "query": "What are the graduation requirements for MSCS?",
        "category": "Factual",
        "expected_elements": ["30", "credits", "GPA", "core", "elective"],
        "complexity": "low"
    },
    {
        "query": "I failed a required course. What are my options and how will this affect my graduation timeline?",
        "category": "Problem-Solving",
        "expected_elements": ["retake", "timeline", "GPA", "probation", "advisor"],
        "complexity": "high"
    },
    {
        "query": "Compare the benefits of taking summer courses versus regular semester",
        "category": "Comparison",
        "expected_elements": ["accelerate", "cost", "intensity", "schedule", "financial aid"],
        "complexity": "medium"
    },
]

def calculate_quality_metrics(answer: str, query_info: dict) -> dict:
    """Calculate various quality metrics for the answer"""
    
    #--------------------------
    #Benchmark Quality Metrics
    #--------------------------

    # 1. Completeness - How many expected elements are mentioned
    expected_elements = query_info.get('expected_elements', [])
    mentioned_elements = sum(1 for elem in expected_elements if elem.lower() in answer.lower())
    completeness_score = (mentioned_elements / len(expected_elements)) if expected_elements else 0
    
    # 2. Readability - Flesch Reading Ease approximation
    sentences = len(re.findall(r'[.!?]+', answer))
    words = len(answer.split())
    syllables = sum(max(1, len(re.findall(r'[aeiou]+', word.lower()))) for word in answer.split())
    
    if sentences > 0 and words > 0:
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        readability = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        readability = max(0, min(100, readability))  
    else:
        readability = 50
    
    # 3. Structure - Has proper paragraphs and organization
    paragraphs = len(answer.split('\n\n'))
    has_structure = paragraphs > 1
    
    # 4. Specificity - Uses numbers, specific terms
    has_numbers = bool(re.search(r'\d+', answer))
    specific_terms = ['specific', 'exactly', 'precisely', 'must', 'required', 'recommended']
    specificity_count = sum(1 for term in specific_terms if term in answer.lower())
    specificity_score = min(1.0, specificity_count / 3)
    
    # 5. Helpfulness indicators
    has_actionable_steps = any(word in answer.lower() for word in ['first', 'second', 'step', 'should', 'need to'])
    has_examples = 'example' in answer.lower() or 'for instance' in answer.lower()
    has_caveats = any(word in answer.lower() for word in ['however', 'although', 'depends', 'may vary'])
    
    # 6. Length appropriateness (based on query complexity)
    complexity = query_info.get('complexity', 'medium')
    ideal_length = {'low': 200, 'medium': 400, 'high': 600}
    actual_length = len(answer)
    length_ratio = actual_length / ideal_length[complexity]
    length_appropriateness = 1.0 - abs(1.0 - length_ratio) if length_ratio < 2 else 0.5
    
    return {
        'completeness_score': round(completeness_score, 3),
        'readability_score': round(readability, 1),
        'has_structure': has_structure,
        'paragraph_count': paragraphs,
        'has_numbers': has_numbers,
        'specificity_score': round(specificity_score, 3),
        'has_actionable_steps': has_actionable_steps,
        'has_examples': has_examples,
        'has_caveats': has_caveats,
        'length_appropriateness': round(length_appropriateness, 3),
        'word_count': words,
        'sentence_count': sentences,
        'avg_sentence_length': round(words / sentences, 1) if sentences > 0 else 0
    }

def test_model_with_query(query_info: dict, model: dict) -> dict:
    """Test a specific model with a query and calculate quality metrics"""
    
    query = query_info['query']
    
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
            answer = data.get('answer', '')
            
            # Calculate basic metrics
            input_tokens = data.get('input_tokens', 0) or 0
            output_tokens = data.get('output_tokens', 0) or 0
            
            cost = (input_tokens * model['pricing']['input'] + 
                   output_tokens * model['pricing']['output'])
            
            # Calculate quality metrics
            quality_metrics = calculate_quality_metrics(answer, query_info)
            
            # Calculate tokens per second (throughput)
            tokens_per_second = (output_tokens / (latency_ms / 1000)) if latency_ms > 0 else 0
            
            return {
                'success': True,
                'latency_ms': latency_ms,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'tokens_per_second': round(tokens_per_second, 1),
                'cost': cost,
                'cost_per_token': cost / (input_tokens + output_tokens) if (input_tokens + output_tokens) > 0 else 0,
                'answer': answer,
                'answer_length': len(answer),
                'model_info': f"{model['display_name']} ({model['provider']})",
                **quality_metrics  # Add all quality metrics
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}",
                'latency_ms': latency_ms
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def run_enhanced_benchmark():
    """Run enhanced benchmark with performance and quality metrics"""
    
    print("\n" + "="*80)
    print("  ENHANCED MULTI-MODEL BENCHMARK WITH PERFORMANCE AND QUALITY METRICS")
    print("="*80)
    print(f"\nTesting {len(MODELS_TO_TEST)} models with {len(TEST_QUERIES)} queries")
    print("\nMetrics tracked:")
    print("    Performance: Latency, Cost, Throughput")
    print("    Quality: Completeness, Readability, Structure")
    print("    Content: Specificity, Actionability, Examples")
    print("\n" + "="*80 + "\n")
    
    all_results = []
    
    for query_idx, query_info in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"Query {query_idx}/{len(TEST_QUERIES)} - Category: {query_info['category']}")
        print(f"Complexity: {query_info['complexity']}")
        print(f"{'='*80}")
        print(f"{query_info['query'][:80]}...")
        print("-" * 80)
        
        query_results = {
            'query': query_info['query'],
            'category': query_info['category'],
            'complexity': query_info['complexity'],
            'query_id': query_idx,
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model in MODELS_TO_TEST:
            print(f"\n Testing {model['display_name']}...", end=" ", flush=True)
            
            result = test_model_with_query(query_info, model)
            query_results['models'][model['name']] = result
            
            if result['success']:
                print(f"Success")
                print(f"     Latency: {result['latency_ms']:.0f}ms ({result['tokens_per_second']:.1f} tok/s)")
                print(f"     Cost: ${result['cost']:.6f} (${result['cost_per_token']:.8f}/token)")
                print(f"     Length: {result['word_count']} words, {result['sentence_count']} sentences")
                print(f"     Quality: Completeness {result['completeness_score']:.0%}, Readability {result['readability_score']:.0f}")
                print(f"     Features: {'Yes' if result['has_actionable_steps'] else 'No'} Steps, "
                      f"{'Yes' if result['has_examples'] else 'No'} Examples, "
                      f"{'Yes' if result['has_structure'] else 'No'} Structure")
            else:
                print(f" Failed: {result.get('error', 'Unknown')[:60]}")
            
            time.sleep(1)
        
        all_results.append(query_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"enhanced_benchmark_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, indent=2, fp=f)
    
    print("\n" + "="*80)
    print(f" Results saved to: {results_file}")
    print("="*80)
    
    # Generate enhanced summary
    generate_enhanced_summary(all_results)
    
    return all_results

def generate_enhanced_summary(results):
    """Generate comprehensive summary with quality metrics"""
    
    print("\n" + "="*80)
    print(" ENHANCED BENCHMARK SUMMARY")
    print("="*80 + "\n")
    
    # Aggregate by model
    model_stats = {}
    
    for query_result in results:
        for model_name, result in query_result['models'].items():
            if result['success']:
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'latencies': [],
                        'costs': [],
                        'tokens_per_second': [],
                        'completeness_scores': [],
                        'readability_scores': [],
                        'specificity_scores': [],
                        'word_counts': [],
                        'has_steps_count': 0,
                        'has_examples_count': 0,
                        'has_structure_count': 0,
                        'total_queries': 0,
                        'display_name': result['model_info']
                    }
                
                stats = model_stats[model_name]
                stats['latencies'].append(result['latency_ms'])
                stats['costs'].append(result['cost'])
                stats['tokens_per_second'].append(result['tokens_per_second'])
                stats['completeness_scores'].append(result['completeness_score'])
                stats['readability_scores'].append(result['readability_score'])
                stats['specificity_scores'].append(result['specificity_score'])
                stats['word_counts'].append(result['word_count'])
                if result['has_actionable_steps']:
                    stats['has_steps_count'] += 1
                if result['has_examples']:
                    stats['has_examples_count'] += 1
                if result['has_structure']:
                    stats['has_structure_count'] += 1
                stats['total_queries'] += 1
    
    # Print comprehensive comparison
    print("PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"{'Model':<25} {'Latency':>12} {'Throughput':>12} {'Total Cost':>12}")
    print("-" * 80)
    
    for model_name, stats in model_stats.items():
        avg_lat = sum(stats['latencies']) / len(stats['latencies'])
        avg_throughput = sum(stats['tokens_per_second']) / len(stats['tokens_per_second'])
        total_cost = sum(stats['costs'])
        
        print(f"{stats['display_name']:<25} {avg_lat:>10.0f}ms {avg_throughput:>9.1f}t/s ${total_cost:>10.6f}")
    
    print("\n" + "="*80)
    print("QUALITY METRICS:")
    print("-" * 80)
    print(f"{'Model':<25} {'Complete':>10} {'Readable':>10} {'Specific':>10} {'Avg Words':>10}")
    print("-" * 80)
    
    for model_name, stats in model_stats.items():
        avg_complete = sum(stats['completeness_scores']) / len(stats['completeness_scores'])
        avg_readable = sum(stats['readability_scores']) / len(stats['readability_scores'])
        avg_specific = sum(stats['specificity_scores']) / len(stats['specificity_scores'])
        avg_words = sum(stats['word_counts']) / len(stats['word_counts'])
        
        print(f"{stats['display_name']:<25} {avg_complete:>9.0%} {avg_readable:>9.1f} {avg_specific:>9.0%} {avg_words:>9.0f}")
    
    print("\n" + "="*80)
    print("CONTENT FEATURES:")
    print("-" * 80)
    print(f"{'Model':<25} {'Has Steps':>12} {'Has Examples':>15} {'Has Structure':>15}")
    print("-" * 80)
    
    for model_name, stats in model_stats.items():
        steps_pct = (stats['has_steps_count'] / stats['total_queries']) * 100
        examples_pct = (stats['has_examples_count'] / stats['total_queries']) * 100
        structure_pct = (stats['has_structure_count'] / stats['total_queries']) * 100
        
        print(f"{stats['display_name']:<25} {steps_pct:>10.0f}% {examples_pct:>13.0f}% {structure_pct:>13.0f}%")
    
    print("\n" + "="*80)
    print("\n CATEGORY WINNERS:\n")
    
    # Determine winners
    fastest = min(model_stats.items(), key=lambda x: sum(x[1]['latencies'])/len(x[1]['latencies']))
    cheapest = min(model_stats.items(), key=lambda x: sum(x[1]['costs']))
    most_complete = max(model_stats.items(), key=lambda x: sum(x[1]['completeness_scores'])/len(x[1]['completeness_scores']))
    most_readable = max(model_stats.items(), key=lambda x: sum(x[1]['readability_scores'])/len(x[1]['readability_scores']))
    fastest_throughput = max(model_stats.items(), key=lambda x: sum(x[1]['tokens_per_second'])/len(x[1]['tokens_per_second']))
    
    print(f"  Fastest Response: {fastest[1]['display_name']}")
    print(f"  Highest Throughput: {fastest_throughput[1]['display_name']}")
    print(f"  Most Cost-Effective: {cheapest[1]['display_name']}")
    print(f"  Most Complete Answers: {most_complete[1]['display_name']}")
    print(f"  Most Readable: {most_readable[1]['display_name']}")
    
    print("\n" + "="*80)

def main():
    try:
        results = run_enhanced_benchmark()
        print("\n  Enhanced benchmark complete!")
        
    except KeyboardInterrupt:
        print("\n\n  Benchmark cancelled")
    except Exception as e:
        print(f"\n\n  Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()