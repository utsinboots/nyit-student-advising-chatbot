"""
Benchmark Script - Uses Actual Questions from JSONL Files
Automatically generates test queries from your knowledge base
"""

import json
import time
import requests
from datetime import datetime
from pathlib import Path
import random

# Configuration
API_URL = "http://localhost:8000"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Paths to your data files
# The benchmark.py is in backend/, but data is in project root
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from backend/ to project root
DATA_DIR = PROJECT_ROOT / "data"

INTENTS_FILE = DATA_DIR / "nyit_advising_kb_intents_expanded.jsonl"
RAG_CORPUS_FILE = DATA_DIR / "nyit_rag_corpus_expanded.jsonl"

print(f"DEBUG: Looking for data in: {DATA_DIR.absolute()}")

# OpenAI API Pricing
PRICING = {
    "gpt-4o-mini": {
        "input": 0.00015 / 1000,
        "output": 0.0006 / 1000,
    },
    "gpt-3.5-turbo": {
        "input": 0.0005 / 1000,
        "output": 0.0015 / 1000,
    },
    "gpt-4": {
        "input": 0.03 / 1000,
        "output": 0.06 / 1000,
    },
    "gpt-4-turbo": {
        "input": 0.01 / 1000,
        "output": 0.03 / 1000,
    },
    "gpt-4o": {
        "input": 0.0025 / 1000,
        "output": 0.01 / 1000,
    }
}

def load_intent_questions(num_samples=10):
    """Load questions from intent JSONL file"""
    questions = []
    
    if not INTENTS_FILE.exists():
        print(f"⚠️  Intent file not found: {INTENTS_FILE}")
        print(f"   Looking for: {INTENTS_FILE.absolute()}")
        return []
    
    print(f"📖 Reading intents from: {INTENTS_FILE}")
    
    with open(INTENTS_FILE, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            line_num += 1
            if line.strip():
                try:
                    intent = json.loads(line)
                    
                    # Your JSONL has 'questions' (array) not 'question' (string)
                    questions_list = intent.get('questions', [])
                    
                    if questions_list and isinstance(questions_list, list):
                        # Add all questions from this intent
                        for q in questions_list:
                            if q and isinstance(q, str) and len(q) > 5:
                                questions.append({
                                    "query": q,
                                    "category": "Simple FAQ",
                                    "expected_route": "rule_based",
                                    "source": "intents_jsonl",
                                    "intent_id": intent.get('id', f'line_{line_num}'),
                                    "domain": intent.get('domain', 'unknown'),
                                    "topic": intent.get('topic', 'unknown')
                                })
                    
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Skipping invalid JSON on line {line_num}")
                    continue
                except Exception as e:
                    print(f"   ⚠️  Error on line {line_num}: {e}")
                    continue
    
    print(f"   Found {len(questions)} intent-based questions")
    
    # Sample random questions if we have too many
    if len(questions) > num_samples:
        questions = random.sample(questions, num_samples)
    
    return questions

def load_rag_queries(num_samples=8):
    """Generate queries from RAG corpus titles/topics"""
    queries = []
    
    if not RAG_CORPUS_FILE.exists():
        print(f"⚠️  RAG corpus file not found: {RAG_CORPUS_FILE}")
        print(f"   Looking for: {RAG_CORPUS_FILE.absolute()}")
        return []
    
    print(f"📖 Reading RAG corpus from: {RAG_CORPUS_FILE}")
    
    with open(RAG_CORPUS_FILE, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            line_num += 1
            if line.strip():
                try:
                    doc = json.loads(line)
                    
                    # Your structure has: title, text, doc_id, source_url
                    title = doc.get('title', '')
                    text = doc.get('text', '')
                    doc_id = doc.get('doc_id', f'line_{line_num}')
                    metadata = doc.get('metadata', {})
                    topic = metadata.get('topic', 'general')
                    
                    if not title or not isinstance(title, str):
                        continue
                    
                    # Extract key phrases from title for better questions
                    # Example: "MSCS overview: total credits and track options"
                    title_lower = title.lower()
                    
                    # Create contextually relevant questions based on title
                    if 'credit' in title_lower or 'requirement' in title_lower:
                        question_templates = [
                            f"What are the credit requirements for {topic}?",
                            f"How many credits do I need for {topic}?",
                            f"Tell me about {topic} credit requirements",
                        ]
                    elif 'policy' in title_lower or 'procedure' in title_lower:
                        question_templates = [
                            f"What is the policy for {topic}?",
                            f"Explain the {topic} policy",
                            f"What are the rules for {topic}?",
                        ]
                    elif 'overview' in title_lower or 'introduction' in title_lower:
                        question_templates = [
                            f"Tell me about {topic}",
                            f"What is {topic}?",
                            f"Give me an overview of {topic}",
                        ]
                    elif 'track' in title_lower or 'option' in title_lower:
                        question_templates = [
                            f"What are the {topic} track options?",
                            f"What tracks are available for {topic}?",
                            f"Explain {topic} tracks",
                        ]
                    else:
                        # Generic questions for any title
                        question_templates = [
                            f"What information is available about {title[:50]}?",
                            f"Tell me about {title[:50]}",
                            f"Explain {title[:50]}",
                        ]
                    
                    # Pick a random template
                    query = random.choice(question_templates)
                    
                    queries.append({
                        "query": query,
                        "category": "Policy/Procedure",
                        "expected_route": "rag",
                        "source": "rag_corpus",
                        "doc_id": doc_id,
                        "title": title,
                        "topic": topic
                    })
                            
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Skipping invalid JSON on line {line_num}")
                    continue
                except Exception as e:
                    print(f"   ⚠️  Error on line {line_num}: {e}")
                    continue
    
    print(f"   Found {len(queries)} RAG-based queries")
    
    # Sample random queries
    if len(queries) > num_samples:
        queries = random.sample(queries, num_samples)
    
    return queries

def create_complex_queries():
    """Create complex queries that should use LLM"""
    return [
        {
            "query": "I'm planning to graduate next semester. What courses should I take if I have 12 credits left?",
            "category": "Complex Planning",
            "expected_route": "llm",
            "source": "synthetic"
        },
        {
            "query": "Should I choose the thesis or non-thesis track if I want to pursue a PhD later?",
            "category": "Complex Planning",
            "expected_route": "llm",
            "source": "synthetic"
        },
        {
            "query": "I failed a required course. What are my options and how will this affect my timeline?",
            "category": "Complex Planning",
            "expected_route": "llm",
            "source": "synthetic"
        },
        {
            "query": "Compare the benefits of taking summer courses versus regular semester for finishing faster",
            "category": "Complex Planning",
            "expected_route": "llm",
            "source": "synthetic"
        },
    ]

def generate_test_queries(verbose=True):
    """Generate test queries from actual data files"""
    print("\n📂 Loading test queries from data files...")
    
    # Load from actual data - increased samples for better coverage
    intent_queries = load_intent_questions(num_samples=12)
    rag_queries = load_rag_queries(num_samples=10)
    complex_queries = create_complex_queries()
    
    # Combine all queries
    all_queries = intent_queries + rag_queries + complex_queries
    
    print(f"✓ Loaded {len(intent_queries)} queries from intents JSONL")
    print(f"✓ Generated {len(rag_queries)} queries from RAG corpus")
    print(f"✓ Created {len(complex_queries)} complex queries")
    print(f"✓ Total: {len(all_queries)} test queries")
    
    if len(all_queries) == 0:
        print("\n❌ ERROR: No test queries generated!")
        print("   Check that your JSONL files exist and have data:")
        print(f"   - {INTENTS_FILE}")
        print(f"   - {RAG_CORPUS_FILE}")
    
    # Show sample queries if verbose
    if verbose and all_queries:
        print("\n📋 Sample queries by category:")
        for category in ["Simple FAQ", "Policy/Procedure", "Complex Planning"]:
            cat_queries = [q for q in all_queries if q['category'] == category]
            if cat_queries:
                print(f"\n  {category} (expected: {cat_queries[0]['expected_route']}):")
                for q in cat_queries[:3]:  # Show first 3
                    print(f"    - {q['query'][:80]}...")
    
    print()
    
    return all_queries

def print_separator():
    """Print a visual separator"""
    print("=" * 70)

def calculate_cost(route: str, model: str = "gpt-4o-mini", 
                  input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Calculate API cost"""
    if route in ["rule_based", "rag"]:
        return 0.0
    
    if route == "llm":
        if model not in PRICING:
            model = "gpt-4o-mini"
        
        input_cost = input_tokens * PRICING[model]["input"]
        output_cost = output_tokens * PRICING[model]["output"]
        return input_cost + output_cost
    
    return 0.0

def chat_with_bot(query: str, include_debug: bool = True) -> dict:
    """Send query to chatbot API"""
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{API_URL}/chat",
            json={"query": query, "include_debug": include_debug},
            timeout=30
        )
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            data['latency_ms'] = latency_ms
            return data
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def run_benchmark():
    """Run the complete benchmark suite"""
    
    print_separator()
    print("🚀 Starting Benchmark (Using Actual JSONL Data)")
    print_separator()
    
    # Generate test queries from actual data
    test_queries = generate_test_queries()
    
    if not test_queries:
        print("❌ No test queries generated! Check your JSONL files.")
        return []
    
    print(f"Total queries: {len(test_queries)}")
    print(f"API endpoint: {API_URL}")
    print(f"Metrics: Latency, Cost, Route Accuracy")
    print_separator()
    
    results = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Testing: {test['category']}")
        print(f"Query: '{test['query']}'")
        print(f"Expected route: {test['expected_route']}")
        print(f"Source: {test.get('source', 'unknown')}")
        
        # Send query to chatbot
        response = chat_with_bot(test['query'], include_debug=True)
        
        if response:
            # Extract route information
            route_used = response.get('route_used', 'unknown')
            
            # Normalize route names
            if route_used == "llm_only" or route_used == "llm_with_rag":
                route_used = "llm"
            elif route_used == "rag_only":
                route_used = "rag"
            elif route_used == "hybrid":
                route_used = "rag"
            
            # Calculate cost
            model = response.get('model', 'gpt-4o-mini')
            input_tokens = response.get('input_tokens') or 0
            output_tokens = response.get('output_tokens') or 0
            
            # If tokens not provided, estimate
            if (input_tokens == 0 or output_tokens == 0) and route_used == "llm":
                input_tokens = int(len(test['query'].split()) * 1.3)
                output_tokens = int(len(response.get('answer', '').split()) * 1.3)
            
            # Ensure tokens are integers
            input_tokens = int(input_tokens) if input_tokens else 0
            output_tokens = int(output_tokens) if output_tokens else 0
            
            cost = calculate_cost(route_used, model, input_tokens, output_tokens)
            
            # Check if route matched
            route_matched = (route_used == test['expected_route'])
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "query": test['query'],
                "category": test['category'],
                "expected_route": test['expected_route'],
                "route_used": route_used,
                "route_matched": route_matched,
                "confidence": response.get('confidence', 0),
                "latency_ms": response.get('latency_ms', 0),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "answer_length": len(response.get('answer', '')),
                "answer_preview": response.get('answer', '')[:100] + "...",
                "source": test.get('source', 'unknown')
            }
            
            results.append(result)
            
            # Print summary
            match_icon = "✓" if route_matched else "✗"
            print(f"{match_icon} Route: {route_used} " + 
                  (f"(Expected: {test['expected_route']})" if not route_matched else ""))
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Latency: {result['latency_ms']:.0f}ms")
            if cost > 0:
                print(f"  Tokens: {input_tokens}in + {output_tokens}out")
                print(f"  Cost: ${cost:.6f}")
            else:
                print(f"  Cost: $0 (no API call)")
        else:
            print("✗ Failed to get response")
            results.append({
                "timestamp": datetime.now().isoformat(),
                "query": test['query'],
                "category": test['category'],
                "expected_route": test['expected_route'],
                "status": "failed",
                "source": test.get('source', 'unknown')
            })
        
        time.sleep(0.5)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"benchmark_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, indent=2, fp=f)
    
    print_separator()
    print(f"✓ Results saved to: {results_file}")
    print_separator()
    
    # Print summary
    print_summary(results)
    
    return results

def print_summary(results: list):
    """Print summary statistics"""
    
    print("\n📊 Benchmark Summary")
    print_separator()
    
    successful = [r for r in results if 'status' not in r]
    failed = [r for r in results if 'status' in r]
    
    if not successful:
        print("❌ No successful queries!")
        return
    
    total = len(results)
    
    print(f"Total Queries: {total}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()
    
    # Route distribution
    routes = {}
    for r in successful:
        route = r['route_used']
        routes[route] = routes.get(route, 0) + 1
    
    print("Route Distribution:")
    for route, count in sorted(routes.items()):
        pct = (count / len(successful)) * 100
        print(f"  {route:12s}: {count:2d} ({pct:5.1f}%)")
    print()
    
    # Route accuracy
    route_matches = sum(1 for r in successful if r.get('route_matched', False))
    accuracy = (route_matches / len(successful)) * 100
    print(f"Route Accuracy: {accuracy:.1f}% ({route_matches}/{len(successful)})")
    print()
    
    # Performance metrics
    avg_latency = sum(r['latency_ms'] for r in successful) / len(successful)
    min_latency = min(r['latency_ms'] for r in successful)
    max_latency = max(r['latency_ms'] for r in successful)
    
    total_cost = sum(r.get('cost', 0) for r in successful)
    llm_queries = [r for r in successful if r['route_used'] == 'llm']
    avg_llm_cost = (sum(r.get('cost', 0) for r in llm_queries) / len(llm_queries)) if llm_queries else 0
    
    print(f"Latency:")
    print(f"  Average: {avg_latency:.0f}ms")
    print(f"  Min: {min_latency:.0f}ms")
    print(f"  Max: {max_latency:.0f}ms")
    print()
    
    print(f"Cost:")
    print(f"  Total: ${total_cost:.6f}")
    if llm_queries:
        print(f"  LLM queries: {len(llm_queries)}")
        print(f"  Avg per LLM query: ${avg_llm_cost:.6f}")
    print()
    
    # Query sources
    print("Query Sources:")
    sources = {}
    for r in successful:
        src = r.get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    
    print_separator()

def main():
    """Main entry point"""
    try:
        results = run_benchmark()
        print("\n✅ Benchmark complete!")
        print("📊 Run 'python generate_charts_v2.py' to create visualizations")
    except KeyboardInterrupt:
        print("\n\n❌ Benchmark cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()