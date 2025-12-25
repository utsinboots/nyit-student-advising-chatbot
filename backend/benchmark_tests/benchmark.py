"""
Benchmark Script - Uses Questions from JSONL Files
"""

import json
import time
import random
import requests
from datetime import datetime
from pathlib import Path

# -----------------------
# Configuration
# -----------------------
API_URL = "http://localhost:8000"

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

INTENTS_FILE = DATA_DIR / "nyit_advising_kb_intents_expanded.jsonl"
RAG_CORPUS_FILE = DATA_DIR / "nyit_rag_corpus_expanded.jsonl"

print(f"DEBUG: benchmark.py location : {Path(__file__).resolve()}")
print(f"DEBUG: PROJECT_ROOT          : {PROJECT_ROOT}")
print(f"DEBUG: DATA_DIR              : {DATA_DIR}")
print(f"DEBUG: INTENTS_FILE exists   : {INTENTS_FILE.exists()}  -> {INTENTS_FILE}")
print(f"DEBUG: RAG_CORPUS exists     : {RAG_CORPUS_FILE.exists()}  -> {RAG_CORPUS_FILE}")
print(f"DEBUG: RESULTS_DIR           : {RESULTS_DIR}")
print()


PRICING = {
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
    "gpt-4o": {"input": 0.0025 / 1000, "output": 0.01 / 1000},
    "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
    "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
    "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
}
COUNT_LLM_WITH_RAG_AS_RAG = False


def print_separator():
    print("=" * 70)


def normalize_route(route_used_value: str) -> tuple[str, str]:
    """
    Returns (route_used_normalized, route_used_raw)
    Normalized is one of: rule_based, rag, llm, unknown, or a cleaned raw string.
    """
    raw = (route_used_value or "unknown").strip()
    low = raw.lower()

    # Normalize common variants
    if low in ("rule_based", "rulebased", "rule", "intent", "intent_only"):
        return "rule_based", raw

    if low in ("rag_only", "rag"):
        return "rag", raw

    if low in ("llm_only", "llm"):
        return "llm", raw

    if low in ("hybrid",):
        return "rag", raw

    if low in ("llm_with_rag", "rag_with_llm", "llm+rag", "llm_rag", "rag_llm"):
        if COUNT_LLM_WITH_RAG_AS_RAG:
            return "rag", raw
        return "llm", raw

    return low if low else "unknown", raw


def calculate_cost(route_used: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate API cost (only for llm route)."""
    if route_used in ("rule_based", "rag"):
        return 0.0

    if route_used == "llm":
        pricing = PRICING.get(model) or PRICING["gpt-4o-mini"]
        return input_tokens * pricing["input"] + output_tokens * pricing["output"]

    return 0.0


def chat_with_bot(query: str, include_debug: bool = False) -> dict | None:
    """Send query to chatbot API and measure latency."""
    try:
        start_time = time.time()
        resp = requests.post(
            f"{API_URL}/chat",
            json={"query": query, "include_debug": include_debug},
            timeout=30,
        )
        latency_ms = (time.time() - start_time) * 1000

        if resp.status_code != 200:
            print(f"Error: HTTP {resp.status_code} - {resp.text}")
            return None

        data = resp.json()
        data["latency_ms"] = latency_ms
        return data

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# -----------------------
# Query generation
# -----------------------
def load_intent_questions(num_samples: int = 12) -> list[dict]:
    """Load questions from intent JSONL file."""
    questions: list[dict] = []

    if not INTENTS_FILE.exists():
        print(f"Intent file not found: {INTENTS_FILE}")
        return []

    print(f"Reading intents from: {INTENTS_FILE}")

    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                intent = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_num}")
                continue

            questions_list = intent.get("questions", [])
            if questions_list and isinstance(questions_list, list):
                for q in questions_list:
                    if isinstance(q, str) and len(q.strip()) > 5:
                        questions.append(
                            {
                                "query": q.strip(),
                                "category": "Simple FAQ",
                                "expected_route": "rule_based",
                                "source": "intents_jsonl",
                                "intent_id": intent.get("id", f"line_{line_num}"),
                                "domain": intent.get("domain", "unknown"),
                                "topic": intent.get("topic", "unknown"),
                            }
                        )

    print(f"Found {len(questions)} intent-based questions")

    if len(questions) > num_samples:
        questions = random.sample(questions, num_samples)

    return questions


def load_rag_queries(num_samples: int = 10) -> list[dict]:
    """
    Generate RAG queries from corpus titles/topics
    """
    queries: list[dict] = []

    if not RAG_CORPUS_FILE.exists():
        print(f"RAG corpus file not found: {RAG_CORPUS_FILE}")
        return []

    print(f"Reading RAG corpus from: {RAG_CORPUS_FILE}")

    with open(RAG_CORPUS_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = doc.get("title", "")
            metadata = doc.get("metadata", {}) or {}
            topic = metadata.get("topic", "general")

            if not isinstance(title, str) or not title.strip():
                continue

            templates = [
                f"According to NYIT policy, what is the procedure for {topic}?",
                f"What are the official NYIT rules and requirements related to {topic}?",
                f"Can you summarize the NYIT policy details for {topic} with key steps?",
                f"What deadlines, forms, or steps are involved for {topic} at NYIT?",
                f"Explain the process for {topic} and any conditions or restrictions.",
            ]

            query = random.choice(templates)

            queries.append(
                {
                    "query": query,
                    "category": "Policy/Procedure",
                    "expected_route": "rag",
                    "source": "rag_corpus",
                    "doc_id": doc.get("doc_id", f"line_{line_num}"),
                    "title": title,
                    "topic": topic,
                }
            )

    print(f"Found {len(queries)} RAG-based queries")

    if len(queries) > num_samples:
        queries = random.sample(queries, num_samples)

    return queries


def create_complex_queries() -> list[dict]:
    """Create complex queries that should use LLM."""
    return [
        {
            "query": "I'm planning to graduate next semester. What courses should I take if I have 12 credits left?",
            "category": "Complex Planning",
            "expected_route": "llm",
            "source": "synthetic",
        },
        {
            "query": "Should I choose the thesis or non-thesis track if I want to pursue a PhD later?",
            "category": "Complex Planning",
            "expected_route": "llm",
            "source": "synthetic",
        },
        {
            "query": "I failed a required course. What are my options and how will this affect my timeline?",
            "category": "Complex Planning",
            "expected_route": "llm",
            "source": "synthetic",
        },
        {
            "query": "Compare the benefits of taking summer courses versus regular semester for finishing faster.",
            "category": "Complex Planning",
            "expected_route": "llm",
            "source": "synthetic",
        },
    ]


def generate_test_queries(verbose: bool = True) -> list[dict]:
    """Generate test queries from actual data files + a few complex synthetic queries."""
    print("\nLoading test queries from data files...")
    intent_queries = load_intent_questions(num_samples=12)
    rag_queries = load_rag_queries(num_samples=10)
    complex_queries = create_complex_queries()

    all_queries = intent_queries + rag_queries + complex_queries

    print(f"Loaded {len(intent_queries)} queries from intents JSONL (expected rule_based)")
    print(f"Generated {len(rag_queries)} queries from RAG corpus (expected rag)")
    print(f"Created {len(complex_queries)} complex queries (expected llm)")
    print(f"Total: {len(all_queries)} test queries\n")

    if verbose and all_queries:
        print("Sample queries by category:")
        for category in ["Simple FAQ", "Policy/Procedure", "Complex Planning"]:
            cat = [q for q in all_queries if q["category"] == category]
            if cat:
                print(f"\n{category} (expected: {cat[0]['expected_route']}):")
                for q in cat[:3]:
                    print(f"  - {q['query'][:90]}...")

    return all_queries


# -----------------------
# Benchmark runner
# -----------------------
def print_summary(results: list[dict]):
    print("\nBenchmark Summary")
    print_separator()

    successful = [r for r in results if "status" not in r]
    failed = [r for r in results if "status" in r]

    if not successful:
        print("No successful queries.")
        return

    print(f"Total Queries: {len(results)}")
    print(f"Successful   : {len(successful)}")
    print(f"Failed       : {len(failed)}\n")

    # Route distribution
    routes: dict[str, int] = {}
    for r in successful:
        routes[r["route_used"]] = routes.get(r["route_used"], 0) + 1

    print("Route Distribution (normalized route_used):")
    for route, count in sorted(routes.items()):
        pct = (count / len(successful)) * 100
        print(f"  {route:12s}: {count:2d} ({pct:5.1f}%)")
    print()

    # Route accuracy
    matches = sum(1 for r in successful if r.get("route_matched"))
    accuracy = (matches / len(successful)) * 100
    print(f"Route Accuracy: {accuracy:.1f}% ({matches}/{len(successful)})\n")

    # Cost
    total_cost = sum(r.get("cost", 0) for r in successful)
    print(f"Total Cost: ${total_cost:.6f}")

    llm_rows = [r for r in successful if r["route_used"] == "llm"]
    if llm_rows:
        avg_llm_cost = sum(r.get("cost", 0) for r in llm_rows) / len(llm_rows)
        print(f"LLM Queries: {len(llm_rows)} | Avg LLM cost: ${avg_llm_cost:.6f}")

    print_separator()


def run_benchmark():
    print_separator()
    print("Starting Benchmark (Using Actual JSONL Data)")
    print_separator()

    test_queries = generate_test_queries(verbose=True)
    if not test_queries:
        print("No test queries generated! Check your JSONL files.")
        return []

    print(f"\nTotal queries: {len(test_queries)}")
    print(f"API endpoint: {API_URL}")
    print("Metrics: Latency, Cost, Route Accuracy")
    print_separator()

    results: list[dict] = []

    for i, test in enumerate(test_queries, start=1):
        print(f"\n[{i}/{len(test_queries)}] Testing: {test['category']}")
        print(f"Query: '{test['query']}'")
        print(f"Expected route: {test['expected_route']}")
        print(f"Source: {test.get('source', 'unknown')}")

        response = chat_with_bot(test["query"], include_debug=False)

        if response:
            # Normalize route
            route_used, route_used_raw = normalize_route(response.get("route_used", "unknown"))

            # tokens
            model = response.get("model", "gpt-4o-mini")
            input_tokens = int(response.get("input_tokens") or 0)
            output_tokens = int(response.get("output_tokens") or 0)

            # Estimate if missing but LLM
            if route_used == "llm" and (input_tokens == 0 or output_tokens == 0):
                input_tokens = int(len(test["query"].split()) * 1.3)
                output_tokens = int(len(response.get("answer", "").split()) * 1.3)

            cost = calculate_cost(route_used, model, input_tokens, output_tokens)
            route_matched = (route_used == test["expected_route"])

            result = {
                "timestamp": datetime.now().isoformat(),
                "query": test["query"],
                "category": test["category"],
                "expected_route": test["expected_route"],
                "route_used": route_used,          # normalized
                "route_used_raw": route_used_raw,  # raw from backend
                "route_matched": route_matched,
                "confidence": response.get("confidence", 0),
                "latency_ms": response.get("latency_ms", 0),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "answer_length": len(response.get("answer", "")),
                "answer_preview": response.get("answer", "")[:100] + "...",
                "source": test.get("source", "unknown"),
            }

            results.append(result)

            match_text = "YES" if route_matched else "NO"
            print(f"{match_text} Route: {route_used} (raw: {route_used_raw})"
                  + (f" | Expected: {test['expected_route']}" if not route_matched else ""))
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Latency: {result['latency_ms']:.0f}ms")
            if cost > 0:
                print(f"  Tokens: {input_tokens} in + {output_tokens} out")
                print(f"  Cost: ${cost:.6f}")
            else:
                print("  Cost: $0 (no API call)")

        else:
            print("Failed to get response")
            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "query": test["query"],
                    "category": test["category"],
                    "expected_route": test["expected_route"],
                    "status": "failed",
                    "source": test.get("source", "unknown"),
                }
            )

        time.sleep(0.5)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_separator()
    print(f"Results saved to: {results_file}")
    print_separator()

    print_summary(results)
    return results


def main():
    try:
        run_benchmark()
        print("\nBenchmark complete!")
    except KeyboardInterrupt:
        print("\nBenchmark cancelled by user")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
