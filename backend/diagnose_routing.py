"""
Diagnostic Script - Test Intent Matching and Routing Logic
This helps debug why rule-based queries are routing to LLM
"""

import requests
import json

API_URL = "http://localhost:8000"

# Test queries that SHOULD route to rule_based
TEST_QUERIES = [
    "What are your office hours?",
    "How do I contact advising?",
    "What is the add/drop deadline?",
    "Where is the registrar office located?",
]

def test_intent_matching():
    """Test if intent matcher is finding matches"""
    print("\n" + "="*70)
    print("🔍 TESTING INTENT MATCHING")
    print("="*70)
    
    for query in TEST_QUERIES:
        print(f"\n📝 Query: '{query}'")
        
        try:
            response = requests.post(
                f"{API_URL}/debug/intent",
                params={"query": query}
            )
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                if matches:
                    print(f"✓ Found {len(matches)} intent matches:")
                    for i, match in enumerate(matches[:3], 1):
                        # Get first question from the questions array
                        questions = match['intent_data'].get('questions', [])
                        first_q = questions[0] if questions else 'N/A'
                        print(f"  [{i}] Score: {match['score']:.3f} | Intent: {first_q}")
                else:
                    print("✗ No intent matches found!")
            else:
                print(f"✗ Error: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Request failed: {e}")

def test_routing_decision():
    """Test the actual routing decisions"""
    print("\n" + "="*70)
    print("🔀 TESTING ROUTING DECISIONS")
    print("="*70)
    
    for query in TEST_QUERIES:
        print(f"\n📝 Query: '{query}'")
        
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"query": query, "include_debug": True}
            )
            
            if response.status_code == 200:
                data = response.json()
                debug_info = data.get('debug_info', {})
                route_decision = debug_info.get('route_decision', {})
                
                print(f"Route: {data['route_used']}")
                print(f"Confidence: {data['confidence']:.3f}")
                print(f"Reasoning: {route_decision.get('reasoning', 'N/A')}")
                
                if 'intent_score' in route_decision:
                    print(f"Intent Score: {route_decision['intent_score']:.3f}")
                if 'rag_score' in route_decision:
                    print(f"RAG Score: {route_decision['rag_score']:.3f}")
                if 'is_complex' in route_decision:
                    print(f"Is Complex: {route_decision['is_complex']}")
                
                # Show intent matches
                intent_matches = debug_info.get('intent_matches', [])
                if intent_matches:
                    # Get first question from the questions array
                    questions = intent_matches[0]['intent_data'].get('questions', [])
                    first_q = questions[0] if questions else 'N/A'
                    print(f"Top Intent Match: {first_q} (score: {intent_matches[0]['score']:.3f})")
                else:
                    print("No intent matches!")
                    
            else:
                print(f"✗ Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"✗ Request failed: {e}")

def check_config():
    """Check current configuration"""
    print("\n" + "="*70)
    print("⚙️  CHECKING CONFIGURATION")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            config = data.get('config', {})
            
            print(f"Intents Loaded: {config.get('intents_loaded', 0)}")
            print(f"Documents Loaded: {config.get('documents_loaded', 0)}")
            print(f"OpenAI Configured: {config.get('openai_configured', False)}")
            
            # Try to get router thresholds (may not be exposed)
            print("\nExpected thresholds:")
            print("  Intent threshold: 0.85 (for rule_based route)")
            print("  RAG threshold: 0.70 (for rag route)")
            
        else:
            print(f"✗ Error: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Request failed: {e}")

def main():
    print("\n" + "="*70)
    print("🔧 CHATBOT ROUTING DIAGNOSTIC TOOL")
    print("="*70)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code != 200:
            print(f"\n❌ Server not responding correctly at {API_URL}")
            return
    except:
        print(f"\n❌ Cannot connect to server at {API_URL}")
        print("   Make sure your backend is running!")
        return
    
    print(f"\n✓ Connected to {API_URL}\n")
    
    # Run diagnostics
    check_config()
    test_intent_matching()
    test_routing_decision()
    
    # Print recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    print("""
If queries are routing to LLM instead of rule_based:

1. Intent scores too low (< 0.85):
   - Check if questions in JSONL file match test queries
   - Lower intent_threshold in config/router
   - Add more pattern variations to intents

2. No intent matches found:
   - Verify nyit_advising_kb_intents_expanded.jsonl has these questions
   - Check intent_matcher is loading correctly
   - Inspect the actual questions in your JSONL file

3. Queries marked as "complex":
   - Review _is_complex_query() logic in router.py
   - May need to adjust complexity keywords

Recommended actions:
   a) Check your JSONL file for exact question matches
   b) Try lowering intent_threshold from 0.85 to 0.70
   c) Add variations of common questions to your intents
""")
    
    print("="*70)

if __name__ == "__main__":
    main()