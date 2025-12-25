"""
Interactive test client for NYIT Chatbot
"""

import requests
import json
import time
from typing import Dict

API_BASE_URL = "http://localhost:8000"

def print_separator():
    print("\n" + "="*70 + "\n")

def test_health():
    """Test health endpoint"""
    print("  Testing Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("  Server is healthy!")
            print(f"  Intents loaded: {data['config']['intents_loaded']}")
            print(f"  Documents loaded: {data['config']['documents_loaded']}")
            return True
        else:
            print(f"Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  Cannot connect to server. Is it running?")
        print("  Start with: cd backend && python run.py")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def chat(query: str, include_debug: bool = False) -> Dict:
    """Send chat request"""
    try:
        payload = {
            "query": query,
            "include_debug": include_debug
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=30
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_intent(query: str):
    """Test intent matching only"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/debug/intent",
            params={"query": query},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_rag(query: str, top_k: int = 5):
    """Test RAG retrieval only"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/debug/rag",
            params={"query": query, "top_k": top_k},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def display_response(response: Dict):
    """Pretty print chat response"""
    print("\n" + "─"*70)
    print("  ANSWER:")
    print("─"*70)
    print(response['answer'])
    
    print("\n" + "─"*70)
    print("  METADATA:")
    print("─"*70)
    print(f"Route Used: {response['route_used']}")
    print(f"Confidence: {response['confidence']:.2f}")
    print(f"Latency: {response['latency_ms']:.0f}ms")
    
    if response.get('sources'):
        print("\n  SOURCES:")
        for i, source in enumerate(response['sources'], 1):
            print(f"{i}. {source.get('source_name', 'Unknown')}")
            if 'url' in source:
                print(f"   {source['url']}")
    
    if response.get('follow_up_questions'):
        print("\n  FOLLOW-UP QUESTIONS:")
        for q in response['follow_up_questions']:
            print(f"    {q}")
    
    if response.get('debug_info'):
        print("\n DEBUG INFO:")
        debug = response['debug_info']
        
        if debug.get('intent_matches'):
            print("\nIntent Matches:")
            for match in debug['intent_matches'][:3]:
                print(f"    {match['intent_id']}")
                print(f"    Score: {match['score']:.2f} | Type: {match['match_type']}")
        
        if debug.get('rag_results'):
            print("\nRAG Results:")
            for result in debug['rag_results'][:3]:
                print(f"    {result['title']}")
                print(f"    Score: {result['score']:.2f}")

def run_test_suite():
    """Run predefined test queries"""
    print_separator()
    print(" Running Test Suite")
    print_separator()
    
    test_queries = [
        {
            "query": "How many credits do I need to graduate from MSCS?",
            "category": "Simple FAQ"
        },
        {
            "query": "What is the add/drop deadline for Spring 2026?",
            "category": "Simple FAQ"
        },
        {
            "query": "Can I transfer credits from another university?",
            "category": "Moderate"
        },
        {
            "query": "What's the difference between thesis and non-thesis tracks?",
            "category": "Moderate"
        },
        {
            "query": "I'm planning to graduate in May. What do I need to do?",
            "category": "Complex"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Testing: {test['category']}")
        print(f"Query: '{test['query']}'")
        
        response = chat(test['query'], include_debug=True)
        
        if response:
            print(f" Route: {response['route_used']}")
            print(f" Confidence: {response['confidence']:.2f}")
            print(f" Latency: {response['latency_ms']:.0f}ms")
            
            results.append({
                "query": test['query'],
                "category": test['category'],
                "route": response['route_used'],
                "confidence": response['confidence'],
                "latency_ms": response['latency_ms']
            })
        else:
            print(" Failed")
            results.append({
                "query": test['query'],
                "category": test['category'],
                "status": "failed"
            })
        
        time.sleep(0.5)
    
    # Summary
    print_separator()
    print(" Test Summary")
    print_separator()
    
    if results:
        avg_latency = sum(r.get('latency_ms', 0) for r in results) / len(results)
        avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
        
        print(f"Total Tests: {len(results)}")
        print(f"Average Latency: {avg_latency:.0f}ms")
        print(f"Average Confidence: {avg_confidence:.2f}")
        
        print("\nRoute Distribution:")
        from collections import Counter
        routes = Counter(r.get('route', 'unknown') for r in results)
        for route, count in routes.items():
            print(f"  {route}: {count}")

def interactive_mode():
    """Interactive chat mode"""
    print_separator()
    print(" Interactive Chat Mode")
    print("Type 'quit' to exit, 'debug' to toggle debug mode")
    print_separator()
    
    debug_mode = False
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            
            if query.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue
            
            response = chat(query, include_debug=debug_mode)
            
            if response:
                display_response(response)
            else:
                print("✗ Failed to get response")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

def main():
    """Main menu"""
    print("\n" + "-"*35)
    print("  NYIT Academic Chatbot - Test Client")
    print("-"*35)
    
    # First, check if server is running
    if not test_health():
        return
    
    while True:
        print_separator()
        print("Choose an option:")
        print("1. Interactive chat")
        print("2. Run test suite")
        print("3. Test intent matching")
        print("4. Test RAG retrieval")
        print("5. Check server health")
        print("6. Exit")
        print_separator()
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '1':
            interactive_mode()
        
        elif choice == '2':
            run_test_suite()
        
        elif choice == '3':
            query = input("Enter query: ").strip()
            if query:
                result = test_intent(query)
                if result:
                    print("\n Intent Matches:")
                    for match in result.get('matches', []):
                        print(f"\n• {match['intent_id']}")
                        print(f"  Score: {match['score']:.2f}")
                        print(f"  Type: {match['match_type']}")
                        print(f"  Matched: {match['matched_question']}")
        
        elif choice == '4':
            query = input("Enter query: ").strip()
            if query:
                result = test_rag(query)
                if result:
                    print("\n RAG Results:")
                    for i, doc in enumerate(result.get('results', []), 1):
                        print(f"\n{i}. {doc['title']}")
                        print(f"   Score: {doc['score']:.2f}")
                        print(f"   Text: {doc['text'][:150]}...")
        
        elif choice == '5':
            test_health()
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()