from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import time
from pathlib import Path

from app.config import settings
from app.services.intent_matcher import IntentMatcher
from app.services.rag_service import RAGService
from app.services.router import QueryRouter, RouteDecision
from app.services.llm_service import LLMService

app = FastAPI(
    title="NYIT Academic Advisor Chatbot",
    description="Hybrid AI chatbot for academic advising",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    include_debug: bool = False

class ChatResponse(BaseModel):
    answer: str
    route_used: str
    confidence: float
    sources: List[Dict] = []
    follow_up_questions: List[str] = []
    debug_info: Optional[Dict] = None
    latency_ms: float
    # Added fields for cost tracking
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

intent_matcher = None
rag_service = None
query_router = None
llm_service = None

@app.on_event("startup")
async def startup_event():
    global intent_matcher, rag_service, query_router, llm_service
    
    print("Initializing services...")
    
    intents_path = settings.DATA_DIR / "nyit_advising_kb_intents_expanded.jsonl"
    intent_matcher = IntentMatcher(intents_path)
    print(f"  Loaded {len(intent_matcher.intents_db)} intents")
    
    corpus_path = settings.DATA_DIR / "nyit_rag_corpus_expanded.jsonl"
    index_path = settings.INDEX_DIR / "rag_index"
    rag_service = RAGService(corpus_path=corpus_path, index_path=index_path)
    print(f"  Loaded {len(rag_service.documents)} documents")
    
    query_router = QueryRouter(
        intent_threshold=settings.INTENT_SIMILARITY_THRESHOLD,
        rag_threshold=settings.RAG_SIMILARITY_THRESHOLD
    )
    print("  Initialized query router")
    
    llm_service = LLMService()
    print("  Initialized LLM service")
    
    print("Services ready!")

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "NYIT Academic Advisor Chatbot",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "intent_matcher": intent_matcher is not None,
            "rag_service": rag_service is not None,
            "query_router": query_router is not None,
            "llm_service": llm_service is not None
        },
        "config": {
            "intents_loaded": len(intent_matcher.intents_db) if intent_matcher else 0,
            "documents_loaded": len(rag_service.documents) if rag_service else 0,
            "openai_configured": bool(settings.OPENAI_API_KEY)
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    
    try:
        intent_matches = intent_matcher.match_intent(
            request.query,
            threshold=settings.INTENT_SIMILARITY_THRESHOLD,
            top_k=settings.MAX_INTENT_CANDIDATES
        )
        
        rag_results = rag_service.retrieve(
            request.query,
            top_k=settings.RAG_TOP_K,
            threshold=settings.RAG_SIMILARITY_THRESHOLD
        )
        
        route_decision = query_router.route_query(request.query, intent_matches, rag_results)
        response_data = await generate_response(request.query, route_decision, intent_matches, rag_results)
        
        latency_ms = (time.time() - start_time) * 1000
        
        response = ChatResponse(
            answer=response_data['answer'],
            route_used=route_decision['route'],
            confidence=route_decision['confidence'],
            sources=response_data.get('sources', []),
            follow_up_questions=response_data.get('follow_up', []),
            latency_ms=latency_ms,
            model=response_data.get('model'),
            input_tokens=response_data.get('input_tokens'),
            output_tokens=response_data.get('output_tokens')
        )
        
        if request.include_debug:
            response.debug_info = {
                'intent_matches': intent_matches,
                'rag_results': [
                    {'doc_id': r['doc_id'], 'score': r['score'], 'title': r['title']}
                    for r in rag_results
                ],
                'route_decision': route_decision,
                'llm_metadata': response_data.get('llm_metadata')
            }
        
        return response
        
    except Exception as e:
        import traceback
        print(f"Error in /chat endpoint: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def generate_response(query: str, route_decision: Dict, intent_matches: List[Dict], rag_results: List[Dict]) -> Dict:
    """
    Generate response based on simplified 3-route system:
    - RULE_BASED: Use curated intent answers
    - RAG: Use retrieval-only answers
    - LLM: Use LLM with optional RAG context
    """
    route = route_decision['route']
    
    # Route 1: RULE_BASED - High-confidence intent match
    if route == RouteDecision.RULE_BASED:
        intent = intent_matches[0]['intent_data']
        answer_data = intent_matcher.get_answer(intent['id'])
        
        return {
            'answer': answer_data['long_answer'] or answer_data['short_answer'],
            'sources': answer_data.get('citations', []),
            'follow_up': answer_data.get('follow_up', [])
        }
    
    # Route 2: RAG - Retrieval-based answer (no LLM)
    elif route == RouteDecision.RAG:
        if not rag_results:
            # Fallback if no RAG results
            return {
                'answer': "I couldn't find specific information about that. Please contact NYIT office or your advisor for answers.",
                'sources': [],
                'follow_up': []
            }
        
        # Combine top RAG results into an answer
        answer_parts = []
        sources = []
        
        for i, result in enumerate(rag_results[:3], 1):
            answer_parts.append(f"{result['text']}")
            sources.append({
                'source_name': result['title'], 
                'url': result.get('source_url', '')
            })
        
        return {
            'answer': "\n\n".join(answer_parts),
            'sources': sources,
            'follow_up': []
        }
    
    # Route 3: LLM - Complex reasoning with optional RAG context
    elif route == RouteDecision.LLM:
        # Build context from RAG if available
        context = ""
        if rag_results and route_decision.get('rag_data'):
            context = rag_service.get_context_for_llm(query, top_k=3)
        
        try:
            llm_response = llm_service.generate_response(
                query=query,
                context=context,
                model=settings.CHAT_MODEL,
                max_tokens=500,
                temperature=0.7
            )
            
            sources = []
            if rag_results:
                sources = [
                    {'source_name': r['title'], 'url': r.get('source_url', '')}
                    for r in rag_results[:3]
                ]
            
            return {
                'answer': llm_response['answer'],
                'sources': sources,
                'follow_up': [],
                'model': llm_response.get('model_used', settings.CHAT_MODEL),
                'input_tokens': llm_response.get('tokens_used', {}).get('prompt_tokens', 0),
                'output_tokens': llm_response.get('tokens_used', {}).get('completion_tokens', 0),
                'llm_metadata': {
                    'model': llm_response.get('model_used'),
                    'tokens': llm_response.get('tokens_used'),
                    'cost': llm_response.get('cost'),
                    'latency_ms': llm_response.get('latency_ms')
                }
            }
            
        except Exception as e:
            # Fallback to RAG if LLM fails
            print(f"LLM error: {e}")
            import traceback
            traceback.print_exc()
            
            if rag_results:
                return {
                    'answer': f"I encountered an issue with the AI service. Here's information from our knowledge base:\n\n{rag_results[0]['text']}",
                    'sources': [{'source_name': rag_results[0]['title'], 'url': rag_results[0].get('source_url', '')}],
                    'follow_up': []
                }
            else:
                return {
                    'answer': "I'm having trouble processing your request right now. Please contact NYIT office or your advisor for answers.",
                    'sources': [],
                    'follow_up': []
                }
    
    else:
        # Unknown route 
        return {
            'answer': "I couldn't determine how to best answer your question. Please contact your advisor.",
            'sources': [],
            'follow_up': []
        }

@app.post("/debug/intent")
async def debug_intent(query: str):
    matches = intent_matcher.match_intent(query)
    return {"query": query, "matches": matches}

@app.post("/debug/rag")
async def debug_rag(query: str, top_k: int = 5):
    results = rag_service.retrieve(query, top_k)
    return {"query": query, "results": results}

@app.get("/stats/llm")
async def llm_usage_stats():
    """Get LLM usage statistics"""
    if not llm_service:
        return {"error": "LLM service not initialized"}
    
    return llm_service.get_usage_stats()