from typing import Dict, List, Optional
from enum import Enum


class RouteDecision(str, Enum):
    """Simplified routing for capstone project"""
    RULE_BASED = "rule_based"  # Simple FAQs from curated KB
    RAG = "rag"                # Policy/procedure with retrieval
    LLM = "llm"                # Complex reasoning (may use RAG internally)


class QueryRouter:
    """
    Improved query router that checks both intent and RAG before deciding.
    
    Routes queries to one of three modules:
    - rule_based: High-confidence pattern match for simple FAQs
    - rag: Retrieval-based answers for policies/procedures
    - llm: LLM reasoning for complex queries (can use RAG context)
    """

    def __init__(
        self,
        intent_threshold: float = 0.85,
        rag_threshold: float = 0.55,
        intent_strong_threshold: float = 0.95, 
    ):
        self.intent_threshold = intent_threshold
        self.rag_threshold = rag_threshold
        self.intent_strong_threshold = intent_strong_threshold

    def route_query(
        self, 
        query: str, 
        intent_matches: List[Dict], 
        rag_results: List[Dict]
    ) -> Dict:
        """
        Determine how to handle the query.
        
        Improved logic: Checks both intent AND RAG, then decides which is better.
        
        Returns:
            Dict with route decision, confidence, and data to use
        """
        
        top_intent_score = intent_matches[0]["score"] if intent_matches else 0.0
        top_rag_score = rag_results[0]["score"] if rag_results else 0.0

        has_strong_intent = top_intent_score >= self.intent_strong_threshold
        has_good_intent = top_intent_score >= self.intent_threshold
        has_good_rag = top_rag_score >= self.rag_threshold
        is_complex = self._is_complex_query(query)

        # Decision Logic - IMPROVED
        
        # Route 1: VERY strong intent (≥0.95) + simple → RULE_BASED
        # This catches only the most confident matches
        if has_strong_intent and not is_complex:
            return {
                "route": RouteDecision.RULE_BASED,
                "confidence": top_intent_score,
                "reasoning": "Very high-confidence intent match for simple FAQ",
                "intent_data": intent_matches[0] if intent_matches else None,
                "rag_data": None,
                # Debug info
                "intent_score": top_intent_score,
                "rag_score": top_rag_score,
                "is_complex": is_complex,
            }

        # Route 2: Good RAG (≥0.70) + not complex → RAG
        # Check RAG BEFORE checking weaker intent matches
        if has_good_rag and not is_complex:
            # If we also have good intent, choose based on which score is higher
            if has_good_intent:
                # Both intent and RAG are good, prefer RAG if scores are close
                if top_rag_score >= (top_intent_score - 0.1):
                    return {
                        "route": RouteDecision.RAG,
                        "confidence": top_rag_score,
                        "reasoning": "Good RAG results for policy/procedure query (chose RAG over intent)",
                        "intent_data": None,
                        "rag_data": rag_results,
                        # Debug info
                        "intent_score": top_intent_score,
                        "rag_score": top_rag_score,
                        "is_complex": is_complex,
                    }
                else:
                    # Intent score is significantly higher, use rule_based
                    return {
                        "route": RouteDecision.RULE_BASED,
                        "confidence": top_intent_score,
                        "reasoning": "Intent score significantly higher than RAG",
                        "intent_data": intent_matches[0] if intent_matches else None,
                        "rag_data": None,
                        # Debug info
                        "intent_score": top_intent_score,
                        "rag_score": top_rag_score,
                        "is_complex": is_complex,
                    }
            else:
                # Only RAG is good, use it
                return {
                    "route": RouteDecision.RAG,
                    "confidence": top_rag_score,
                    "reasoning": "Good retrieval results for policy/procedure query",
                    "intent_data": None,
                    "rag_data": rag_results,
                    # Debug info
                    "intent_score": top_intent_score,
                    "rag_score": top_rag_score,
                    "is_complex": is_complex,
                }

        # Route 3: Good intent (≥0.85 but <0.95) + not complex → RULE_BASED
        if has_good_intent and not is_complex:
            return {
                "route": RouteDecision.RULE_BASED,
                "confidence": top_intent_score,
                "reasoning": "Good intent match for simple FAQ",
                "intent_data": intent_matches[0] if intent_matches else None,
                "rag_data": None,
                # Debug info
                "intent_score": top_intent_score,
                "rag_score": top_rag_score,
                "is_complex": is_complex,
            }

        # Route 4: Complex query or fallback → LLM
        # Note: LLM can use RAG context internally if available
        confidence = max(top_rag_score, 0.5) if has_good_rag else 0.5
        
        return {
            "route": RouteDecision.LLM,
            "confidence": confidence,
            "reasoning": "Complex query requiring LLM reasoning" if is_complex else "Fallback to LLM (no strong intent or RAG match)",
            "intent_data": intent_matches[0] if intent_matches else None,
            "rag_data": rag_results if has_good_rag else None,
            # Debug info
            "intent_score": top_intent_score,
            "rag_score": top_rag_score,
            "is_complex": is_complex,
        }

    def _is_complex_query(self, query: str) -> bool:
        """
        Detect if query requires complex reasoning.
        
        Complex queries involve:
        - Planning, comparison, or multi-step reasoning
        - Hypothetical scenarios
        - Personalized advice
        """
        q = query.strip().lower()

        # Keywords indicating complexity
        complex_keywords = [
            # Reasoning & comparison
            "explain", "why", "how does", "compare",
            "difference between", "which should i",
            "pros and cons", "tradeoff", "better",
            
            # Decision making
            "should i", "recommend", "advise",
            "best option", "what if", "depending on",
            
            # Planning
            "planning", "timeline", "schedule",
            "in my situation", "my case",
            
            # Multiple parts
            "and also", "as well as", "in addition",
        ]

        for keyword in complex_keywords:
            if keyword in q:
                return True

        # Multi-part questions
        if q.count("?") > 1:
            return True
        
        # Long questions often complex
        if len(q.split()) > 15:
            return True

        # Multiple clauses with conjunctions
        conjunctions = ["and", "or", "but", "if", "unless", "while", "when"]
        conjunction_count = sum(q.split().count(c) for c in conjunctions)
        if conjunction_count >= 2:
            return True

        return False