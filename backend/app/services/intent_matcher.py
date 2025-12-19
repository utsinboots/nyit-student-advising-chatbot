import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class IntentMatcher:
    """Handles intent classification through keyword and TF-IDF matching"""
    
    def __init__(self, intents_path: Path):
        self.intents_path = intents_path
        self.intents_db = []
        self.questions_list = []
        self.intent_ids = []
        
        self._load_intents()
        
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        self.question_vectors = self.vectorizer.fit_transform(self.questions_list)
        
    def _load_intents(self):
        """Load intents from JSONL file"""
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            for line in f:
                intent = json.loads(line)
                self.intents_db.append(intent)
                
                for q in intent.get('questions', []):
                    self.questions_list.append(q.lower())
                    self.intent_ids.append(intent['id'])
    
    def match_intent(self, query: str, threshold: float = 0.75, top_k: int = 3) -> List[Dict]:
        """Match user query to intents"""
        query_lower = query.lower()
        
        # Step 1: Keyword matching
        exact_matches = self._keyword_match(query_lower)
        if exact_matches:
            return exact_matches[:top_k]
        
        # Step 2: TF-IDF matching
        tfidf_matches = self._tfidf_match(query, threshold, top_k)
        return tfidf_matches
    
    def _keyword_match(self, query: str) -> List[Dict]:
        """Rule-based keyword matching"""
        matches = []
        
        keyword_patterns = {
            'graduation_credits': ['total credits', 'how many credits', 'credits to graduate', 'mscs credits'],
            'add_drop': ['add drop', 'add/drop', 'drop class', 'add class', 'drop deadline'],
            'withdrawal': ['withdraw', 'withdrawal', 'drop after add/drop', 'w grade'],
            'thesis_option': ['thesis', 'non-thesis', 'thesis option'],
            'holds': ['hold', 'holds', 'to dos', 'tasks', 'hub hold'],
        }
        
        for intent_pattern, keywords in keyword_patterns.items():
            for keyword in keywords:
                if keyword in query:
                    for intent in self.intents_db:
                        if intent_pattern in intent.get('intent', ''):
                            matches.append({
                                'intent_id': intent['id'],
                                'score': 1.0,
                                'matched_question': keyword,
                                'intent_data': intent,
                                'match_type': 'keyword'
                            })
        
        return matches
    
    def _tfidf_match(self, query: str, threshold: float, top_k: int) -> List[Dict]:
        """TF-IDF similarity matching"""
        query_vec = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vec, self.question_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]
        
        matches = []
        seen_intents = set()
        
        for idx in top_indices:
            score = similarities[idx]
            if score < threshold:
                continue
            
            intent_id = self.intent_ids[idx]
            if intent_id in seen_intents:
                continue
            
            seen_intents.add(intent_id)
            intent_data = next((i for i in self.intents_db if i['id'] == intent_id), None)
            
            if intent_data:
                matches.append({
                    'intent_id': intent_id,
                    'score': float(score),
                    'matched_question': self.questions_list[idx],
                    'intent_data': intent_data,
                    'match_type': 'tfidf'
                })
            
            if len(matches) >= top_k:
                break
        
        return matches
    
    def get_intent_by_id(self, intent_id: str) -> Optional[Dict]:
        """Retrieve intent by ID"""
        return next((i for i in self.intents_db if i['id'] == intent_id), None)
    
    def get_answer(self, intent_id: str) -> Dict:
        """Get pre-defined answer for intent"""
        intent = self.get_intent_by_id(intent_id)
        if not intent:
            return {}
        
        return {
            'short_answer': intent.get('answer', {}).get('short', ''),
            'long_answer': intent.get('answer', {}).get('long', ''),
            'citations': intent.get('answer', {}).get('citations', []),
            'follow_up': intent.get('answer', {}).get('follow_up_questions', [])
        }