import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer

class RAGService:
    """Retrieval-Augmented Generation service"""
    
    def __init__(self, corpus_path: Path, index_path: Optional[Path] = None):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.documents = []
        
        self._load_corpus()
        
        # Use local embeddings (free)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        self.index = None
        self._build_or_load_index()
    
    def _load_corpus(self):
        """Load RAG corpus from JSONL"""
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.documents.append(doc)
    
    def _build_or_load_index(self):
        """Build or load FAISS index"""
        if self.index_path and (self.index_path / "faiss.index").exists():
            self.index = faiss.read_index(str(self.index_path / "faiss.index"))
            print(f"Loaded FAISS index from {self.index_path}")
        else:
            self._build_index()
    
    def _build_index(self):
        """Build FAISS index from scratch"""
        print("Building FAISS index...")
        
        texts = [doc.get('text', '') for doc in self.documents]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Use Inner Product (dot product) instead of L2 for better similarity
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # IP = Inner Product
        self.index.add(embeddings.astype('float32'))
        
        print(f"Built FAISS index with {len(texts)} documents")
        
        if self.index_path:
            self.index_path.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path / "faiss.index"))
            print(f"Saved index to {self.index_path}")
    
    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """
        Retrieve relevant documents using cosine similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1, where 1 is perfect match)
        
        Returns:
            List of matching documents with scores
        """
        # Encode and normalize query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity
        
        # Search returns cosine similarities (since we're using IndexFlatIP with normalized vectors)
        similarities, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], similarities[0]):
            # Score is already cosine similarity (0-1 range)
            # Filter by threshold
            if score < threshold:
                continue
            
            if idx < 0 or idx >= len(self.documents):
                continue
            
            doc = self.documents[idx]
            results.append({
                'doc_id': doc.get('doc_id', ''),
                'title': doc.get('title', ''),
                'text': doc.get('text', ''),
                'score': float(score),
                'source_url': doc.get('source_url', ''),
                'metadata': doc.get('metadata', {})
            })
        
        return results
    
    def get_context_for_llm(self, query: str, top_k: int = 5) -> str:
        """Format context for LLM prompt"""
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}] {result['title']}\n"
                f"{result['text']}\n"
                f"URL: {result['source_url']}\n"
            )
        
        return "\n---\n".join(context_parts)