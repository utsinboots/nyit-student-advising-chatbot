import os
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = BASE_DIR / "indexes"
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    


    
    # Model Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-5-mini"
    CHAT_MODEL_ADVANCED: str = "gpt-5"
    CHAT_MODEL_COMPARISON: str = "gpt-4o-mini" 
    
    # Intent Matching
    INTENT_SIMILARITY_THRESHOLD: float = 0.75
    MAX_INTENT_CANDIDATES: int = 3
    
    # RAG Configuration
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.70
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Routing Strategy
    USE_RULE_BASED: bool = True
    USE_INTENT_MATCHING: bool = True
    USE_RAG: bool = True
    USE_LLM_FALLBACK: bool = True
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_VERSION: str = "v1"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Benchmarking
    ENABLE_METRICS: bool = True
    TRACK_LATENCY: bool = True
    TRACK_COSTS: bool = True
    
    # GPT-5 Specific Configuration - NEW: Added these
    REASONING_EFFORT: str = "medium"  # Options: minimal, low, medium, high, xhigh
    VERBOSITY: str = "medium"         # Options: low, medium, high
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # NEW: This allows extra fields in .env without errors

settings = Settings()