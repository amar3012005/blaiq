"""
RAG Service Configuration with Multi-Provider LLM Support

Updated to support multiple LLM providers: Gemini, OpenAI, OpenRouter, Claude, Ollama.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """
    Configuration for RAG microservice with multi-provider LLM support.
    
    Core Attributes:
        knowledge_base_path: Knowledge base directory path (required)
        llm_provider: LLM provider name ("gemini", "openai", "openrouter", "claude", "ollama")
        llm_api_key: API key for the LLM provider
        llm_model: Model identifier for the provider
        
    Vector Store:
        vector_store_path: Vector store directory (default: /app/index)
        embedding_model_name: HuggingFace model (default: sentence-transformers/all-MiniLM-L6-v2)
        
    Retrieval:
        top_k: Top-K candidates to retrieve (default: 8)
        top_n: Top-N documents after filtering (default: 5)
        similarity_threshold: Minimum similarity (default: 0.3)
        
    Chunking:
        chunk_size_min: Minimum chunk size (default: 500)
        chunk_size_max: Maximum chunk size (default: 800)
        chunk_overlap: Overlap between chunks (default: 100)
        
    Response:
        response_style: Response style (default: friendly_casual)
        max_response_length: Max response length (default: 450)
        enable_humanization: Enable humanization (default: true)
        min_quality_score: Min quality score (default: 0.5)
        timeout: Query timeout (default: 30.0s)
        enable_streaming: Enable streaming responses (default: true)
        
    Caching:
        cache_ttl: Redis cache TTL in seconds (default: 3600)
        
    Search:
        enable_hybrid_search: Enable hybrid search with pattern detection (default: true)
        enable_web_search: Enable web search augmentation (default: false)
    """
    
    # Required settings
    knowledge_base_path: str
    
    # LLM Provider settings (NEW - replaces Gemini-specific config)
    llm_provider: str = "groq"  # "gemini", "openai", "openrouter", "claude", "ollama", "groq"
    llm_api_key: str = ""  # API key for the provider
    llm_model: str = "openai/gpt-oss-20b"  # Model name
    analytics_model: str = "qwen/qwen3-32b"  # Reasoning model for analytics
    
    # Provider-specific options (optional)
    llm_options: dict = field(default_factory=dict)  # Additional provider config
    
    # DEPRECATED: Keep for backwards compatibility with existing code
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    
    # Vector store settings
    vector_store_path: str = "/app/index"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    
    # Retrieval settings
    top_k: int = 8
    top_n: int = 5
    similarity_threshold: float = 0.2
    
    # Chunking settings
    chunk_size_min: int = 500
    chunk_size_max: int = 800
    chunk_overlap: int = 100
    
    # Response settings
    response_style: str = "friendly_casual"
    max_response_length: int = 450
    enable_humanization: bool = True
    min_quality_score: float = 0.5
    timeout: float = 30.0
    enable_streaming: bool = True
    
    # Cache settings
    cache_ttl: int = 3600
    
    # Hybrid search settings
    enable_hybrid_search: bool = True
    
    # Web Search settings
    google_search_api_key: str = ""
    google_cse_id: str = ""
    enable_web_search: bool = False
    
    # Qdrant Memory (Hive Mind)
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "tara_case_memory"
    enable_hive_mind: bool = True
    
    # Retrieval flags
    enable_local_retrieval: bool = True
    
    # Organization context
    organization_name: str = "Daytona"
    organization_location: str = "Global"
    
    # Fallback provider configuration (NEW)
    enable_llm_fallback: bool = False
    fallback_llm_provider: Optional[str] = None
    fallback_llm_api_key: Optional[str] = None
    fallback_llm_model: Optional[str] = None
    
    # OpenRouter specific settings
    openrouter_site_url: str = "https://tara.daytona.io"
    openrouter_app_name: str = "TARA-Daytona"
    openrouter_enable_reasoning: bool = False
    
    # Logging
    log_queries: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        
        # Validate knowledge base path
        if not os.path.exists(self.knowledge_base_path):
            logger.warning(
                f"⚠️ Knowledge base path does not exist: {self.knowledge_base_path}"
            )
        
        # Validate similarity threshold
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )
        
        # Validate top_k >= top_n
        if self.top_k < self.top_n:
            raise ValueError(
                f"top_k ({self.top_k}) must be >= top_n ({self.top_n})"
            )
        
        # Validate chunk sizes
        if self.chunk_size_max <= self.chunk_size_min:
            raise ValueError(
                f"chunk_size_max ({self.chunk_size_max}) must be > chunk_size_min ({self.chunk_size_min})"
            )
        
        # Validate chunk overlap
        if self.chunk_overlap >= self.chunk_size_min:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size_min ({self.chunk_size_min})"
            )
        
        # Validate timeout
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        
        # Validate quality score
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError(
                f"min_quality_score must be between 0.0 and 1.0, got {self.min_quality_score}"
            )
        
        # Backwards compatibility: migrate from gemini_* to llm_* if needed
        if not self.llm_api_key and self.gemini_api_key:
            logger.info("🔄 Migrating from GEMINI_API_KEY to LLM_API_KEY (backwards compatibility)")
            self.llm_api_key = self.gemini_api_key
            self.llm_provider = "gemini"
            if self.gemini_model:
                self.llm_model = self.gemini_model
        
        # Validate LLM provider
        supported_providers = ["gemini", "openai", "openrouter", "claude", "anthropic", "ollama", "groq"]
        if self.llm_provider.lower() not in supported_providers:
            raise ValueError(
                f"llm_provider must be one of {supported_providers}, got '{self.llm_provider}'"
            )
        
        # Warn if LLM API key missing (except for Ollama which runs locally)
        if not self.llm_api_key and self.llm_provider.lower() != "ollama":
            logger.warning(
                f"⚠️ LLM_API_KEY not set for provider '{self.llm_provider}'. Response generation will fail."
            )
        
        # Log configuration if verbose
        if self.verbose:
            logger.info(
                f"🤖 RAGConfig loaded: provider={self.llm_provider}, model={self.llm_model}, "
                f"top_k={self.top_k}, top_n={self.top_n}, "
                f"similarity_threshold={self.similarity_threshold}, "
                f"chunk_size={self.chunk_size_min}-{self.chunk_size_max}, "
                f"timeout={self.timeout}s, cache_ttl={self.cache_ttl}s"
            )
    
    @staticmethod
    def from_env() -> "RAGConfig":
        """
        Load configuration from environment variables.
        
        Returns:
            RAGConfig instance with values from environment
        
        Environment Variables:
            # LLM Provider (NEW - replaces GEMINI_*)
            LLM_PROVIDER: Provider name ("gemini", "openai", "openrouter", "claude", "ollama")
            LLM_API_KEY: API key for the provider
            LLM_MODEL: Model identifier
            
            # Backwards compatibility (DEPRECATED)
            GEMINI_API_KEY: Falls back to this if LLM_API_KEY not set
            GEMINI_MODEL: Falls back to this if LLM_MODEL not set
            
            # Vector Store
            DAYTONA_RAG_KNOWLEDGE_BASE_PATH: Knowledge base directory
            DAYTONA_RAG_VECTOR_STORE_PATH: Vector store directory
            DAYTONA_RAG_EMBEDDING_MODEL: Embedding model name
            
            # Retrieval
            DAYTONA_RAG_TOP_K: Top-K candidates
            DAYTONA_RAG_TOP_N: Top-N documents
            DAYTONA_RAG_SIMILARITY_THRESHOLD: Minimum similarity
            
            # Chunking
            DAYTONA_RAG_CHUNK_SIZE_MIN: Minimum chunk size
            DAYTONA_RAG_CHUNK_SIZE_MAX: Maximum chunk size
            DAYTONA_RAG_CHUNK_OVERLAP: Chunk overlap
            
            # Response
            DAYTONA_RAG_RESPONSE_STYLE: Response style
            DAYTONA_RAG_MAX_RESPONSE_LENGTH: Max response length
            DAYTONA_RAG_ENABLE_HUMANIZATION: Enable humanization
            DAYTONA_RAG_MIN_QUALITY_SCORE: Min quality score
            DAYTONA_RAG_TIMEOUT: Query timeout
            DAYTONA_RAG_ENABLE_STREAMING: Enable streaming
            
            # Cache
            DAYTONA_RAG_CACHE_TTL: Cache TTL
            
            # Search
            DAYTONA_RAG_ENABLE_HYBRID_SEARCH: Enable hybrid search
            ENABLE_WEB_SEARCH: Enable web search
            
            # Logging
            LOG_LEVEL: Logging level
        """
        # NEW: LLM Provider configuration
        llm_provider = (os.getenv("DAYTONA_RAG_LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
        llm_api_key = (os.getenv("DAYTONA_RAG_LLM_API_KEY") or os.getenv("LLM_API_KEY", "")).strip()
        llm_model = (os.getenv("DAYTONA_RAG_LLM_MODEL") or os.getenv("LLM_MODEL", "")).strip()
        
        # DEPRECATED: Backwards compatibility with GEMINI_* env vars
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite").strip()
        
        # Fallback logic: use GEMINI_* or GROQ_* if LLM_* not set
        if not llm_api_key:
            if os.getenv("GROQ_API_KEY", "").strip():
                llm_api_key = os.getenv("GROQ_API_KEY", "").strip()
                llm_provider = "groq"
            elif gemini_api_key:
                llm_api_key = gemini_api_key
                llm_provider = "gemini"
        
        if not llm_model:
            # Use provider-specific defaults
            if llm_provider == "openai":
                llm_model = "gpt-4-turbo-preview"
            elif llm_provider == "openrouter":
                llm_model = "meta-llama/llama-3-8b-instruct:free"  # OpenRouter format
            elif llm_provider in ["claude", "anthropic"]:
                llm_model = "claude-3-opus-20240229"
            elif llm_provider == "ollama":
                llm_model = "llama3.1:8b"
            elif llm_provider == "groq":
                llm_model = "openai/gpt-oss-20b"
            else:  # gemini
                llm_model = gemini_model or "gemini-2.0-flash-lite"
        
        analytics_model = (os.getenv("DAYTONA_RAG_ANALYTICS_MODEL") or os.getenv("ANALYTICS_MODEL", "qwen/qwen3-32b")).strip()
        
        def get_env_int(name, default):
            val = os.getenv(name, "").strip()
            return int(val) if val else default

        def get_env_float(name, default):
            val = os.getenv(name, "").strip()
            return float(val) if val else default

        def get_env_bool(name, default):
            val = os.getenv(name, "").strip().lower()
            if not val:
                return default
            return val in ("true", "1", "yes", "on")

        config = RAGConfig(
            # Required settings
            knowledge_base_path=os.getenv(
                "DAYTONA_RAG_KNOWLEDGE_BASE_PATH",
                "leibniz_knowledge_base"
            ) or "leibniz_knowledge_base",
            
            # LLM Provider (NEW)
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            analytics_model=analytics_model,
            
            # Backwards compatibility (DEPRECATED)
            gemini_api_key=gemini_api_key if gemini_api_key else None,
            gemini_model=gemini_model if gemini_model else None,
            
            # Vector store settings
            vector_store_path=os.getenv(
                "DAYTONA_RAG_VECTOR_STORE_PATH",
                "/app/index"
            ) or "/app/index",
            embedding_model_name=os.getenv(
                "DAYTONA_RAG_EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ) or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            embedding_dimension=get_env_int("DAYTONA_RAG_EMBEDDING_DIMENSION", 384),
            
            # Retrieval settings
            top_k=get_env_int("DAYTONA_RAG_TOP_K", 8),
            top_n=get_env_int("DAYTONA_RAG_TOP_N", 5),
            similarity_threshold=get_env_float("DAYTONA_RAG_SIMILARITY_THRESHOLD", 0.3),
            
            # Chunking settings
            chunk_size_min=get_env_int("DAYTONA_RAG_CHUNK_SIZE_MIN", 500),
            chunk_size_max=get_env_int("DAYTONA_RAG_CHUNK_SIZE_MAX", 800),
            chunk_overlap=get_env_int("DAYTONA_RAG_CHUNK_OVERLAP", 100),
            
            # Response settings
            response_style=os.getenv("DAYTONA_RAG_RESPONSE_STYLE", "friendly_casual") or "friendly_casual",
            max_response_length=get_env_int("DAYTONA_RAG_MAX_RESPONSE_LENGTH", 800),
            enable_humanization=get_env_bool("DAYTONA_RAG_ENABLE_HUMANIZATION", True),
            min_quality_score=get_env_float("DAYTONA_RAG_MIN_QUALITY_SCORE", 0.5),
            timeout=get_env_float("DAYTONA_RAG_TIMEOUT", 30.0),
            enable_streaming=get_env_bool("DAYTONA_RAG_ENABLE_STREAMING", True),
            
            # Cache settings
            cache_ttl=get_env_int("DAYTONA_RAG_CACHE_TTL", 3600),
            
            # Hybrid search settings
            enable_hybrid_search=get_env_bool("DAYTONA_RAG_ENABLE_HYBRID_SEARCH", True),
            
            # Web Search settings
            google_search_api_key=os.getenv("GOOGLE_SEARCH_API_KEY", "").strip(),
            google_cse_id=os.getenv("GOOGLE_CSE_ID", "").strip(),
            enable_web_search=get_env_bool("ENABLE_WEB_SEARCH", False),
            
            # Qdrant settings (Hive Mind)
            qdrant_url=os.getenv("DAYTONA_RAG_QDRANT_URL") or os.getenv("QDRANT_URL", "").strip() or None,
            qdrant_api_key=os.getenv("DAYTONA_RAG_QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY", "").strip() or None,
            qdrant_collection=os.getenv("DAYTONA_RAG_QDRANT_COLLECTION") or os.getenv("QDRANT_COLLECTION", "tara_case_memory").strip(),
            enable_hive_mind=get_env_bool("DAYTONA_RAG_ENABLE_HIVE_MIND", get_env_bool("ENABLE_HIVE_MIND", True)),
            
            # Local retrieval flag
            enable_local_retrieval=get_env_bool("ENABLE_LOCAL_RETRIEVAL", True),
            
            # Organization context
            organization_name=os.getenv("ORGANIZATION_NAME", "Daytona") or "Daytona",
            organization_location=os.getenv("ORGANIZATION_LOCATION", "Global") or "Global",
            
            # Fallback provider settings
            enable_llm_fallback=get_env_bool("ENABLE_LLM_FALLBACK", False),
            fallback_llm_provider=os.getenv("FALLBACK_LLM_PROVIDER", "").strip() or None,
            fallback_llm_api_key=os.getenv("FALLBACK_LLM_API_KEY", "").strip() or None,
            fallback_llm_model=os.getenv("FALLBACK_LLM_MODEL", "").strip() or None,
            
            # OpenRouter settings
            openrouter_site_url=os.getenv("OPENROUTER_SITE_URL", "https://tara.daytona.io") or "https://tara.daytona.io",
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", "TARA-Daytona") or "TARA-Daytona",
            openrouter_enable_reasoning=get_env_bool("OPENROUTER_ENABLE_REASONING", False),
            
            # Logging
            log_queries=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
            verbose=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
        )
        # Debug prints for Qdrant
        logger.info(f"DEBUG: Config Qdrant URL: {config.qdrant_url}")
        logger.info(f"DEBUG: Config Enable Hive Mind: {config.enable_hive_mind}")
        return config
