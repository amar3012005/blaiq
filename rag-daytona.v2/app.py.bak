"""
RAG Service FastAPI Application

HTTP REST API for knowledge base queries with Redis caching.

Reference:
    - Cloud Transformation doc (lines 474-641) - RAG service specifications
    - services/intent/app.py - FastAPI pattern
"""

import os
import sys
import time
import logging
import json
import hashlib
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Union

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from daytona_agent.services.shared.redis_client import get_redis_client, close_redis_client, ping_redis
from daytona_agent.services.rag.config import RAGConfig
from daytona_agent.services.rag.rag_engine import RAGEngine
from daytona_agent.services.rag.index_builder import IndexBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    context: Optional[Dict[str, Any]] = Field(None, description="Context from intent service")
    enable_streaming: Optional[bool] = Field(None, description="Enable streaming response")
    history_context: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="Conversation history for context-aware responses")
    language: Optional[str] = Field("english", description="Response language: 'english' or 'german'")
    tenant_id: Optional[str] = Field("demo", description="Tenant/Agent identifier for cache isolation")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(..., description="Source document filenames")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Retrieval confidence")
    timing_breakdown: Dict[str, float] = Field(..., description="Timing metrics")
    cached: bool = Field(..., description="Whether result was served from cache")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_size: int
    cache_hit_rate: float
    redis_connected: bool
    gemini_available: bool
    qdrant_enabled: bool
    qdrant_url: Optional[str] = None
    uptime_seconds: float


class RebuildIndexRequest(BaseModel):
    knowledge_base_path: Optional[str] = Field(None, description="Override knowledge base path")


class RebuildIndexResponse(BaseModel):
    status: str
    documents_indexed: int
    categories: int
    build_time_seconds: float


class SaveCaseRequest(BaseModel):
    user_id: str = Field(..., description="User phone number or ID")
    issue: Optional[str] = Field(None, description="The problem that was resolved")
    solution: Optional[str] = Field(None, description="How the problem was solved")
    history_context: Optional[str] = Field(None, description="Conversation history to distill")
    tenant_id: str = Field("demo", description="Tenant identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class SaveCaseResponse(BaseModel):
    status: str
    message: str
    case_id: Optional[str] = None


# Global state (initialized in lifespan)
rag_engine: Optional[RAGEngine] = None
redis_client: Optional[redis.Redis] = None
cache_hits = 0
cache_misses = 0
app_start_time = 0.0


# Redis client utilities
# Removed custom implementation in favor of shared client
# async def get_redis_client() -> redis.Redis: ...
# async def close_redis_client(client: redis.Redis): ...
# async def ping_redis(client: redis.Redis) -> bool: ...


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup/shutdown."""
    global rag_engine, redis_client, cache_hits, cache_misses, app_start_time
    
    # Startup
    logger.info(" Starting RAG service...")
    app_start_time = time.time()
    
    try:
        # Load config
        config = RAGConfig.from_env()
        logger.info(" Configuration loaded")
        
        # Log web search configuration
        if config.enable_web_search:
            if config.google_search_api_key and config.google_cse_id:
                logger.info(f" ✅ Web search ENABLED (API key: {config.google_search_api_key[:10]}..., CSE ID: {config.google_cse_id})")
            else:
                logger.warning(f" ⚠️ Web search enabled but credentials missing (API key: {'present' if config.google_search_api_key else 'missing'}, CSE ID: {'present' if config.google_cse_id else 'missing'})")
        else:
            logger.info(" ℹ️ Web search DISABLED (set ENABLE_WEB_SEARCH=true to enable)")
        
        # Create RAG engine
        rag_engine = RAGEngine(config)
        
        # Log Qdrant (Hive Mind) status
        if rag_engine.qdrant and rag_engine.qdrant.enabled:
            logger.info(f"🧠 ✅ Qdrant Hive Mind CONNECTED: {rag_engine.qdrant.url}")
            logger.info(f"🧠    Collection: {rag_engine.qdrant.collection_name}")
        else:
            logger.warning(f"🧠 ⚠️ Qdrant Hive Mind NOT AVAILABLE - check QDRANT_URL and QDRANT_API_KEY env vars")
        
        # If index not loaded, try to build it (only if local retrieval is enabled)
        if config.enable_local_retrieval:
            if not rag_engine.vector_store or not rag_engine.documents:
                logger.warning(" FAISS index not found, attempting to build...")
                from daytona_agent.services.rag.index_builder import IndexBuilder
                builder = IndexBuilder(config)
                if builder.build_index():
                    rag_engine.load_index()  # Reload after build
                    logger.info(f" Index built successfully: {len(rag_engine.documents)} documents")
                else:
                    logger.error(" Index build failed")
            
            # Final validation
            if not rag_engine.vector_store or not rag_engine.documents:
                logger.error(" FAISS index not available - service in degraded mode")
            else:
                logger.info(f" RAG engine initialized: {len(rag_engine.documents)} documents")
        else:
            logger.info("ℹ️ Local retrieval disabled - skipping FAISS index checks")
        
        # Connect to Redis (optional - service can run without it)
        try:
            # Use shared client which handles config from env vars
            redis_client = await asyncio.wait_for(get_redis_client(), timeout=15.0)
            await asyncio.wait_for(ping_redis(redis_client), timeout=5.0)
            logger.info(f" Redis connected successfully")
        except asyncio.TimeoutError:
            logger.warning(f" Redis connection timeout - service will run in degraded mode")
            redis_client = None
        except Exception as redis_error:
            logger.warning(f" Redis connection failed: {redis_error} - caching disabled")
            redis_client = None
        
        # Initialize counters
        cache_hits = 0
        cache_misses = 0
        
        # Store in app state
        app.state.rag_engine = rag_engine
        app.state.redis = redis_client
        app.state.cache_hits = cache_hits
        app.state.cache_misses = cache_misses
        app.state.start_time = app_start_time
        
        logger.info(" RAG service ready")
        
        yield
        
        # Shutdown
        logger.info(" Shutting down RAG service...")
        
        # Log performance stats
        if rag_engine:
            stats = rag_engine.get_performance_stats()
            logger.info(f" Performance stats: {stats}")
        
        # Close Redis
        if redis_client:
            # Shared client manages its own lifecycle, but we can close our reference
            # Actually, shared client is singleton, so we shouldn't close it here if other services use it
            # But since this is microservice, we are the only user in this process.
            # However, close_redis_client in shared lib closes the global client.
            await close_redis_client() 
            logger.info(" Redis connection closed")
        
        logger.info(" RAG service shutdown complete")
    
    except Exception as e:
        logger.error(f" Startup error: {e}", exc_info=True)
        raise


# Support for path-based routing (e.g. /rag)
root_path = os.getenv("RAG_ROOT_PATH", "")

# Create FastAPI app
app = FastAPI(
    title="Daytona RAG Service",
    description="Knowledge base retrieval and intelligence service",
    version="2.1.0",
    lifespan=lifespan,
    root_path=root_path
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_middleware(GZipMiddleware, minimum_size=1000) # Disabled to prevent blank page issues

# Middleware to strip /rag or /cartesia prefix if present (for path-based routing)
@app.middleware("http")
async def strip_path_prefix(request: Request, call_next):
    path = request.url.path
    logger.info(f"🔍 Incoming request path: {path}")
    
    # Store whether we should skip generation in request state
    # Triggered by /cartesia prefix
    request.state.skip_generation = path.startswith("/cartesia")
    
    prefix = None
    if path.startswith("/rag"):
        prefix = "/rag"
    elif path.startswith("/cartesia"):
        prefix = "/cartesia"
        
    if prefix:
        # Create a new scope with the adjusted path
        new_path = path[len(prefix):] or "/"
        logger.info(f"✂️ Stripping prefix '{prefix}': {path} -> {new_path}")
        request.scope["path"] = new_path
        # Also update raw_path if needed
        if "raw_path" in request.scope:
            request.scope["raw_path"] = new_path.encode()
    else:
        logger.info(f"➡️ No prefix matched (checked /rag, /cartesia)")
            
    response = await call_next(request)
    return response

# Helper function to get static directory path
def get_static_dir():
    """Get the static directory path, handling both direct and module imports."""
    # Try current file's directory first
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(current_dir, "static")
    
    if os.path.exists(static_dir):
        return static_dir
    
    # Fallback: try relative to package root
    package_dir = os.path.dirname(os.path.dirname(current_dir))
    static_dir = os.path.join(package_dir, "rag-leibniz", "static")
    
    if os.path.exists(static_dir):
        return static_dir
    
    return None

# Mount static files directory for client.html
# Note: Mount must happen before route definitions to avoid conflicts
static_dir = get_static_dir()
if static_dir:
    try:
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(f"✅ Static files mounted from: {static_dir}")
    except Exception as e:
        logger.warning(f"⚠️ Could not mount static files: {e}")
else:
    logger.warning(f"⚠️ Static directory not found")

@app.get("/")
async def root():
    """Serve the RAG testing client interface."""
    static_dir_path = get_static_dir()
    if static_dir_path:
        client_path = os.path.join(static_dir_path, "client.html")
        if os.path.exists(client_path):
            logger.info(f"Serving client from: {client_path}")
            return FileResponse(client_path, media_type="text/html")
        else:
            logger.error(f"Client file not found at: {client_path}")
    else:
        logger.error("Static directory not found")
    
    # Fallback: return simple HTML with instructions
    return {
        "message": "RAG Service API",
        "client_url": "/static/client.html",
        "docs": "/docs",
        "health": "/health"
    }

from fastapi.responses import RedirectResponse
@app.get("/client")
async def client_redirect():
    return RedirectResponse(url="/")


class RetrieveResponse(BaseModel):
    query_english: str
    original_language: str
    relevant_docs: List[Dict[str, Any]]
    hive_mind_context: str
    web_results: str
    timing: Dict[str, float]
    fast_path_type: Optional[str]
    history_context: str


@app.post("/api/v1/retrieve", response_model=RetrieveResponse)
async def retrieve_only(request: QueryRequest):
    """
    Retrieve relevant context (Docs, Hive Mind, Web) WITHOUT generating an answer.
    
    Used by voice agents (abella) to fetch context for local generation.
    """
    try:
        # Ensure context exists and contains language
        context = request.context or {}
        if request.language:
            context['language'] = request.language
            
        # Call the new retrieve_context method
        result = await app.state.rag_engine.retrieve_context(
            request.query,
            context,
            history_context=request.history_context
        )
        
        return RetrieveResponse(**result)
    
    except Exception as e:
        logger.error(f"Retrieve error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.post("/api/v1/query")
async def query_knowledge_base(request_data: QueryRequest, request: Request):
    """
    Process knowledge base query with context-aware retrieval.
    
    If request arrived via /cartesia prefix, skips LLM generation and 
    returns retrieval context only.
    """
    try:
        # Check if we should skip generation (detected by middleware)
        skip_generation = getattr(request.state, "skip_generation", False)
        
        if skip_generation:
            logger.info(f"⚡ Context-only retrieval triggered (via /cartesia)")
            # Execute retrieval only
            context_data = request_data.context or {}
            if request_data.language:
                context_data['language'] = request_data.language
            
            result = await app.state.rag_engine.retrieve_context(
                request_data.query,
                context_data,
                history_context=request_data.history_context
            )
            return result

        # Original query logic proceeds...
        # Generate cache key (include language to prevent cross-language cache pollution)
        lang_suffix = f":{request_data.language}" if request_data.language else ""
        tenant_prefix = f"{request_data.tenant_id or 'demo'}:"
        cache_key = f"rag:{tenant_prefix}{hashlib.md5(request_data.query.encode()).hexdigest()}{lang_suffix}"
        
        # Check Redis cache (only if connected and TTL > 0)
        cached = None
        if app.state.redis and app.state.rag_engine.config.cache_ttl > 0:
            try:
                cached = await app.state.redis.get(cache_key)
            except Exception as cache_read_error:
                logger.warning(f" Cache read failed: {cache_read_error}")
                cached = None
        
        if cached:
            # Cache hit
            app.state.cache_hits += 1
            result = json.loads(cached)
            result['cached'] = True
            
            logger.info(f" CACHE HIT: {request_data.query[:50]}...")
            return QueryResponse(**result)
        
        # Cache miss
        app.state.cache_misses += 1
        
        
        # Ensure context exists and contains language
        context_data = request_data.context or {}
        if request_data.language:
            context_data['language'] = request_data.language
            
        # Process query
        result = await app.state.rag_engine.process_query(
            request_data.query,
            context_data,
            streaming_callback=None,  # Streaming handled separately if needed
            history_context=request_data.history_context,
            tenant_id=request_data.tenant_id
        )
        
        # Add cached flag
        result['cached'] = False
        
        # Cache result (only if Redis is available and TTL > 0)
        if app.state.redis and app.state.rag_engine.config.cache_ttl > 0:
            try:
                await app.state.redis.setex(
                    cache_key,
                    app.state.rag_engine.config.cache_ttl,
                    json.dumps({
                        'answer': result['answer'],
                        'sources': result['sources'],
                        'confidence': result['confidence'],
                        'timing_breakdown': result['timing_breakdown'],
                        'metadata': result['metadata']
                    })
                )
            except Exception as cache_error:
                logger.warning(f"️ Cache write failed: {cache_error}")
        
        # Log query
        if app.state.rag_engine.config.log_queries:
            logger.info(
                f" QUERY: {request_data.query[:50]}... → "
                f"{result['confidence']:.2f} confidence, "
                f"{result['timing_breakdown']['total_ms']:.1f}ms"
            )
            logger.info(f" RESPONSE: {result['answer'][:100]}...")
            if 'first_chunk_ms' in result['timing_breakdown']:
                logger.info(f" FIRST CHUNK TIME: {result['timing_breakdown']['first_chunk_ms']:.1f}ms")
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f" Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/api/v1/stream_query")
async def stream_query_knowledge_base(request: QueryRequest):
    """
    Stream knowledge base query response.
    Returns a stream of JSON objects: {"text": "...", "is_final": bool}
    """
    async def event_generator():
        # Generate cache key (include history hash AND language to prevent stale/wrong-language responses)
        history_hash = hashlib.md5((request.history_context or "").encode()).hexdigest()[:8]
        lang_suffix = f":{request.language}" if request.language else ""
        tenant_prefix = f"{request.tenant_id or 'demo'}:"
        cache_key = f"rag:{tenant_prefix}{hashlib.md5(request.query.encode()).hexdigest()}:{history_hash}{lang_suffix}"

        # Check Redis cache (only if connected)
        cached = None
        if app.state.redis:
            try:
                cached = await app.state.redis.get(cache_key)
            except Exception as cache_read_error:
                logger.warning(f" Cache read failed: {cache_read_error}")
                cached = None
        
        if cached:
            # Cache hit - simulate streaming
            app.state.cache_hits += 1
            result = json.loads(cached)
            answer = result.get('answer', '')
            
            logger.info(f"✅ CACHE HIT (Streaming): {request.query[:50]}...")
            logger.info(f"📤 Complete cached response:")
            logger.info(f"   {answer}")
            
            # Stream the cached answer in chunks to simulate natural typing/speech
            # CRITICAL FIX: Use larger chunks and slower pacing to prevent TTS flooding
            import re
            
            # Clean markdown from entire answer first
            cleaned_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
            cleaned_answer = re.sub(r'\*([^*]+)\*', r'\1', cleaned_answer)
            
            # Split by sentences for more natural chunking (fallback to word-based if no periods)
            sentences = re.split(r'([.!?]\s+)', cleaned_answer)
            chunks = []
            current_chunk = ""
            
            for part in sentences:
                if len(current_chunk) + len(part) > 80:  # Target ~80 chars per chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = part
                else:
                    current_chunk += part
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If no sentence boundaries found, fall back to word-based chunking
            if not chunks:
                words = cleaned_answer.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 > 80:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = word
                    else:
                        current_chunk += (" " + word if current_chunk else word)
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
            
            # Stream chunks with proper pacing (50ms delay to prevent TTS flooding)
            for chunk in chunks:
                if chunk.strip():  # Skip empty chunks
                    yield json.dumps({"text": chunk, "is_final": False}) + "\n"
                    # 50ms delay between chunks (prevents detected_unusual_activity in TTS)
                    await asyncio.sleep(0.05)
            
            # Final chunk
            yield json.dumps({"text": "", "is_final": True}) + "\n"
            return

        # Cache miss
        app.state.cache_misses += 1
        
        q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        # Container for the full result to cache later
        full_result_container = {}
        accumulated_response = ""  # Track complete response for logging
        
        def callback(text, is_final):
            # Clean text of markdown (e.g., **bold** or *italic*)
            import re
            cleaned_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold**
            cleaned_text = re.sub(r'\*([^*]+)\*', r'\1', cleaned_text)  # Remove *italic*
            q.put_nowait((cleaned_text, is_final))
            
        async def run_query():
            try:
                # Build context with language if provided
                query_context = request.context or {}
                
                # Map short language codes to full names
                lang = request.language or "english"
                if lang.lower() in ("en", "eng"):
                    lang = "english"
                elif lang.lower() in ("de", "deu", "ger"):
                    lang = "german"
                
                query_context['language'] = lang
                
                # Process query with streaming callback
                result = await app.state.rag_engine.process_query(
                    request.query,
                    query_context,
                    streaming_callback=callback,
                    history_context=request.history_context,
                    tenant_id=request.tenant_id
                )
                
                # Store result for caching
                full_result_container['data'] = result
                
            except Exception as e:
                logger.error(f"Streaming query error: {e}")
                loop.call_soon_threadsafe(q.put_nowait, (f"Error: {str(e)}", True))
            finally:
                await q.put(None) # Sentinel

        # Start query task
        asyncio.create_task(run_query())

        while True:
            item = await q.get()
            if item is None:
                break
            text, is_final = item
            accumulated_response += text  # Accumulate for logging
            yield json.dumps({"text": text, "is_final": is_final}) + "\n"
            
        # Log complete response after streaming finishes
        if accumulated_response:
            logger.info(f"📤 Complete streaming response for query '{request.query[:50]}...':")
            logger.info(f"   {accumulated_response}")
            
        # After streaming is done, cache the result if we have it and TTL > 0
        if 'data' in full_result_container and app.state.redis and app.state.rag_engine.config.cache_ttl > 0:
            try:
                result = full_result_container['data']
                await app.state.redis.setex(
                    cache_key,
                    app.state.rag_engine.config.cache_ttl,
                    json.dumps({
                        'answer': result['answer'],
                        'sources': result['sources'],
                        'confidence': result['confidence'],
                        'timing_breakdown': result['timing_breakdown'],
                        'metadata': result['metadata']
                    })
                )
                logger.info(f" Cached streamed result for: {request.query[:30]}...")
            except Exception as cache_error:
                logger.warning(f"️ Cache write failed: {cache_error}")

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Service health check.
    
    Returns index status, cache hit rate, Redis/Gemini availability, and uptime.
    """
    try:
        # Calculate cache hit rate
        total_requests = app.state.cache_hits + app.state.cache_misses
        cache_hit_rate = app.state.cache_hits / total_requests if total_requests > 0 else 0.0
        
        # Check Redis health (if available)
        redis_connected = False
        if app.state.redis:
            redis_connected = await ping_redis(app.state.redis)
        
        # Get RAG engine stats
        index_loaded = app.state.rag_engine.vector_store is not None
        index_size = len(app.state.rag_engine.documents)
        gemini_available = app.state.rag_engine.llm is not None
        
        # Get Qdrant status
        qdrant_enabled = False
        qdrant_url = None
        if app.state.rag_engine.qdrant:
            qdrant_enabled = app.state.rag_engine.qdrant.enabled
            qdrant_url = app.state.rag_engine.qdrant.url
        
        # Calculate uptime
        uptime_seconds = time.time() - app.state.start_time
        
        # Determine status
        if not index_loaded:
            status = "unhealthy"
            status_code = 503
        elif not redis_connected:
            status = "degraded"
            status_code = 200
        else:
            status = "healthy"
            status_code = 200
        
        return HealthResponse(
            status=status,
            index_loaded=index_loaded,
            index_size=index_size,
            cache_hit_rate=cache_hit_rate,
            redis_connected=redis_connected,
            gemini_available=gemini_available,
            qdrant_enabled=qdrant_enabled,
            qdrant_url=qdrant_url,
            uptime_seconds=uptime_seconds
        )
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/metrics")
async def get_metrics():
    """
    Detailed performance metrics.
    
    Returns RAG engine stats, cache stats, and index stats.
    """
    try:
        # RAG engine stats
        rag_stats = app.state.rag_engine.get_performance_stats()
        
        # Cache stats
        total_requests = app.state.cache_hits + app.state.cache_misses
        cache_stats = {
            'cache_hits': app.state.cache_hits,
            'cache_misses': app.state.cache_misses,
            'cache_hit_rate': app.state.cache_hits / total_requests if total_requests > 0 else 0.0
        }
        
        # Index stats
        index_stats = {
            'total_documents': len(app.state.rag_engine.documents),
            'categories': len(set(m.get('category', '') for m in app.state.rag_engine.doc_metadata)),
            'embedding_dimension': app.state.rag_engine.vector_store.d if app.state.rag_engine.vector_store else 0
        }
        
        # Qdrant (Hive Mind) stats
        qdrant_stats = {
            'enabled': app.state.rag_engine.qdrant.enabled if app.state.rag_engine.qdrant else False,
            'url': app.state.rag_engine.qdrant.url if app.state.rag_engine.qdrant else None,
            'collection': app.state.rag_engine.qdrant.collection_name if app.state.rag_engine.qdrant else None
        }
        
        # Uptime
        uptime_seconds = time.time() - app.state.start_time
        
        return {
            'rag_engine': rag_stats,
            'cache': cache_stats,
            'index': index_stats,
            'qdrant_hive_mind': qdrant_stats,
            'uptime_seconds': uptime_seconds
        }
    
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.post("/api/v1/admin/rebuild_index", response_model=RebuildIndexResponse)
async def rebuild_index(request: RebuildIndexRequest):
    """
    Rebuild FAISS index from knowledge base (admin endpoint).
    
    Useful for knowledge base updates. Clears cache after rebuild.
    """
    try:
        build_start = time.time()
        
        # Create config (override path if provided)
        config = RAGConfig.from_env()
        if request.knowledge_base_path:
            config.knowledge_base_path = request.knowledge_base_path
        
        # Build index
        builder = IndexBuilder(config)
        success = builder.build_index()
        
        if not success:
            raise HTTPException(status_code=500, detail="Index build failed")
        
        # Reload index in RAG engine
        app.state.rag_engine.load_index()
        
        # Clear cache (only if Redis is available)
        if app.state.redis:
            try:
                # Delete all rag:* keys
                keys = await app.state.redis.keys("rag:*")
                if keys:
                    await app.state.redis.delete(*keys)
                    logger.info(f"️ Cleared {len(keys)} cached queries")
            except Exception as cache_error:
                logger.warning(f"️ Cache clear failed: {cache_error}")
        else:
            logger.info("️ Redis not available - skipping cache clear")
        
        # Get stats
        stats = builder.get_index_stats()
        build_time = time.time() - build_start
        
        logger.info(f" Index rebuilt: {stats['total_documents']} documents in {build_time:.2f}s")
        
        return RebuildIndexResponse(
            status="success",
            documents_indexed=stats['total_documents'],
            categories=stats['categories'],
            build_time_seconds=build_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rebuild error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


@app.post("/api/v1/save_case", response_model=SaveCaseResponse)
async def save_case(request: SaveCaseRequest):
    """
    Save a resolved case to Qdrant Hive Mind for collective learning.
    If history_context is provided, distills it into issue/solution first.
    """
    try:
        if not app.state.rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        if not app.state.rag_engine.qdrant or not app.state.rag_engine.qdrant.enabled:
            return SaveCaseResponse(
                status="skipped",
                message="Qdrant not configured - case not saved"
            )

        issue = request.issue
        solution = request.solution

        # AUTO-DISTILL: If history provided, use LLM to extract core issue/solution
        if request.history_context and (not issue or not solution):
            logger.info(f"🧠 Distilling history for User {request.user_id[:10]}...")
            distilled = await app.state.rag_engine.distill_history_to_case(request.history_context)
            if distilled:
                issue = issue or distilled.get('issue')
                solution = solution or distilled.get('solution')

        if not issue or not solution:
            return SaveCaseResponse(
                status="failed",
                message="Missing issue/solution and distillation failed"
            )
        
        # Embed the issue for vector storage
        issue_vector = app.state.rag_engine.embeddings.embed_query(issue)
        
        # Save to Qdrant
        await app.state.rag_engine.qdrant.upsert_case(
            user_id=request.user_id,
            issue=issue,
            solution=solution,
            vector=issue_vector,
            tenant_id=request.tenant_id,
            metadata=request.metadata
        )
        
        logger.info(f"🧠 Saved distilled case to Hive Mind: {issue[:50]}...")
        
        return SaveCaseResponse(
            status="success",
            message="Case distilled and saved successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to save case: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save case: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Hive Mind Visualization Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class HiveMindPoint(BaseModel):
    id: str
    x: float
    y: float
    issue: str
    solution: str
    issue_type: Optional[str] = None
    customer_segment: Optional[str] = None


class HiveMindVisualizationResponse(BaseModel):
    points: List[HiveMindPoint]
    collection_name: str
    total_points: int
    dimension: int
    algorithm: str


@app.get("/api/v1/hive-mind/visualize", response_model=HiveMindVisualizationResponse)
async def visualize_hive_mind(limit: int = 100, algorithm: str = "tsne", tenant_id: str = "demo"):
    """
    Fetch Hive Mind vectors and reduce to 2D for visualization.
    
    Uses t-SNE (default) or PCA to project high-dimensional embeddings to 2D coordinates.
    Returns points with x,y coordinates and payload metadata for interactive visualization.
    
    Args:
        limit: Maximum number of points to return (default 100)
        algorithm: Dimensionality reduction algorithm ('tsne' or 'pca')
    
    Returns:
        List of points with 2D coordinates and metadata
    """
    try:
        if not app.state.rag_engine or not app.state.rag_engine.qdrant:
            raise HTTPException(status_code=503, detail="Hive Mind (Qdrant) not available")
        
        qdrant = app.state.rag_engine.qdrant
        if not qdrant.enabled:
            raise HTTPException(status_code=503, detail="Hive Mind is disabled")
        
        # Fetch vectors with payloads from Qdrant
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        
        sync_client = QdrantClient(url=qdrant.url, api_key=qdrant.api_key)
        
        # Scroll through collection to get all points (filtered by tenant)
        scroll_result = sync_client.scroll(
            collection_name=qdrant.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id)
                    )
                ]
            ),
            limit=min(limit, 500),  # Cap at 500 for performance
            with_payload=True,
            with_vectors=True
        )
        
        points_data = scroll_result[0]
        
        if not points_data:
            return HiveMindVisualizationResponse(
                points=[],
                collection_name=qdrant.collection_name,
                total_points=0,
                dimension=qdrant.embedding_dim,
                algorithm=algorithm
            )
        
        # Extract vectors and payloads
        import numpy as np
        vectors = []
        payloads = []
        point_ids = []
        
        for point in points_data:
            if point.vector:
                vectors.append(point.vector)
                payloads.append(point.payload or {})
                point_ids.append(str(point.id))
        
        if len(vectors) < 2:
            # Need at least 2 points for dimensionality reduction
            return HiveMindVisualizationResponse(
                points=[
                    HiveMindPoint(
                        id=point_ids[0] if point_ids else "0",
                        x=0.0,
                        y=0.0,
                        issue=payloads[0].get("issue", "") if payloads else "",
                        solution=payloads[0].get("solution", "") if payloads else "",
                        issue_type=payloads[0].get("issue_type") if payloads else None,
                        customer_segment=payloads[0].get("customer_segment") if payloads else None
                    )
                ] if vectors else [],
                collection_name=qdrant.collection_name,
                total_points=len(vectors),
                dimension=qdrant.embedding_dim,
                algorithm=algorithm
            )
        
        vectors_np = np.array(vectors)
        
        # Dimensionality reduction
        if algorithm.lower() == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(vectors_np)
        else:
            # Default: t-SNE
            from sklearn.manifold import TSNE
            # Adjust perplexity based on number of samples
            perplexity = min(30, max(5, len(vectors) // 3))
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=500)
            coords_2d = reducer.fit_transform(vectors_np)
        
        # Normalize to [-1, 1] range for visualization
        coords_min = coords_2d.min(axis=0)
        coords_max = coords_2d.max(axis=0)
        coords_range = coords_max - coords_min
        coords_range[coords_range == 0] = 1  # Avoid division by zero
        coords_normalized = 2 * (coords_2d - coords_min) / coords_range - 1
        
        # Build response points
        result_points = []
        for i, (x, y) in enumerate(coords_normalized):
            payload = payloads[i]
            result_points.append(HiveMindPoint(
                id=point_ids[i],
                x=float(x),
                y=float(y),
                issue=payload.get("issue", "Unknown issue"),
                solution=payload.get("solution", "No solution recorded"),
                issue_type=payload.get("issue_type"),
                customer_segment=payload.get("customer_segment")
            ))
        
        logger.info(f"🧠 Hive Mind visualization: {len(result_points)} points using {algorithm.upper()}")
        
        return HiveMindVisualizationResponse(
            points=result_points,
            collection_name=qdrant.collection_name,
            total_points=len(result_points),
            dimension=qdrant.embedding_dim,
            algorithm=algorithm
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hive Mind visualization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Hive Mind Insights Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class InsightItem(BaseModel):
    id: str
    issue: str
    solution: str
    issue_type: Optional[str] = None
    customer_segment: Optional[str] = None
    timestamp: Optional[str] = None


class DomainStat(BaseModel):
    domain: str
    count: int
    percentage: float


class HiveMindInsightsResponse(BaseModel):
    recent_knowledge: List[InsightItem]
    trending_domains: List[DomainStat]
    total_knowledge: int
    unique_domains: int
    customer_segments: Dict[str, int]
    collection_name: str


@app.get("/api/v1/hive-mind/insights", response_model=HiveMindInsightsResponse)
async def get_hive_mind_insights(limit: int = 10):
    """
    Get insights from the Hive Mind collective intelligence.
    
    Returns:
        - Recent knowledge additions
        - Trending/popular domains
        - Customer segment breakdown
        - Total knowledge count
    """
    try:
        if not app.state.rag_engine or not app.state.rag_engine.qdrant:
            raise HTTPException(status_code=503, detail="Hive Mind (Qdrant) not available")
        
        qdrant = app.state.rag_engine.qdrant
        if not qdrant.enabled:
            raise HTTPException(status_code=503, detail="Hive Mind is disabled")
        
        from qdrant_client import QdrantClient
        sync_client = QdrantClient(url=qdrant.url, api_key=qdrant.api_key)
        
        # Fetch all points for analysis (cap at 500)
        scroll_result = sync_client.scroll(
            collection_name=qdrant.collection_name,
            limit=500,
            with_payload=True,
            with_vectors=False  # Don't need vectors for insights
        )
        
        all_points = scroll_result[0]
        
        if not all_points:
            return HiveMindInsightsResponse(
                recent_knowledge=[],
                trending_domains=[],
                total_knowledge=0,
                unique_domains=0,
                customer_segments={},
                collection_name=qdrant.collection_name
            )
        
        # Extract recent knowledge (by ID, higher = newer for our dataset)
        # Sort by ID descending to get recent first
        sorted_points = sorted(all_points, key=lambda p: int(p.id) if str(p.id).isdigit() else 0, reverse=True)
        recent = sorted_points[:limit]
        
        recent_knowledge = [
            InsightItem(
                id=str(p.id),
                issue=p.payload.get("issue", "Unknown"),
                solution=p.payload.get("solution", "")[:200] + "..." if len(p.payload.get("solution", "")) > 200 else p.payload.get("solution", ""),
                issue_type=p.payload.get("issue_type"),
                customer_segment=p.payload.get("customer_segment"),
                timestamp=p.payload.get("timestamp")
            )
            for p in recent
        ]
        
        # Calculate domain statistics
        domain_counts = {}
        segment_counts = {}
        
        for point in all_points:
            payload = point.payload or {}
            domain = payload.get("issue_type", "general")
            segment = payload.get("customer_segment", "unknown")
            
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            segment_counts[segment] = segment_counts.get(segment, 0) + 1
        
        total = len(all_points)
        
        # Sort domains by count (trending)
        trending_domains = [
            DomainStat(
                domain=domain.replace("_", " ").title(),
                count=count,
                percentage=round(count / total * 100, 1)
            )
            for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        ]
        
        logger.info(f"🧠 Hive Mind insights: {total} nodes, {len(domain_counts)} domains")
        
        return HiveMindInsightsResponse(
            recent_knowledge=recent_knowledge,
            trending_domains=trending_domains,
            total_knowledge=total,
            unique_domains=len(domain_counts),
            customer_segments=segment_counts,
            collection_name=qdrant.collection_name
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hive Mind insights error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Insights failed: {str(e)}")


# Static file serving (for index.html demo page)
if __name__ == "__main__":
    import uvicorn
    
    # Docker vs local: single worker to avoid FAISS index duplication
    # Rely on async concurrency for throughput
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8003")),
        workers=1,
        log_level="info"
    )
