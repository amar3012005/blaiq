"""
Optimized GraphRAG Retrieval API with Caching and Parallel Execution
Drop-in replacement for retriever_api.py with performance enhancements.
"""

import os
import time
import asyncio
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.auth import verify_api_key
from utils.auth import verify_api_key
from utils.job_store import JobStore
from utils.llm_logger import log_llm_event

from core.cache_manager import CacheManager
from core.session_manager import SessionManager
from core.async_retriever import AsyncRetriever
from core.reranker import get_reranker
from retriever.graphrag_retriever import GraphRAGRetriever, generate_answer
from fastapi import Request
from langchain_core.documents import Document

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import json

load_dotenv()

# Initialize cache manager
cache_manager = CacheManager(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    ttl=int(os.getenv("CACHE_TTL", "3600")),
    enabled=os.getenv("ENABLE_CACHE", "true").lower() == "true"
)

# Initialize session manager
session_manager = SessionManager(cache_manager)

retriever_job_store = JobStore(storage_dir="_jobs", api_name="retriever")

# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., description="Your question", min_length=1)
    k: int = Field(default=15, description="Number of chunks to retrieve", ge=1)
    debug: bool = Field(default=False, description="Enable debug mode")
    generate_answer: bool = Field(default=True, description="Generate LLM answer")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    user_prompt: Optional[str] = Field(default=None, description="Custom user prompt")
    entity_extraction_prompt: Optional[str] = Field(default=None, description="Custom entity extraction prompt")
    
    # Qdrant config
    qdrant_url: Optional[str] = Field(default=None)
    qdrant_host: Optional[str] = Field(default=None)
    qdrant_port: Optional[int] = Field(default=None)
    qdrant_api_key: Optional[str] = Field(default=None)
    collection_name: Optional[str] = Field(default=None)
    
    # Cache control
    use_cache: bool = Field(default=True, description="Use Redis cache")
    
    # Retrieval Mode
    mode: str = Field(default="local", description="Retrieval mode: 'local' (standard) or 'global' (hive intelligence)")
    
    # Reranker control (Elite Status)
    use_reranker: bool = Field(default=True, description="Enable BGE Reranker")
    rerank_top_k: int = Field(default=20, description="Number of candidates to rerank")
    
    # Session management
    session_id: Optional[str] = Field(default=None, description="Unique session ID for conversation history")
    
    # Content format control
    content_mode: Optional[str] = Field(default="DEFAULT", description="Specialized content mode (EMAIL, TABLE, INVOICE)")


class QueryResponse(BaseModel):
    """Response model"""
    query: str
    answer: str
    chunks_retrieved: int
    retrieval_stats: Dict[str, Any]
    retrieval_time: float
    answer_time: float
    total_time: float
    cached: bool = False
    cache_stats: Optional[Dict[str, Any]] = None

app = FastAPI(
    title="GraphRAG Retriever API (Optimized)",
    description="""
Optimized GraphRAG retrieval with:
- Redis caching (1-hour TTL)
- Parallel Vector + Graph + Keyword search
- 10-50x faster keyword search
- Sub-2s response times for cache misses
    """,
    version="4.0.0",
)

# Enable CORS for local testing
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global logging setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphrag_api")

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"✅ Static files mounted from {static_dir}")

@app.get("/rag")
@app.get("/client")
async def serve_client():
    """Serve the GraphRAG client UI."""
    client_path = os.path.join(static_dir, "client.html")
    if os.path.exists(client_path):
        return FileResponse(client_path)
    return {"error": "Client UI not found"}

# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Connect to Redis on startup and log configuration."""
    logger.info("🚀 Starting GraphRAG Optimized API")
    
    # 1. Log Configuration (Masked)
    logger.info("🔧 Configuration Check:")
    
    qdrant_url = os.getenv("QDRANT_URL", "Not Set")
    logger.info(f"   📡 Qdrant URL: {qdrant_url}")
    logger.info(f"   📦 Qdrant Collection: {os.getenv('QDRANT_COLLECTION', 'graphrag_chunks')}")
    
    neo4j_uri = os.getenv("NEO4J_URI", "Not Set")
    logger.info(f"   🔗 Neo4j URI: {neo4j_uri}")
    logger.info(f"   👤 Neo4j User: {os.getenv('NEO4J_USER', 'neo4j')}")
    
    openai_key = os.getenv("OPENAI_API_KEY", "")
    masked_key = f"{openai_key[:8]}...{openai_key[-4:]}" if len(openai_key) > 10 else "Not Configured"
    logger.info(f"   🔑 OpenAI API Key: {masked_key}")
    
    api_key_auth = os.getenv("API_KEY", "")
    logger.info(f"   🛡️ API Auth Key: {'Configured' if api_key_auth else 'Not Configured (Public Mode)'}")

    # Log LLM Models
    logger.info("   🤖 LLM Models:")
    logger.info(f"      - Pre-Retrieval (Graph): {os.getenv('LITELLM_PRE_MODEL', 'gpt-4o-mini')}")
    logger.info(f"      - Planner: {os.getenv('LITELLM_PLANNER_MODEL', 'gpt-4o')}")
    logger.info(f"      - Post-Retrieval: {os.getenv('LITELLM_POST_MODEL', 'gpt-4o')}")

    # 2. Warm up GraphRAG Retriever (Shared Connections)
    try:
        get_shared_retriever()
        logger.info("✅ GraphRAG Retriever warmed up (Qdrant & Neo4j connected)")
    except Exception as e:
        logger.error(f"❌ Failed to warm up GraphRAG Retriever: {e}")

    # 3. Warm up Reranker (if enabled)
    if os.getenv("ENABLE_RERANKER", "true").lower() == "true":
        try:
            from core.reranker import get_reranker
            get_reranker()
            logger.info("✅ Reranker warmed up and ready")
        except Exception as e:
            logger.error(f"❌ Failed to warm up Reranker: {e}")
    else:
        logger.info("ℹ️ Reranker is disabled by environment")

    # 4. Redis Connection
    try:
        await cache_manager.connect()
        logger.info("✅ Redis Cache Manager connected")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")

@app.on_event("shutdown")
async def shutdown():
    """Close Redis and Database connections on shutdown."""
    logger.info("🛑 Shutting down GraphRAG Optimized API")
    await cache_manager.close()
    if _global_retriever:
        _global_retriever.close()

def is_conversational(query: str) -> bool:
    """Check if the query is a simple greeting or common small talk."""
    greetings = {
        "hello", "hi", "hey", "guten tag", "hallo", "moin", "servus", "ciao",
        "good morning", "good afternoon", "good evening", "how are you",
        "wie geht es dir", "was machst du", "wer bist du", "who are you"
    }
    q = query.lower().strip().strip('?!.')
    return q in greetings or len(q) < 3


# Global Retriever Instance for Reuse (Singleton Pattern)
_global_retriever = None

# Tenant Mapping Configuration
TENANT_MAPPING = {
    "bundb": "bundb_app_blaiq_ai_knowledgeglobal_421765297988070tsxTz6itX0BGb9_nXy35B",
    "116.202.24.69": "bundb_app_blaiq_ai_knowledgeglobal_421765297988070tsxTz6itX0BGb9_nXy35B",
    "default": os.getenv("QDRANT_COLLECTION", "graphrag_chunks")
}

def resolve_tenant(request: Request) -> str:
    """Detect tenant from Host header (e.g., bundb.rag.blaiq.ai -> bundb)"""
    host = request.headers.get("host", "")
    # Remove port if present
    base_host = host.split(":")[0].lower()
    
    # Try exact match (for IP)
    if base_host in TENANT_MAPPING:
        return TENANT_MAPPING[base_host]
        
    # Try subdomain match
    subdomain = base_host.split(".")[0].lower()
    collection = TENANT_MAPPING.get(subdomain, TENANT_MAPPING["default"])
    return collection

def get_shared_retriever(request_config: Optional[QueryRequest] = None) -> GraphRAGRetriever:
    """Gets or creates the shared retriever instance."""
    global _global_retriever
    
    # Check if we can reuse the existing one
    if _global_retriever is not None:
        if request_config and any([
            request_config.qdrant_url, 
            request_config.qdrant_api_key,
            request_config.collection_name and request_config.collection_name != _global_retriever.collection_name
        ]):
             logger.info(f"🔄 Creating specialized retriever for tenant: {request_config.collection_name}")
             return GraphRAGRetriever(
                qdrant_url=request_config.qdrant_url,
                qdrant_host=request_config.qdrant_host,
                qdrant_port=request_config.qdrant_port,
                qdrant_api_key=request_config.qdrant_api_key,
                collection_name=request_config.collection_name,
                entity_extraction_prompt=request_config.entity_extraction_prompt,
             )
        return _global_retriever
    
    # Initialize default
    _global_retriever = GraphRAGRetriever()
    return _global_retriever


# ============================================================================
# Optimized GraphRAG Endpoint
# ============================================================================

@app.post("/query/graphrag", response_model=QueryResponse)
async def query_graphrag_optimized(request: QueryRequest):
    """
    Optimized GraphRAG query with caching and parallel execution.
    
    Performance improvements:
    - Redis caching: <100ms for cache hits
    - Parallel retrieval: 2-3x faster than sequential
    - Optimized keyword search: 10-50x faster
    - Expected total time: <2s for cache misses
    """
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    
    total_start = time.time()
    
    # Check cache first
    cached_result = None
    if request.use_cache:
        cached_result = await cache_manager.get(
            request.query,
            request.collection_name
        )
    
    if cached_result:
        # Cache hit - return immediately
        cached_result["cached"] = True
        cached_result["cache_stats"] = cache_manager.get_stats()
        logger.info(f"⚡ CACHE HIT for: {request.query[:50]}...")
        return QueryResponse(**cached_result)
    
    # Check for Conversational Fast Path
    if is_conversational(request.query):
        logger.info(f"💨 Fast Path (Conversational) for: {request.query}")
        result = {
            "query": request.query,
            "answer": "Hello! I am your Enterprise Knowledge Hive. I have access to your corporate documents and can help you analyze risks, find project details, or synthesize information. How can I help you specifically today?",
            "chunks_retrieved": 0,
            "retrieval_stats": {"mode": "fast_path_conversational"},
            "retrieval_time": 0,
            "answer_time": round(time.time() - total_start, 3),
            "total_time": round(time.time() - total_start, 3),
            "cached": False
        }
        return QueryResponse(**result)
    
    # Check for Global Hive Mode
    if request.mode == "global":
        logger.info(f"📊 Global Hive Mode: Strategic summary for '{request.query}'")
        retriever = get_shared_retriever(request)
        
        answer = retriever.generate_global_hive_summary(request.query)
        # We DON'T close the shared retriever here
        
        if not answer:
            raise HTTPException(404, "Could not generate global summary for this graph.")
            
        result = {
            "query": request.query,
            "answer": answer,
            "chunks_retrieved": 0,
            "retrieval_stats": {"mode": "global_hive"},
            "retrieval_time": 0,
            "answer_time": round(time.time() - total_start, 3),
            "total_time": round(time.time() - total_start, 3),
            "cached": False
        }
        return QueryResponse(**result)
    
    # Cache miss - perform retrieval
    retriever = None
    async_wrapper = None

    try:
        retriever = get_shared_retriever(request)
        retriever.debug = request.debug
        async_wrapper = AsyncRetriever(retriever)

        retrieval_start = time.time()

        # ── Unified Planner (1 LLM call: intent + entities + keywords) ──
        plan_start = time.time()
        plan = retriever.plan_retrieval(request.query)
        logger.info(f"⏱️ Unified Planning: {time.time() - plan_start:.3f}s")

        search_plan = plan.get("search_plan", {})
        entities = plan.get("entities", [])
        keywords = plan.get("keywords", [])

        # Build expanded_query from planner keywords (no separate LLM call)
        expanded_query = {
            "original": request.query,
            "keywords": keywords,
            "numbers": [k for k in keywords if any(c.isdigit() for c in k)],
            "years": [], "percentages": [], "expected_patterns": [],
        }

        # ── Parallel Retrieval ──
        broad_k = request.k * 10
        rank_start = time.time()
        rankings, timings, special_results = await async_wrapper.parallel_retrieval(
            request.query, entities, expanded_query, broad_k, plan=search_plan
        )
        logger.info(f"⏱️ Parallel Retrieval: {time.time() - rank_start:.3f}s (Breakdown: {timings})")

        # Adjacent chunks
        all_top = []
        for r_dict in rankings.values():
            all_top.extend(list(r_dict.keys())[:10])
        all_top = list(set(all_top[:30]))
        if all_top:
            adjacent_results = retriever.get_adjacent_chunks(all_top[:20], window_size=1)
            if adjacent_results:
                rankings["adjacent"] = adjacent_results

        # RRF Fusion
        if "graph" in rankings and rankings["graph"]:
            weights = {"graph": 0.40, "vector": 0.45, "keyword": 0.10, "adjacent": 0.05}
        else:
            weights = {"vector": 0.70, "keyword": 0.25, "adjacent": 0.05}
        fused_results = retriever.weighted_rrf_fusion(rankings, weights=weights, k=60)

        # Reranking
        rerank_time = 0.0
        global_reranker_enabled = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        if request.use_reranker and global_reranker_enabled:
            rerank_start = time.time()
            candidates = retriever._retrieve_chunks(fused_results[:request.rerank_top_k])
            if candidates:
                reranker = get_reranker()
                chunks = reranker.rerank(request.query, candidates, top_n=request.k)
            else:
                chunks = retriever._retrieve_chunks(fused_results[:request.k])
            rerank_time = time.time() - rerank_start
        else:
            chunks = retriever._retrieve_chunks(fused_results[:request.k])

        retrieval_time = time.time() - retrieval_start
        logger.info(f"🚀 TOTAL Retrieval: {retrieval_time:.3f}s")

        if not chunks:
            raise HTTPException(404, "No relevant chunks found")

        # ── Answer Generation (with integrated formatting) ──
        answer = ""
        answer_time = 0.0

        if request.generate_answer:
            if not os.getenv("OPENAI_API_KEY"):
                raise HTTPException(500, "OPENAI_API_KEY not configured")

            answer_start = time.time()
            answer = generate_answer(
                request.query,
                chunks,
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
            )
            answer_time = time.time() - answer_start

        total_time = time.time() - total_start

        stats = {
            "total_candidates": len(fused_results),
            "graph_chunks": len(rankings.get("graph", {})),
            "vector_chunks": len(rankings.get("vector", {})),
            "keyword_chunks": len(rankings.get("keyword", {})),
            "entities_extracted": entities,
            "parallel_timings": timings,
            "rerank_time": round(rerank_time, 3),
            "filter_label": retriever.filter_label,
        }

        result = {
            "query": request.query,
            "answer": answer,
            "chunks_retrieved": len(chunks),
            "retrieval_stats": stats,
            "retrieval_time": round(retrieval_time, 3),
            "answer_time": round(answer_time, 3),
            "total_time": round(total_time, 3),
            "cached": False,
            "cache_stats": cache_manager.get_stats()
        }

        if request.use_cache:
            await cache_manager.set(request.query, result, request.collection_name)

        return QueryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"GraphRAG query failed: {str(e)}")
    finally:
        if async_wrapper:
            async_wrapper.shutdown()


# ============================================================================
# Health & Stats
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "GraphRAG Retriever API (Optimized)",
        "version": "4.0.0",
        "optimizations": [
            "Redis caching (1-hour TTL)",
            "Parallel Vector + Graph + Keyword search",
            "Elite Status Reranking (BGE-Reranker)",
            "Optimized Qdrant keyword search",
            "Async execution"
        ],
        "cache_stats": cache_manager.get_stats()
    }


@app.get("/config")
async def get_config():
    """Get public configuration for the UI."""
    return {
        "qdrant_collection": os.getenv("QDRANT_COLLECTION", "graphrag_chunks"),
        "default_mode": "local",
        "rerank_enabled": os.getenv("ENABLE_RERANKER", "true").lower() == "true"
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return cache_manager.get_stats()


@app.post("/query/graphrag/stream")
async def query_graphrag_stream(request_data: QueryRequest, http_request: Request):
    """
    Streaming version of GraphRAG query with session persistence and multi-tenancy.
    """
    # 1. Resolve Tenant
    tenant_collection = request_data.collection_name or resolve_tenant(http_request)
    request_data.collection_name = tenant_collection
    
    if not request_data.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    async def stream_generator():
        total_start = time.time()

        # ── 1. Load History ──
        history_str = ""
        past_messages = []
        if request_data.session_id:
            logger.info(f"📜 Loading history for session: {request_data.session_id} (Tenant: {tenant_collection})")
            past_messages = await session_manager.get_history(tenant_collection, request_data.session_id, limit=8)
            for msg in past_messages:
                history_str += f"{msg['role'].upper()}: {msg['content']}\n"
            await session_manager.add_message(tenant_collection, request_data.session_id, "user", request_data.query)

        metrics = {
            "timings": {},
            "sources": {"vector": 0, "graph": 0, "keyword": 0},
            "chunks_retrieved": 0,
            "tenant": tenant_collection
        }

        retriever = get_shared_retriever(request_data)
        async_wrapper = AsyncRetriever(retriever)

        # ── 2. Query Rewrite (conditional, no LLM if not context-dependent) ──
        context_query = retriever.rewrite_query(request_data.query, history_str)
        if context_query != request_data.query:
            logger.info(f"🔄 Rewrite: '{request_data.query}' -> '{context_query}'")

        # Fast path for greetings
        if is_conversational(request_data.query) and not history_str:
            yield f"data: {json.dumps({'answer': 'Hallo! Ich bin BLAIQ, Ihr strategischer Wissenspartner. Wie kann ich Ihnen helfen?'})}\n\n"
            yield f"data: {json.dumps({'metrics': metrics})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # ── 3. UNIFIED PLANNER (1 LLM call: intent + entities + keywords) ──
        yield f"data: {json.dumps({'log': '🎯 Strategic Planning...'})}\n\n"
        plan_start = time.time()
        plan = retriever.plan_retrieval(context_query)
        metrics["timings"]["planning"] = round(time.time() - plan_start, 3)

        search_plan = plan.get("search_plan", {})
        entities = plan.get("entities", [])
        keywords = plan.get("keywords", [])

        yield f"data: {json.dumps({'planning': plan})}\n\n"

        logger.info(json.dumps({
            "event": "pipeline_start",
            "session_id": request_data.session_id,
            "query": request_data.query,
            "rewritten_query": context_query,
            "timestamp": time.time(),
            "intent": plan.get("mode", "UNKNOWN"),
            "planner_routes": search_plan,
            "entities": entities,
            "keywords": keywords[:8]
        }, indent=2, ensure_ascii=False))

        # Handle SMALL_TALK from planner
        if plan.get("mode") == "SMALL_TALK":
            reply = plan.get("direct_reply") or "Hallo! Wie kann ich Ihnen helfen?"
            yield f"data: {json.dumps({'delta': reply})}\n\n"
            yield f"data: {json.dumps({'metrics': metrics})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # ── 4. Build expanded_query from planner keywords (NO separate LLM call) ──
        expanded_query = {
            "original": context_query,
            "keywords": keywords,
            "numbers": [k for k in keywords if any(c.isdigit() for c in k)],
            "years": [],
            "percentages": [],
            "expected_patterns": [],
        }

        # ── 5. PARALLEL RETRIEVAL (vector + graph + keyword) ──
        active_sources = sum(1 for v in [search_plan.get("use_vector"), search_plan.get("use_graph"), search_plan.get("use_keyword")] if v)
        yield f"data: {json.dumps({'log': f'🚀 Parallel Retrieval ({active_sources} sources)...'})}\n\n"

        retrieval_start = time.time()
        rankings, timings, special_results = await async_wrapper.parallel_retrieval(
            context_query, entities, expanded_query, request_data.k * 20,
            plan=search_plan
        )

        metrics["sources"] = {k: len(v) for k, v in rankings.items()}
        metrics["timings"]["parallel_retrieval"] = timings

        fused_results = retriever.weighted_rrf_fusion(rankings, k=60)
        chunks = retriever._retrieve_chunks(fused_results[:request_data.k])

        # Inject HiveMind if available
        if special_results.get("hive_mind"):
            logger.info("🐝 Including HiveMind Global Summary")
            hive_doc = Document(
                page_content=f"*** HIVE MIND INTELLIGENCE ***\n{special_results['hive_mind']}",
                metadata={"source": "HiveMind", "score": 1.0, "is_hive_mind": True}
            )
            chunks.insert(0, hive_doc)

        metrics["chunks_retrieved"] = len(chunks)
        metrics["timings"]["total_retrieval"] = round(time.time() - retrieval_start, 3)

        retrieval_ms = round(metrics["timings"]["total_retrieval"] * 1000)
        yield f"data: {json.dumps({'log': f'✅ Retrieved {len(chunks)} chunks in {retrieval_ms}ms'})}\n\n"

        # ── 6. ANSWER GENERATION with integrated formatting (1 LLM call) ──
        yield f"data: {json.dumps({'log': '🧬 Generating response...'})}\n\n"

        llm_start = time.time()
        first_token_time = None
        final_answer = ""

        from retriever.graphrag_retriever import generate_answer_stream

        # Load formatting template if content_mode is set
        content_mode = request_data.content_mode
        format_template = None
        if content_mode and content_mode.upper() != "DEFAULT":
            format_template = retriever.loader.load_template(content_mode)
            if format_template:
                logger.info(f"📋 Injecting {content_mode.upper()} template into response generation")
            else:
                logger.warning(f"⚠️ Template for '{content_mode}' not found, using default format")

        log_llm_event("llm_start", {"model": retriever.post_model})

        response_stream = generate_answer_stream(
            context_query,
            chunks,
            history=past_messages,
            model=retriever.post_model,
            system_prompt=retriever.response_generator_prompt,
            content_mode=content_mode,
            format_template=format_template,
        )

        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                content = chunk.choices[0].delta.content
                final_answer += content
                yield f"data: {json.dumps({'delta': content})}\n\n"

        # ── 7. Save to History ──
        if request_data.session_id:
            await session_manager.add_message(tenant_collection, request_data.session_id, "assistant", final_answer)

        # ── 8. Metrics ──
        metrics["timings"]["llm_generation"] = round(time.time() - llm_start, 3)
        metrics["timings"]["total"] = round(time.time() - total_start, 3)
        if first_token_time:
            metrics["timings"]["ttfc_ms"] = round((first_token_time - llm_start) * 1000, 1)

        logger.info(json.dumps({
            "event": "llm_complete",
            "response_length": len(final_answer),
            "total_latency_ms": round(metrics["timings"]["total"] * 1000),
            "llm_latency_ms": round(metrics["timings"]["llm_generation"] * 1000),
            "ttfc_ms": metrics["timings"].get("ttfc_ms", 0),
            "retrieval_latency_ms": round(metrics["timings"]["total_retrieval"] * 1000),
            "timestamp": time.time()
        }, indent=2))

        yield f"data: {json.dumps({'metrics': metrics})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.get("/history/{session_id}")
async def get_chat_history(session_id: str, http_request: Request):
    """Retrieve history for a session."""
    tenant = resolve_tenant(http_request)
    past_messages = await session_manager.get_history(tenant, session_id)
    return {"session_id": session_id, "tenant": tenant, "history": past_messages}


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cache entries"""
    deleted = await cache_manager.clear()
    return {"status": "success", "deleted": deleted}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
