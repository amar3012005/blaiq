# python src/retriever/retriever_api.py

"""
GraphRAG Retriever API
Query your processed documents using RAG or GraphRAG retrieval modes.
Supports configurable prompts and optional answer generation.

Endpoints:
- POST /query/rag      - Vector + Keyword search (Qdrant only)
- POST /query/graphrag - Hybrid: Neo4j Graph + Qdrant Vector + Keyword
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio

from utils.auth import verify_api_key
from utils.job_store import JobStore

from .graphrag_retriever import (
    GraphRAGRetriever,
)
from .graphrag_retriever import (
    generate_answer as graphrag_generate_answer,
    generate_answer_stream as graphrag_generate_answer_stream,
)
from .rag_retriever import (
    RAGRetriever,
)
from .rag_retriever import (
    generate_answer as rag_generate_answer,
)

load_dotenv()

retriever_job_store = JobStore(storage_dir="_jobs", api_name="retriever")

app = FastAPI(
    title="GraphRAG Retriever API",
    description="""
Query documents using RAG or GraphRAG retrieval modes.

## Endpoints

- **POST /query/rag** - Vector + Keyword search using Qdrant
- **POST /query/graphrag** - Hybrid search using Neo4j graph + Qdrant vectors

## Features

- Configurable system and user prompts
- Optional answer generation (return only context if needed)
- Multi-tenant support via collection_name/filter_label
    """,
    version="3.0.0",
    dependencies=[Depends(verify_api_key)],
)


# ============================================================================
# Request/Response Models
# ============================================================================


class QueryRequest(BaseModel):
    """Query request model for both RAG and GraphRAG endpoints"""

    query: str = Field(..., description="Your question", min_length=1)
    k: int = Field(default=20, description="Number of chunks to retrieve", ge=1)
    debug: bool = Field(default=False, description="Enable debug mode")

    # Answer generation control
    generate_answer: bool = Field(
        default=True,
        description="If True, generate LLM answer. If False, return only retrieved chunks.",
    )

    # Custom prompts
    system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt for LLM (uses default if not provided)"
    )
    user_prompt: Optional[str] = Field(
        default=None,
        description="Custom user prompt template. Use {context} and {query} as placeholders.",
    )

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # 251204–BundB Jun – BEGIN
    # NEW: custom entity extraction prompt (GraphRAG only)
    entity_extraction_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Custom entity extraction prompt template for GraphRAG. " "Use {query} as placeholder."
        ),
    )
    # 251204–BundB Jun – END
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Qdrant configuration
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant URL (overrides config)")
    qdrant_host: Optional[str] = Field(default=None, description="Qdrant host (overrides config)")
    qdrant_port: Optional[int] = Field(default=None, description="Qdrant port (overrides config)")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    collection_name: Optional[str] = Field(
        default=None,
        description="Collection name. For GraphRAG, also used as filter_label for Neo4j.",
    )

    # Graph visualization (GraphRAG only)
    include_graph: bool = Field(
        default=False,
        description="Include Mermaid graph visualization in response. Render to SVG in Node.js.",
    )
    graph_depth: int = Field(
        default=1,
        description="Graph traversal depth (1 = direct connections, 2 = 2 hops). Max 3.",
        ge=1,
        le=3,
    )
    
    # 260215-BundB Jun - Add history support
    history: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="Conversation history list of messages [{'role': 'user', 'content': '...'}, ...]"
    )


class GraphEntity(BaseModel):
    """Entity in the knowledge graph"""

    id: str
    name: str
    type: str


class GraphRelationship(BaseModel):
    """Relationship between entities"""

    source_id: str
    target_id: str
    type: str


class GraphVisualization(BaseModel):
    """Graph visualization data with Mermaid code"""

    mermaid_code: str = Field(description="Mermaid diagram code - render to SVG in Node.js")
    nodes: int = Field(description="Number of nodes in graph")
    edges: int = Field(description="Number of edges in graph")
    entities: List[GraphEntity] = Field(description="Raw entity data")
    relationships: List[GraphRelationship] = Field(description="Raw relationship data")


class ChunkDetail(BaseModel):
    """Detailed chunk information"""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    score: float
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Response when generate_answer=True"""

    query: str
    answer: str
    chunks_retrieved: int
    retrieval_stats: Dict[str, Any]
    retrieval_time: float
    answer_time: float
    total_time: float
    graph: Optional[GraphVisualization] = Field(
        default=None, description="Graph visualization data (only if include_graph=True)"
    )


class ContextOnlyResponse(BaseModel):
    """Response when generate_answer=False"""

    query: str
    chunks: List[ChunkDetail]
    retrieval_stats: Dict[str, Any]
    retrieval_time: float
    graph: Optional[GraphVisualization] = Field(
        default=None, description="Graph visualization data (only if include_graph=True)"
    )


# ============================================================================
# Health Check Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Health check and service info"""
    return {
        "status": "healthy",
        "service": "GraphRAG Retriever API",
        "version": "3.0.0",
        "endpoints": {
            "rag": "/query/rag",
            "graphrag": "/query/graphrag",
        },
        "features": {
            "vector_search": True,
            "keyword_search": True,
            "graph_search": True,
            "custom_prompts": True,
            "context_only_mode": True,
            "multi_tenant": True,
        },
        "llm": {
            "provider": "OpenAI-compatible",
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "base_url": os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
        },
        "prompt_placeholders": {
            "user_prompt": ["{context}", "{query}"],
            "entity_extraction_prompt": ["{query}"],  # 251204–BundB Jun
            "description": "Use these placeholders in custom prompts",
        },
    }


@app.get("/status")
async def get_status():
    """Check system status and connections"""
    status = {
        "api": "online",
        "qdrant": "unknown",
        "neo4j": "unknown",
        "llm": "unknown",
    }

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient

        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
        else:
            client = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333))
            )
        client.get_collections()
        status["qdrant"] = "connected"
    except Exception as e:
        status["qdrant"] = f"error: {str(e)[:50]}"

    # Check Neo4j
    try:
        from neo4j import GraphDatabase

        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if neo4j_password:
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            status["neo4j"] = "connected"
        else:
            status["neo4j"] = "not configured (NEO4J_PASSWORD missing)"
    except Exception as e:
        status["neo4j"] = f"error: {str(e)[:50]}"

    # Check LLM
    if os.getenv("OPENAI_API_KEY"):
        status["llm"] = "configured"
    else:
        status["llm"] = "not configured (OPENAI_API_KEY missing)"

    return status


# ============================================================================
# RAG Endpoint (Vector + Keyword)
# ============================================================================


@app.post("/query/rag")
async def query_rag(request: QueryRequest):
    """
    RAG Query - Vector + Keyword search using Qdrant only.

    This endpoint performs:
    1. Query expansion
    2. Vector similarity search
    3. Keyword search
    4. RRF fusion of results
    5. Optional: LLM answer generation

    **Parameters:**
    - **query**: Your question (required)
    - **k**: Number of chunks to retrieve (default: 20)
    - **generate_answer**: If False, returns only chunks without LLM call
    - **system_prompt**: Custom system prompt for LLM
    - **user_prompt**: Custom user prompt template (use {context} and {query})
    - **collection_name**: Qdrant collection to search
    """
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    retriever = None
    try:
        # Initialize RAG retriever
        retriever = RAGRetriever(
            debug=request.debug,
            qdrant_url=request.qdrant_url,
            qdrant_host=request.qdrant_host,
            qdrant_port=request.qdrant_port,
            qdrant_api_key=request.qdrant_api_key,
            collection_name=request.collection_name,
        )

        # Retrieve chunks
        retrieval_start = time.time()
        chunks, stats = retriever.rag_retrieval(request.query, k=request.k, debug=request.debug)
        retrieval_time = time.time() - retrieval_start

        if not chunks:
            raise HTTPException(404, "No relevant chunks found for your query")

        # If generate_answer=False, return only chunks
        if not request.generate_answer:
            chunk_details = [
                ChunkDetail(
                    chunk_id=chunk.metadata.get("chunk_id", "unknown"),
                    doc_id=chunk.metadata.get("doc_id", "unknown"),
                    chunk_index=chunk.metadata.get("chunk_index", 0),
                    text=chunk.page_content,
                    score=chunk.metadata.get("fusion_score", 0.0),
                    metadata={
                        "qdrant_id": chunk.metadata.get("qdrant_id"),
                        "retrieval_rank": chunk.metadata.get("retrieval_rank"),
                    },
                )
                for chunk in chunks
            ]

            return ContextOnlyResponse(
                query=request.query,
                chunks=chunk_details,
                retrieval_stats=stats,
                retrieval_time=round(retrieval_time, 2),
            )

        # Generate answer with LLM
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(500, "OPENAI_API_KEY not configured")

        answer_start = time.time()
        answer = rag_generate_answer(
            request.query,
            chunks,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
        )
        answer_time = time.time() - answer_start

        total_time = retrieval_time + answer_time

        # Log job
        job_id = f"rag_{int(time.time() * 1000)}"
        retriever_job_store.save_job(
            job_id,
            {
                "status": "completed",
                "created_at": time.time(),
                "mode": "rag",
                "query": request.query,
                "chunks_retrieved": len(chunks),
                "retrieval_time": retrieval_time,
                "answer_time": answer_time,
            },
        )

        return QueryResponse(
            query=request.query,
            answer=answer,
            chunks_retrieved=len(chunks),
            retrieval_stats=stats,
            retrieval_time=round(retrieval_time, 2),
            answer_time=round(answer_time, 2),
            total_time=round(total_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"RAG query failed: {str(e)}")
    finally:
        if retriever is not None:
            try:
                retriever.close()
            except Exception:
                pass  # Ignore close errors


# ============================================================================
# GraphRAG Endpoint (Neo4j + Qdrant)
# ============================================================================


@app.post("/query/graphrag")
async def query_graphrag(request: QueryRequest):
    """
    GraphRAG Query - Hybrid search using Neo4j graph + Qdrant vectors.

    This endpoint performs:
    1. LLM-based entity extraction from query
    2. Graph traversal in Neo4j (filtered by collection_name)
    3. Vector similarity search in Qdrant
    4. Keyword search
    5. Weighted RRF fusion of all results
    6. Optional: LLM answer generation
    7. Optional: Mermaid graph visualization

    **Parameters:**
    - **query**: Your question (required)
    - **k**: Number of chunks to retrieve (default: 20)
    - **generate_answer**: If False, returns only chunks without LLM call
    - **system_prompt**: Custom system prompt for LLM
    - **user_prompt**: Custom user prompt template (use {context} and {query})
    - **entity_extraction_prompt**: Custom prompt for LLM-based entity extraction (use {query})
    - **collection_name**: Used as filter_label to filter Neo4j graph data
    - **include_graph**: If True, include Mermaid code for graph visualization
    - **graph_depth**: Graph traversal depth (1-3 hops)
    """
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    retriever = None
    try:
        # Initialize GraphRAG retriever
        retriever = GraphRAGRetriever(
            debug=request.debug,
            qdrant_url=request.qdrant_url,
            qdrant_host=request.qdrant_host,
            qdrant_port=request.qdrant_port,
            qdrant_api_key=request.qdrant_api_key,
            collection_name=request.collection_name,
            entity_extraction_prompt=request.entity_extraction_prompt,  # 251204–BundB Jun: pass through entity extraction prompt from request
        )

        # Retrieve chunks with graph + vector + keyword
        retrieval_start = time.time()
        chunks, stats = retriever.graphrag_retrieval(
            request.query, k=request.k, debug=request.debug
        )
        retrieval_time = time.time() - retrieval_start

        if not chunks:
            raise HTTPException(404, "No relevant chunks found for your query")

        # Optional: Get graph visualization
        graph_data = None
        if request.include_graph:
            try:
                entities_extracted = stats.get("entities_extracted", [])
                graph_result = retriever.get_graph_visualization(
                    entities=entities_extracted, depth=request.graph_depth
                )
                if graph_result:
                    graph_data = GraphVisualization(
                        mermaid_code=graph_result["mermaid_code"],
                        nodes=graph_result["nodes"],
                        edges=graph_result["edges"],
                        entities=[
                            GraphEntity(id=e["id"], name=e["name"], type=e["type"])
                            for e in graph_result["entities"]
                        ],
                        relationships=[
                            GraphRelationship(
                                source_id=r["source_id"], target_id=r["target_id"], type=r["type"]
                            )
                            for r in graph_result["relationships"]
                        ],
                    )
            except Exception as e:
                if request.debug:
                    print(f"  ⚠️ Graph visualization failed: {e}")

        # If generate_answer=False, return only chunks
        if not request.generate_answer:
            chunk_details = [
                ChunkDetail(
                    chunk_id=chunk.metadata.get("chunk_id", "unknown"),
                    doc_id=chunk.metadata.get("doc_id", "unknown"),
                    chunk_index=chunk.metadata.get("chunk_index", 0),
                    text=chunk.page_content,
                    score=chunk.metadata.get("fusion_score", 0.0),
                    metadata={
                        "qdrant_id": chunk.metadata.get("qdrant_id"),
                        "retrieval_rank": chunk.metadata.get("retrieval_rank"),
                    },
                )
                for chunk in chunks
            ]

            return ContextOnlyResponse(
                query=request.query,
                chunks=chunk_details,
                retrieval_stats=stats,
                retrieval_time=round(retrieval_time, 2),
                graph=graph_data,
            )

        # Generate answer with LLM
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(500, "OPENAI_API_KEY not configured")

        answer_start = time.time()
        answer = graphrag_generate_answer(
            request.query,
            chunks,
            system_prompt=request.system_prompt or retriever.response_generator_prompt,
            user_prompt=request.user_prompt,
        )
        answer_time = time.time() - answer_start

        total_time = retrieval_time + answer_time

        # Log job
        job_id = f"graphrag_{int(time.time() * 1000)}"
        retriever_job_store.save_job(
            job_id,
            {
                "status": "completed",
                "created_at": time.time(),
                "mode": "graphrag",
                "query": request.query,
                "chunks_retrieved": len(chunks),
                "entities_extracted": stats.get("entities_extracted", []),
                "graph_chunks": stats.get("graph_chunks", 0),
                "retrieval_time": retrieval_time,
                "answer_time": answer_time,
                "filter_label": stats.get("filter_label"),
                "graph_included": request.include_graph,
            },
        )

        return QueryResponse(
            query=request.query,
            answer=answer,
            chunks_retrieved=len(chunks),
            retrieval_stats=stats,
            retrieval_time=round(retrieval_time, 2),
            answer_time=round(answer_time, 2),
            total_time=round(total_time, 2),
            graph=graph_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"GraphRAG query failed: {str(e)}")
    finally:
        if retriever is not None:
            try:
                retriever.close()
            except Exception:
                pass  # Ignore close errors


# ============================================================================
# GraphRAG Streaming Endpoint
# ============================================================================


@app.post("/query/graphrag/stream")
async def query_graphrag_stream(request: QueryRequest):
    """
    GraphRAG Query (Streaming) - Server-Sent Events (SSE).
    
    Yields events:
    - retrieval_status: {"step": "init", ...}
    - retrieval_result: {"chunks_count": 5, "stats": ...}
    - graph_data: {"mermaid_code": ...} (optional)
    - token: {"content": "The"}
    - error: {"message": "..."}
    - done: {"total_time": ...}
    """
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    async def event_generator():
        retriever = None
        start_time = time.time()
        
        try:
            # 1. Notify Start
            yield f"event: retrieval_status\ndata: {json.dumps({'step': 'starting', 'query': request.query})}\n\n"
            
            # 2. Initialize Retriever
            retriever = GraphRAGRetriever(
                debug=request.debug,
                qdrant_url=request.qdrant_url,
                qdrant_host=request.qdrant_host,
                qdrant_port=request.qdrant_port,
                qdrant_api_key=request.qdrant_api_key,
                collection_name=request.collection_name,
                entity_extraction_prompt=request.entity_extraction_prompt,
            )
            
            # Helper to stream status updates
            async def status_updater(status):
                # status is dict {"step": str, "details": str}
                await queue.put(f"event: retrieval_status\ndata: {json.dumps(status)}\n\n")

            # 3. Perform Retrieval
            retrieval_start = time.time()
            # We use a Queue to bridge the callback to the generator stream
            # But here we are inside the generator, so we can yield directly? NO.
            # Generator methods are tricky. 
            # Best way: Use the callback to just yield? 'yield' is not allowed in nested function.
            # We can use a shared list or queue, but since we await the retrieval, we can't yield concurrently unless we run retrieval in background task.
            
            # OPTION 2: Passed callback simply prints for now? NO, user wants to see it.
            # We must run retrieval in a Task and consume a Queue.
            
            queue = asyncio.Queue()
            
            async def run_retrieval():
                try:
                    chunks, stats = await retriever.graphrag_retrieval_async(
                        request.query, k=request.k, debug=request.debug, status_callback=status_updater
                    )
                    await queue.put({"type": "result", "chunks": chunks, "stats": stats})
                except Exception as e:
                    await queue.put({"type": "error", "error": str(e)})
                finally:
                    await queue.put(None) # Sentinel

            # Start background task
            asyncio.create_task(run_retrieval())

            chunks = []
            stats = {}
            
            # Consume queue
            while True:
                item = await queue.get()
                if item is None:
                    break
                
                if isinstance(item, str):
                    # It's an SSE event string from status_updater
                    yield item
                elif isinstance(item, dict):
                    if "type" in item:
                        if item["type"] == "error":
                             yield f"event: error\ndata: {json.dumps({'message': item['error']})}\n\n"
                             return
                        elif item["type"] == "result":
                             chunks = item["chunks"]
                             stats = item["stats"]
            
            retrieval_time = time.time() - retrieval_start
            
            if not chunks:
                 yield f"event: error\ndata: {json.dumps({'message': 'No relevant chunks found'})}\n\n"
                 return
            
            # 4. Stream Retrieval Results
            yield f"event: retrieval_result\ndata: {json.dumps({'chunks_count': len(chunks), 'retrieval_time': round(retrieval_time, 2), 'stats': stats})}\n\n"
            
            # 5. Graph Visualization (Optional)
            if request.include_graph:
                try:
                    entities_extracted = stats.get("entities_extracted", [])
                    graph_result = retriever.get_graph_visualization(
                        entities=entities_extracted, depth=request.graph_depth
                    )
                    if graph_result:
                         # Serialize graph data - reusing Pydantic models via dict()
                         graph_data = GraphVisualization(
                            mermaid_code=graph_result["mermaid_code"],
                            nodes=graph_result["nodes"],
                            edges=graph_result["edges"],
                            entities=[
                                GraphEntity(id=e["id"], name=e["name"], type=e["type"])
                                for e in graph_result["entities"]
                            ],
                            relationships=[
                                GraphRelationship(
                                    source_id=r["source_id"], target_id=r["target_id"], type=r["type"]
                                )
                                for r in graph_result["relationships"]
                            ],
                        ).dict()
                         yield f"event: graph_data\ndata: {json.dumps(graph_data)}\n\n"
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'message': f'Graph viz failed: {str(e)}'})}\n\n"

            # 6. Stream Answer Generation
            if request.generate_answer:
                if not os.getenv("OPENAI_API_KEY"):
                     yield f"event: error\ndata: {json.dumps({'message': 'OPENAI_API_KEY not configured'})}\n\n"
                     return

                # Assuming retriever.response_generator_prompt is available
                system_prompt = request.system_prompt or retriever.response_generator_prompt
                
                stream = graphrag_generate_answer_stream(
                    request.query,
                    chunks,
                    history=request.history, 
                    system_prompt=system_prompt,
                    user_prompt=request.user_prompt,
                )
                
                # Iterate over LiteLLM/OpenAI stream
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                         yield f"event: token\ndata: {json.dumps({'content': content})}\n\n"
            
            total_time = time.time() - start_time
            yield f"event: done\ndata: {json.dumps({'total_time': round(total_time, 2)})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            if retriever:
                try:
                    retriever.close()
                except:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
