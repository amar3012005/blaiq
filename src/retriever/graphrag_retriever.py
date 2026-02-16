# src/utils/llm_logger.py or directly in the file
# Change log path to avoid Docker permission issues
LOG_PATH = "/tmp/llm_error.log"

def log_llm_event(event_type: str, data: dict) -> None:
    """
    event_type example:
      - "llm_error"
      - "llm_fallback_success"
    """
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event_type,
        **data,
    }

    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

"""
GraphRAG Retriever - Hybrid Graph + Vector + Keyword Search
Uses Neo4j for graph traversal, Qdrant for vector search, and LLM for entity extraction.

Multi-Tenant Support:
- filter_label (derived from collection_name) isolates all Neo4j queries
- Same entity name in different tenants = different graph nodes
- All graph traversals are strictly within tenant boundaries
"""

import json
import asyncio
import logging

import math
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from langchain_core.documents import Document
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue

from utils.bge_m3_embedding import BGEM3Embeddings
from utils.qdrant_helpers import compute_point_id

# 260108-BundB Jun – For error log.
import time
import uuid
from utils.llm_logger import log_llm_event
from prompts.prompt_loader import PromptLoader


# Suppress warnings
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logging.getLogger("neo4j.io").setLevel(logging.ERROR)
logging.getLogger("qdrant_client").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    config_file = Path(config_path)

    if not config_file.exists():
        config_file = Path("..") / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


# Load configuration
CONFIG = load_config()

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST") or CONFIG.get("qdrant", {}).get("host", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", CONFIG.get("qdrant", {}).get("port", 6333)))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# Prioritize environment variable over config file
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION") or CONFIG.get("qdrant", {}).get("collection_name", "graphrag_chunks")

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI") or CONFIG.get("neo4j", {}).get("uri", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER") or CONFIG.get("neo4j", {}).get("user", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# LLM Model Configuration (Groq / OpenAI SDK)
LITELLM_PLANNER_MODEL = os.getenv("LITELLM_PLANNER_MODEL") or "groq/llama-3.1-8b-instant"
LITELLM_PRE_MODEL = os.getenv("LITELLM_PRE_MODEL") or "groq/llama-3.1-8b-instant"
LITELLM_POST_MODEL = os.getenv("LITELLM_POST_MODEL") or "openai/gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "25"))
_token_env = os.getenv("LLM_MAX_OUTPUT_TOKENS", "4000")
LLM_MAX_OUTPUT_TOKENS = int(_token_env) if _token_env and _token_env.strip().isdigit() and int(_token_env) > 0 else None

# Pre-Retrieval: The Search Architect Prompt
PRE_RETRIEVAL_SYSTEM_PROMPT = """You are the 'Search Architect' for a high-performance GraphRAG system.
Your mission is to extract key entities and keywords from the user's query to enable a robust multi-database search (Vector + Graph + Keyword).

CRITICAL INSTRUCTIONS:
1. LANGUAGE NORMALIZATION: The knowledge base is written in GERMAN. 
   - If the query is in English (or any other language), extract the concepts and translate the search terms into German.
   - Example (Query: "Who is the supervisor?"): Keywords: ["Abteilungsleiter", "Vorgesetzter", "Leitender Angestellter"]
2. ENTITY TYPES: Extract Persons, Organizations, Locations, Concepts, and specific Numbers/Dates.
3. OUTPUT FORMAT: Respond ONLY with a raw JSON array of strings. No explanations.

Example Output: ["Bundesregierung", "KI-Strategie", "2024", "Mittelstand"]"""

# Strategic Planner: The Dispatcher Prompt (ENHANCED)
STRATEGIC_PLANNER_PROMPT = """You are the 'Strategic Retrieval Planner'. Your task is to analyze the user's query and determine the OPTIMAL execution plan.

## QUERY CLASSIFICATION RULES:

### 1. SMALL_TALK (No document retrieval needed)
Triggers:
- Greetings: "Hello", "Hi", "Good morning", "Hallo", "Guten Tag"
- Identity questions: "Who are you?", "What can you do?", "Wer bist du?"
- Thanks/Farewell: "Thanks", "Goodbye", "Danke", "Tschüss"
- System questions: "Are you online?", "How does this work?"
- Casual: "Okay", "Yes", "No", "Sure"

Action: Skip retrieval entirely. Provide a direct, friendly response.

### 2. DOCUMENT_SEARCH (Precise document retrieval required)
Triggers:
- Specific entity mentions: "Siemens", "Project Alpha", "Contract #12345"
- Document references: "What does the report say?", "Find the contract", "Show me the proposal"
- Factual questions: "What is the budget?", "When was it signed?", "Who is responsible?"
- Comparison requests: "Compare X and Y", "What are the differences?"

Action: Use vector + keyword search. Graph only if entities are involved.

### 3. ANALYTICAL_SEARCH (Cross-document analysis)
Triggers:
- Pattern questions: "What are the common risks?", "What trends do you see?"
- Summary requests: "Summarize all projects", "Give me an overview"
- Strategic questions: "What should we prioritize?", "What are the key takeaways?"

Action: Use all retrieval methods. Prioritize graph for entity relationships.

### 4. CLARIFICATION_NEEDED (Ambiguous query)
Triggers:
- Very short queries (1-2 words) that aren't greetings: "costs", "status"
- Context-dependent: "What about that?", "And the other one?"
- Incomplete: "The project...", "Can you..."

Action: Ask for clarification before retrieval.

## OUTPUT JSON SPECIFICATION:
{
  "mode": "SMALL_TALK" | "DOCUMENT_SEARCH" | "ANALYTICAL_SEARCH" | "CLARIFICATION_NEEDED",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of classification",
  "search_plan": {
    "use_vector": boolean,
    "use_graph": boolean,
    "use_keyword": boolean,
    "priority_source": "vector" | "graph" | "keyword" | null
  },
  "entities_extracted": ["List of entities in query (people, orgs, projects, dates)"],
  "entities_german": ["Same entities translated to German for search optimization"],
  "direct_reply": "Only for SMALL_TALK or CLARIFICATION_NEEDED, otherwise null",
  "suggested_keywords": ["Additional search terms that might help"]
}

Respond ONLY with valid JSON. No markdown, no explanations outside JSON."""

# Post-Retrieval: The Answer Synthesis Prompt (ENHANCED)
POST_RETRIEVAL_SYSTEM_PROMPT = """You are an expert Corporate Knowledge Analyst. Your role is to synthesize information from retrieved document chunks into precise, actionable answers.

## CORE PRINCIPLES:

### 1. LANGUAGE MATCHING (MANDATORY)
- Detect the user's query language and respond in THE SAME LANGUAGE.
- If query is German → respond in German.
- If query is English → respond in English.
- The source documents may be in German, but you MUST translate insights to match the query language.

### 2. PRECISION OVER VERBOSITY
- Answer the EXACT question asked. Do not provide tangential information.
- If a user asks "What is the budget?", give the budget, not a summary of the entire project.
- If multiple documents contain the answer, prioritize the most recent or most authoritative.

### 3. SOURCE ATTRIBUTION (ALWAYS)
- Every factual claim MUST be linked to a source.
- Format: "According to [Document Name], Page X, ..."
- If information comes from multiple sources, list all.
- Clean document names by removing hash suffixes (e.g., "_89d2d7b8" → ".pdf").

### 4. HANDLING MISSING INFORMATION
- If the answer is NOT in the provided context, say so explicitly.
- Do NOT guess, hallucinate, or use external knowledge.
- Suggest what additional documents might help if applicable.

### 5. STRUCTURED RESPONSES
For complex queries, use this structure:
- **ANSWER**: Direct response to the question
- **DETAILS**: Supporting information (if needed)
- **SOURCES**: Document citations

### 6. ENTITY AWARENESS
- Pay attention to specific entities (people, organizations, dates, amounts).
- When entities are mentioned, ensure your answer addresses them specifically.
- If an entity appears in multiple documents, note any discrepancies.
"""

BGE_M3_DIMENSION = 1024

# Default System Prompt (for standard queries)
DEFAULT_SYSTEM_PROMPT = """You are a friendly and precise corporate document assistant.

## YOUR TASK:
Answer questions accurately using ONLY the provided document context.

## LANGUAGE RULE:
Always respond in the SAME LANGUAGE as the user's question.

## ANSWER QUALITY STANDARDS:
1. **Accuracy**: Only state facts that appear in the provided context.
2. **Completeness**: Address all parts of multi-part questions.
3. **Clarity**: Use simple, professional language. Avoid jargon unless the user used it.
4. **Citations**: Reference source documents with cleaned names and page numbers.

## WHEN YOU DON'T KNOW:
If the information is not in the context, respond:
"I could not find specific information about [topic] in the available documents. The documents provided discuss [brief summary of available topics], but do not address your question directly."

## DOCUMENT NAME CLEANING:
- Remove trailing hash codes (e.g., "_89d2d7b8")
- Add ".pdf" extension
- Example: "Contract_Siemens_2024_a1b2c3d4" -> "Contract_Siemens_2024.pdf"
"""

DEFAULT_USER_PROMPT = """You are a precise document analysis assistant with self-awareness capabilities.

TEXT SNIPPETS:
{context}

USER QUESTION: {query}

=== STEP 1: REASONING & ANALYSIS ===
Before answering, perform this internal validation:

1. **Query Analysis**: What key elements is the user asking about?
   - Key entities/names mentioned: [list them]
   - Numbers/amounts referenced: [list them]
   - Core intent: [what does the user want to know?]

2. **Evidence Check**: Scan the provided snippets for matches:
   - Which snippets mention the key entities? [list snippet IDs or brief quotes]
   - Do any snippets contain the exact numbers/amounts asked about?
   - What is the context of these mentions?

3. **Self-Validation**: 
   - Does my answer DIRECTLY address the user's question?
   - Am I making assumptions not supported by the text?
   - If I found partial information, am I clearly stating what's missing?

=== STEP 2: STRUCTURED ANSWER ===

Provide your response in the following format:

**REASONING**: [2-3 sentences explaining your analysis process - what you looked for, what you found, and how confident you are]

**ANSWER**: [Your clear, direct answer addressing the user's question. If you found the information, state it clearly. If you found related but incomplete information, explain both what you found AND what's missing]

**CONTEXT**: [Describe the document context - what type of document is this? What is its purpose? This helps the user understand the source]

**SOURCE**: Document [cleaned_name.pdf], Page [X]

**CONFIDENCE**: [HIGH/MEDIUM/LOW] - [One sentence explaining why]

=== IMPORTANT RULES ===
1. ALWAYS include the REASONING section - show your thinking
2. If the user mentions a specific number (like €36.041,66), explicitly confirm whether you found it or not
3. Compare what the user asked vs what you found - be explicit about matches and gaps
4. Respond in the same language as the user's question
5. Never hallucinate - if something is not in the snippets, say so clearly

YOUR RESPONSE:"""

# Entity extraction prompt
ENTITY_EXTRACTION_PROMPT = """Extrahiere alle relevanten Entitäten aus der folgenden Frage.

Entitäten sind:
- Personen (Namen)
- Organisationen (Firmen, Behörden, Institutionen)
- Orte (Städte, Länder, Regionen)
- Konzepte (Fachbegriffe, Strategien, Programme)
- Zahlen und Daten (Jahre, Prozente, Beträge)
- Ereignisse

Frage: {query}

Antworte NUR mit einem JSON-Array der gefundenen Entitäten. Keine Erklärungen.
Beispiel: ["Angela Merkel", "Bundesregierung", "2025", "KI-Strategie", "Berlin"]

JSON-Array:"""

# Query Rewriting Prompt
REWRITE_QUERY_PROMPT = """You are a Query Rewriter. Your goal is to rewrite the user's latest question to be a standalone search query by incorporating relevant context from the conversation history.

RULES:
1. If the question is already standalone (e.g. "What is the budget?"), return it exactly as is.
2. If the question is context-dependent (e.g. "What about project Alpha?", "And the other one?"), rewrite it to be complete (e.g. "What is the budget for project Alpha?").
3. Preserve the original language of the query.
4. Output ONLY the rewritten query. No explanations.

Conversation History:
{history}

User's Latest Question: {query}

Rewritten Standalone Query:"""

# 260108–BundB Jun - BEGIN
# Direct Groq + OpenAI SDK Invocation (replaces LiteLLM)
_groq_client = None
_openai_client = None

def _get_groq_client():
    """Lazy-init singleton Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

def _get_openai_client():
    """Lazy-init singleton OpenAI client for BLAIQ proxy."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE_URL)
    return _openai_client

def _resolve_model_and_client(model: str):
    """
    Route to correct SDK client based on model prefix.
    Returns (client, model_name_without_prefix).
    """
    if "/" not in model:
        if model.startswith("llama-") or model.startswith("mixtral-") or model.startswith("gemma"):
            return _get_groq_client(), model
        else:
            return _get_openai_client(), model

    if model.startswith("groq/"):
        return _get_groq_client(), model.replace("groq/", "", 1)
    else:
        # openai/ prefix or any other → BLAIQ proxy
        return _get_openai_client(), model.replace("openai/", "", 1)

def _invoke_llm(messages, model: str, **kwargs):
    """
    Unified completion call using Groq/OpenAI SDK directly.
    """
    call_id = str(uuid.uuid4())
    start = time.time()

    client, model_name = _resolve_model_and_client(model)

    # Remove any litellm-specific or irrelevant kwargs
    kwargs.pop("api_base", None)
    kwargs.pop("api_key", None)
    kwargs.pop("stream_options", None)

    try:
        if os.getenv("DEBUG_LLM", "false").lower() == "true":
            provider = "Groq" if isinstance(client, Groq) else "OpenAI"
            print(f"  [LLM {call_id}] Provider: {provider} | Model: {model_name}")

        params = {
            "model": model_name,
            "messages": messages,
            "timeout": LLM_TIMEOUT_SECONDS,
        }

        # Reasoning models (O1, QwQ, etc.)
        is_reasoning = model_name.lower().startswith("o1") or "qwq" in model_name.lower()

        if is_reasoning:
            if model_name.lower().startswith("o1"):
                kwargs.pop("temperature", None)
            if "max_tokens" in kwargs:
                params["max_completion_tokens"] = kwargs.pop("max_tokens")
            params["timeout"] = 300
        else:
            if "temperature" not in kwargs:
                params["temperature"] = 0.0
            if "max_tokens" not in kwargs and LLM_MAX_OUTPUT_TOKENS:
                params["max_tokens"] = LLM_MAX_OUTPUT_TOKENS

        params.update(kwargs)

        response = client.chat.completions.create(**params)

        duration_ms = int((time.time() - start) * 1000)

        if os.getenv("DEBUG_LLM", "false").lower() == "true":
            usage = getattr(response, "usage", None)
            if usage:
                t_in = getattr(usage, "prompt_tokens", 0)
                t_out = getattr(usage, "completion_tokens", 0)
                t_total = getattr(usage, "total_tokens", 0)
                usage_str = f" | TOKENS: {t_total} [In: {t_in}, Out: {t_out}]"
            else:
                usage_str = ""
            print(f"  [LLM {call_id}] Model: {model_name} | Duration: {duration_ms}ms{usage_str}")

        return response.choices[0].message.content, model

    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        print(f"  Error ({model_name}) after {duration_ms}ms: {str(e)}")

        # Fallback mechanism
        fallback = os.getenv("OPENAI_FALLBACK_MODEL")
        if fallback and model != fallback and fallback != f"openai/{model}":
            print(f"  Attempting fallback to {fallback}...")
            return _invoke_llm(messages, model=fallback, **kwargs)

        raise

def _invoke_llm_stream(messages, model: str, **kwargs):
    """
    Streaming version using Groq/OpenAI SDK directly.
    Returns an iterable stream with chunk.choices[0].delta.content.
    """
    call_id = str(uuid.uuid4())
    start = time.time()

    client, model_name = _resolve_model_and_client(model)

    # Remove any litellm-specific kwargs
    kwargs.pop("api_base", None)
    kwargs.pop("api_key", None)
    kwargs.pop("stream_options", None)

    prompt_chars = sum(len(m.get("content", "")) for m in messages)

    if os.getenv("DEBUG_LLM", "false").lower() == "true":
        provider = "Groq" if isinstance(client, Groq) else "OpenAI"
        print(f"  [STREAM {call_id}] Provider: {provider} | Model: {model_name} | Prompt: {prompt_chars} chars")

    try:
        params = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "timeout": LLM_TIMEOUT_SECONDS,
        }

        # Reasoning model handling
        is_reasoning = model_name.lower().startswith("o1") or "qwq" in model_name.lower()
        if is_reasoning:
            if model_name.lower().startswith("o1"):
                kwargs.pop("temperature", None)
            if "max_tokens" in kwargs:
                params["max_completion_tokens"] = kwargs.pop("max_tokens")
            params["timeout"] = 300
        else:
            if "temperature" not in kwargs:
                params["temperature"] = 0.0
            if "max_tokens" not in kwargs and LLM_MAX_OUTPUT_TOKENS:
                params["max_tokens"] = LLM_MAX_OUTPUT_TOKENS

        params.update(kwargs)

        return client.chat.completions.create(**params)

    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        print(f"  Stream Error ({model_name}) after {duration_ms}ms: {str(e)}")

        # Fallback mechanism
        fallback = os.getenv("OPENAI_FALLBACK_MODEL")
        if fallback and model != fallback and fallback != f"openai/{model}":
            print(f"  Attempting stream fallback to {fallback}...")
            return _invoke_llm_stream(messages, model=fallback, **kwargs)

        log_llm_event("llm_error", {
            "call_id": call_id,
            "model": model_name,
            "duration_ms": duration_ms,
            "error": str(e)
        })
        raise e
# 260108–BundB Jun - END

class GraphRAGRetriever:
    """
    Hybrid GraphRAG retrieval using Neo4j knowledge graph and Qdrant vector store.

    Multi-Tenant Isolation:
    - collection_name = filter_label for both Qdrant and Neo4j
    - All Neo4j queries filter by filter_label
    - Same entity name in different tenants are separate graph nodes
    - Graph traversals never cross tenant boundaries
    """

    def __init__(
        self,
        debug: bool = False,
        # Qdrant config
        qdrant_url: Optional[str] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        entity_extraction_prompt: Optional[
            str
        ] = None,  # 251204–BundB Jun: configurable entity extraction prompt
        # Neo4j uses env vars / config.yaml
    ):
        """
        Initialize GraphRAG retriever with Qdrant and Neo4j

        Args:
            debug: Enable debug output
            qdrant_url: Qdrant URL (overrides config)
            qdrant_host: Qdrant host (overrides config)
            qdrant_port: Qdrant port (overrides config)
            qdrant_api_key: Qdrant API key
            collection_name: Collection name - REQUIRED for multi-tenant.
                            Used as filter_label for Neo4j queries.
            entity_extraction_prompt: Optional custom prompt template for entity extraction.
                                      Use {query} as placeholder.
        """
        self.debug = debug
        self.neo4j_driver = None

        # === QDRANT CONNECTION ===
        final_url = qdrant_url or QDRANT_URL
        final_api_key = qdrant_api_key or QDRANT_API_KEY
        final_host = qdrant_host or QDRANT_HOST
        final_port = qdrant_port or QDRANT_PORT

        if final_url:
            if self.debug:
                print(f"DEBUG: Connecting to Qdrant at {final_url} (verify=False, prefer_grpc=False)")
            
            # Create custom clients with SSL disabled
            import httpx
            http_client = httpx.Client(verify=False, timeout=30.0)
            async_client = httpx.AsyncClient(verify=False, timeout=30.0) # Critical for async search methods
            
            self.qdrant_client = QdrantClient(
                url=final_url, 
                api_key=final_api_key, 
                prefer_grpc=False,
                https=http_client,
                # Force async client via private usage if necessary, but QdrantClient(async=True) isn't used here.
                # Instead, the QdrantClient automatically creates an async client internally.
                # We need to ensure we pass settings that affect that.
                # Note: valid QdrantClient args don't accept 'async_http_client' directly in all versions.
                # We will Monkey-patch if necessary or rely on env vars.
            )
            
            # MONKEY PATCH: Force the internal async client to verify=False
            # This is robust against library versions
            try:
                if hasattr(self.qdrant_client, "_async_client") and self.qdrant_client._async_client:
                    self.qdrant_client._async_client = httpx.AsyncClient(headers=self.qdrant_client._async_client.headers, verify=False, timeout=30.0)
                
                # Check for newer version 'http' property which holds the client
                if hasattr(self.qdrant_client, "http") and hasattr(self.qdrant_client.http, "async_client"):
                     # This is likely where the async calls happen
                     pass # It's hard to replace deeply.
            except Exception as e:
                print(f"⚠️ Failed to patch async client: {e}")

            print(f"✅ Qdrant connected via URL: {final_url}")
        else:
            self.qdrant_client = QdrantClient(host=final_host, port=final_port, prefer_grpc=False)
            print(f"✅ Qdrant connected at {final_host}:{final_port}")
        
        if self.debug:
            print(f"DEBUG: qdrant_client type is {type(self.qdrant_client)}")
            print(f"DEBUG: available methods: {[m for m in dir(self.qdrant_client) if not m.startswith('_')]}")

        # Collection name = filter_label for multi-tenant isolation
        # Sync with QDRANT_COLLECTION environment variable if not provided
        self.collection_name = collection_name or QDRANT_COLLECTION
        self.filter_label = self.collection_name

        if not self.filter_label:
            print("⚠️ WARNING: No collection_name/filter_label provided!")
            print("   Multi-tenant isolation is DISABLED. This may cause data leaks.")
        else:
            print(f"   📁 Collection: {self.collection_name}")
            print(f"   🏷️ Filter label (tenant): {self.filter_label}")

        # === NEO4J CONNECTION ===
        try:
            if not NEO4J_PASSWORD:
                raise ValueError("NEO4J_PASSWORD not set in environment")

            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
            )
            # Test connection
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            print(f"✅ Neo4j connected at {NEO4J_URI}")
        except Exception as e:
            print(f"⚠️ Neo4j connection failed: {e}")
            print("   GraphRAG will work with reduced graph features")
            self.neo4j_driver = None

        # === EMBEDDINGS ===
        self.embeddings = BGEM3Embeddings(timeout=180)
        self.embedding_dim = BGE_M3_DIMENSION

        # Cache embedding meta for stats/debug
        self.embedding_model = getattr(self.embeddings, "model_id", None)
        self.embedding_service_url = getattr(self.embeddings, "service_url", None)

        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.embedding_service_url) if self.embedding_service_url else None
            embedding_host = parsed.hostname if parsed else None
        except Exception:
            embedding_host = None

        print(
            f"✅ Embeddings ready "
            f"(model={self.embedding_model}, dim={self.embedding_dim}, host={embedding_host})"
        )

        # === LLM for architecture ===
        self.planner_model = LITELLM_PLANNER_MODEL
        self.pre_model = LITELLM_PRE_MODEL
        self.post_model = LITELLM_POST_MODEL

        # === PROMPT TEMPLATES ===
        self.loader = PromptLoader()
        
        # Load detailed XML strategies
        self.planner_prompt = self.loader.load_planner_prompt()
        self.entity_extraction_prompt = self.loader.load_entity_prompt()
        self.response_generator_prompt = self.loader.load_response_generator()

        print(f"✅ LLM Config: Planner={self.planner_model}, Pre={self.pre_model}, Post={self.post_model}")
        print("✅ GraphRAG Retriever initialized (Graph + Vector + Keyword)")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding using BGE-M3 Embeddings. Returns None on failure."""
        try:
            embedding = self.embeddings.embed_query(text)
            # Validate embedding
            if not embedding or len(embedding) != self.embedding_dim:
                print(f"  ⚠️ Embedding dimension mismatch: got {len(embedding) if embedding else 0}, expected {self.embedding_dim}")
                return None
            # Check for zero vector (embedding service failure)
            non_zero = sum(1 for v in embedding if abs(v) > 1e-10)
            if non_zero < 10:
                print(f"  ⚠️ Embedding is near-zero ({non_zero} non-zero values). Embedding service may have failed.")
                return None
            return embedding
        except Exception as e:
            print(f"  ⚠️ Error getting embedding: {e}")
            return None

    def reformat_answer(self, base_answer: str, content_mode: str, model: Optional[str] = None) -> str:
        """
        Takes a strategic base answer and reformats it using a specialized XML template.
        Supports: EMAIL, TABLE, INVOICE.
        """
        if not content_mode or content_mode.upper() == "DEFAULT":
            return base_answer
            
        template_xml = self.loader.load_template(content_mode)
        if not template_xml:
            print(f"  ⚠️ Template for mode '{content_mode}' not found. Returning original answer.")
            return base_answer

        actual_model = model or self.post_model
        print(f"  ✨ Reformatting answer into '{content_mode.upper()}' mode using {actual_model}...")

        messages = [
            {"role": "system", "content": template_xml},
            {"role": "user", "content": f"Please reformat this analysis into the target structure:\n\n{base_answer}"},
        ]

        try:
            content, _ = _invoke_llm(
                messages,
                model=actual_model,
                max_tokens=2000 # Formatted outputs usually don't need 4k
            )
            return content
        except Exception as e:
            print(f"  ❌ Reformatting failed: {e}")
            return base_answer

    def reformat_answer_stream(self, base_answer: str, content_mode: str, model: Optional[str] = None):
        """
        Streams a reformatted base answer using specialized XML templates.
        """
        if not content_mode or content_mode.upper() == "DEFAULT":
            # Just yield the base answer if no mode
            yield base_answer
            return
            
        template_xml = self.loader.load_template(content_mode)
        if not template_xml:
            print(f"  ⚠️ Template for mode '{content_mode}' not found.")
            yield base_answer
            return

        actual_model = model or self.post_model
        messages = [
            {"role": "system", "content": template_xml},
            {"role": "user", "content": f"Please reformat this analysis into the target structure:\n\n{base_answer}"},
        ]

        try:
            return _invoke_llm_stream(
                messages,
                model=actual_model,
                max_tokens=4000,
                timeout=120,
                stream_options={"include_usage": True}
            )
        except Exception as e:
            print(f"  ❌ Streaming Reformatting failed: {e}")
            # Yield a mock LiteLLM chunk for robustness
            class MockChunk:
                def __init__(self, c):
                    self.choices = [type('obj', (object,), {'delta': type('obj', (object,), {'content': c})()})()]
            yield MockChunk(base_answer)

    def rewrite_query(self, query: str, history: str) -> str:
        """Rewrite context-dependent queries using conversation history."""
        if not history or not self._is_context_dependent(query):
            return query
            
        try:
            messages = [
                {"role": "system", "content": "You are a Query Rewriter. Output ONLY the rewritten query. No preamble."},
                {"role": "user", "content": REWRITE_QUERY_PROMPT.format(history=history[-2000:], query=query)}
            ]
            
            rewritten, _ = _invoke_llm(
                messages,
                model=self.pre_model,
                max_tokens=100,
                temperature=0.1
            )
            
            rewritten = rewritten.strip()
            if rewritten and len(rewritten) < 200:
                if self.debug:
                    print(f"  🔄 Query Rewritten: '{query}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            if self.debug:
                print(f"  ⚠️ Query rewriting failed: {e}")
                
        return query

    def _is_context_dependent(self, query: str) -> bool:
        """Detect if the query likely refers to previous context."""
        patterns = [
            r"\b(it|that|those|them|they)\b",
            r"\b(this|these)\b",
            r"\b(him|her|his|hers)\b",
            r"previous",
            r"above",
            r"before",
            r"last one",
            r"what about",
            r"how about",
            r"and the\b",
            r"why (did|is|does)\b",
            r"tell me more",
            r"explain (it|that)\b",
            r"what did i (just )?ask",
            r"what was (i|my) (previous )?question",
            r"what were we (talking|discussing) about",
        ]
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in patterns)

    def plan_retrieval(self, query: str) -> Dict:
        """
        Unified strategic planning: intent + entities + keywords in one LLM call.
        Returns plan with search_plan, entities, and keywords.
        """
        try:
            messages = [
                {"role": "system", "content": self.planner_prompt},
                {"role": "user", "content": f"User Query: {query}"}
            ]

            raw_response, used_model = _invoke_llm(
                messages,
                model=self.planner_model,
                max_tokens=800,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            plan = json.loads(raw_response)

            # Normalize search_plan from unified output
            search_plan = plan.get("search_plan", {})
            if not search_plan:
                # Fallback: try "routes" key from old format
                routes = plan.get("routes", {})
                search_plan = {
                    "use_vector": routes.get("use_vector", True),
                    "use_graph": routes.get("use_graph", False),
                    "use_keyword": routes.get("use_keyword", True),
                    "use_hive_mind": routes.get("use_hive_mind", False)
                }
            plan["search_plan"] = search_plan

            # Ensure entities and keywords are lists
            if not isinstance(plan.get("entities"), list):
                plan["entities"] = []
            if not isinstance(plan.get("keywords"), list):
                plan["keywords"] = []

            # POLICY: Always enable graph for DOCUMENT_SEARCH and ANALYTICAL_SEARCH
            mode = plan.get("mode", "DOCUMENT_SEARCH")
            if mode in ["DOCUMENT_SEARCH", "ANALYTICAL_SEARCH"]:
                search_plan["use_graph"] = True

            # SAFETY: Force graph if query has proper names or IDs
            if not search_plan.get("use_graph"):
                has_entities = bool(re.findall(r'\b[A-Z][a-z]{3,12}\b', query))
                has_ids = bool(re.findall(r'\b[A-Z]{2,5}-\d{2,8}\b', query))
                if has_entities or has_ids:
                    search_plan["use_graph"] = True

            # FALLBACK: If planner returned no entities, do fast rule-based extraction
            if not plan["entities"]:
                plan["entities"] = self._extract_entities_fast(query)

            # FALLBACK: If planner returned no keywords, do fast rule-based extraction
            if not plan["keywords"]:
                plan["keywords"] = self._extract_keywords_fast(query)

            if self.debug:
                print(f"🧠 Unified Plan: mode={mode} | entities={plan['entities']} | keywords={plan['keywords'][:5]}")

            return plan

        except Exception as e:
            print(f"⚠️ Planner failed, defaulting to FULL SEARCH with rule-based extraction. Error: {e}")
            return {
                "mode": "FALLBACK",
                "search_plan": {"use_vector": True, "use_graph": True, "use_keyword": True, "use_hive_mind": False},
                "entities": self._extract_entities_fast(query),
                "keywords": self._extract_keywords_fast(query),
            }

    def _extract_entities_fast(self, query: str) -> List[str]:
        """Fast rule-based entity extraction (< 10ms, no LLM call)."""
        entities = []

        # Currency amounts
        entities.extend(re.findall(r'€\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})?', query))
        # Percentages
        entities.extend(re.findall(r'\d+(?:,\d+)?\s*%', query))
        # Years
        entities.extend(re.findall(r'\b(19\d{2}|20\d{2})\b', query))
        # IDs (WFT-25022 pattern)
        entities.extend(re.findall(r'\b[A-Za-z]{2,5}-\d{2,8}\b', query))
        # Proper nouns (capitalized words, skip common articles)
        skip = {'Der', 'Die', 'Das', 'Was', 'Wie', 'Wer', 'Wo', 'Warum', 'Erstelle',
                'The', 'What', 'Who', 'Where', 'When', 'Why', 'How', 'Create', 'Show',
                'Can', 'Could', 'Would', 'Should', 'Is', 'Are', 'Find', 'Eine', 'Einen'}
        for word in query.split():
            if word and word[0].isupper() and word not in skip:
                clean = re.sub(r'[^\w\-]', '', word)
                if len(clean) >= 3:
                    entities.append(clean)
        # German compound nouns (8+ chars, capitalized)
        entities.extend(re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]{7,}\b', query))
        # Organization patterns
        entities.extend(re.findall(r'\b[\w\s]+(?:GmbH|AG|e\.V\.|Ltd|Inc|Corp)\b', query, re.IGNORECASE))

        return list(set(entities))

    def _extract_keywords_fast(self, query: str) -> List[str]:
        """Fast rule-based keyword extraction (< 10ms, no LLM call)."""
        stop_words = {
            "der", "die", "das", "und", "oder", "ist", "sind", "war", "ein", "eine",
            "für", "auf", "mit", "von", "zu", "bei", "nach", "über", "unter", "vor",
            "welche", "welcher", "wie", "was", "wann", "wo", "warum", "wer",
            "the", "a", "an", "and", "or", "but", "if", "for", "with", "about",
            "from", "what", "which", "who", "how", "this", "that", "these", "those",
            "erstelle", "basierend", "folgende", "strukturierte", "soll", "tabelle",
            "create", "based", "following", "structured", "should", "table",
        }

        words = re.findall(r"\b[A-Za-zÄÖÜäöüß]+\b", query)
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 3]

        # Also include numbers and number patterns
        keywords.extend(re.findall(r'\d{1,3}(?:\.\d{3})*(?:,\d{2})?', query))
        keywords.extend(re.findall(r'\b\d+\b', query))

        return list(set(keywords))
    def extract_entities_with_llm(self, query: str) -> List[str]:
        """
        Extract entities from query using a hybrid approach:
        1. Fast rule-based extraction for common patterns (< 10ms)
        2. LLM fallback for complex queries
        
        Returns:
            List of extracted entity strings
        """
        entities = []
        
        # === FAST RULE-BASED EXTRACTION (< 10ms) ===
        
        # 1. Extract European currency amounts (€36.041,66)
        euro_amounts = re.findall(r'€\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})?', query)
        entities.extend(euro_amounts)
        
        # 2. Extract percentages
        percentages = re.findall(r'\d+(?:,\d+)?\s*%', query)
        entities.extend(percentages)
        
        # 3. Extract years (1990-2099)
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        entities.extend(years)
        
        # 4. Extract proper nouns (capitalized words in the middle of sentence)
        # Skip first word and common title words
        words = query.split()
        skip_words = {'Der', 'Die', 'Das', 'Was', 'Wie', 'Wer', 'Wo', 'Warum', 
                      'The', 'What', 'Who', 'Where', 'When', 'Why', 'How', 'Do', 'Does', 'Did',
                      'Can', 'Could', 'Would', 'Should', 'Is', 'Are', 'Was', 'Were', 'Have', 'Has'}
        for i, word in enumerate(words):
            # 4. Extract proper nouns (capitalized words)
            if word and word[0].isupper() and word not in skip_words:
                clean_word = re.sub(r'[^\w\-]', '', word)  # Keep hyphen for IDs like WFT-25022
                if len(clean_word) >= 3:
                    entities.append(clean_word)
            
            # 4b. Explicit ID Match (WFT-25022)
            id_match = re.search(r'\b[A-Za-z]{2,5}-\d{2,8}\b', word)
            if id_match:
                entities.append(id_match.group(0))

        # 4c. Extract capitalized names anywhere
        names = re.findall(r'\b[A-Z][a-z]{2,12}\b', query)
        entities.extend(names)
        
        # 5. Extract German compound nouns (words starting with capital, > 8 chars)
        german_nouns = re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]{7,}\b', query)
        entities.extend(german_nouns)
        
        # 6. Extract organization patterns (GmbH, AG, e.V., etc.)
        org_patterns = re.findall(r'\b[\w\s]+(?:GmbH|AG|e\.V\.|Ltd|Inc|Corp)\b', query, re.IGNORECASE)
        entities.extend(org_patterns)
        
        # Deduplicate
        entities = list(set(entities))
        
        # === LLM FALLBACK for complex queries ===
        # Be more eager for LLM extraction if we have fewer than 2 entities
        use_llm = len(entities) < 2 and len(query) > 15
        
        if use_llm:
            if self.debug:
                print(f"  🤖 Using LLM for complex entity extraction...")
            try:
                messages = [
                    {"role": "system", "content": self.entity_extraction_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ]
                
                content, used_model = _invoke_llm(
                    messages,
                    model=self.pre_model,
                    max_tokens=500,
                )

                if self.debug:
                    print(f"  🧠 Entity extraction model: {used_model}")
                
                response_text = content.strip()

                # Clean up response
                if response_text.startswith("```"):
                    response_text = re.sub(r"```json?\s*", "", response_text)
                    response_text = re.sub(r"```\s*$", "", response_text)

                llm_entities = json.loads(response_text)
                if isinstance(llm_entities, list):
                    entities.extend(llm_entities)
                    entities = list(set(entities))  # Deduplicate again

            except Exception as e:
                if self.debug:
                    print(f"  ⚠️ LLM entity extraction failed: {e}")
        else:
            if self.debug:
                print(f"  ⚡ Fast rule-based extraction used (skipped LLM)")
        
        # EMERGENCY FALLBACK: If we still have no entities, extract the most "rare" capitalized word
        # This ensures Graph search always has something to query
        if not entities and len(query) > 5:
            # Find any capitalized word that's not a common word
            emergency_entities = re.findall(r'\b[A-Z][a-z]{2,15}\b', query)
            if emergency_entities:
                # Pick the longest one (usually the most specific)
                entities = [max(emergency_entities, key=len)]
                if self.debug:
                    print(f"  🚨 EMERGENCY FALLBACK: Using '{entities[0]}' for Graph search")
        
        if self.debug:
            print(f"  🔍 Extracted entities: {entities}")
        
        return entities

    def _extract_entities_simple(self, query: str) -> List[str]:
        """Fallback: Simple entity extraction without LLM"""
        entities = []

        # Capitalized words
        words = query.split()
        for word in words:
            clean_word = word.strip(".,?!:;")
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                entities.append(clean_word)

        # Numbers and percentages
        numbers = re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|Prozent|Euro|€|Mrd\.|Mio\.)?", query)
        entities.extend(numbers)

        # Years
        years = re.findall(r"\b20[0-5]\d\b", query)
        entities.extend(years)

        return list(set(entities))

    def _execute_neo4j_query(self, session, query: str, params: dict):
        """
        Execute Neo4j query with error handling.
        All queries should include filter_label parameter.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return session.run(query, params)
        except Exception as e:
            if self.debug:
                print(f"  ⚠️ Neo4j query error: {e}")
            return None

    def entity_based_retrieval(self, entities: List[str], k: int = 20) -> Dict[int, float]:
        """
        Graph-based retrieval using Neo4j with strict filter_label isolation.

        All queries filter by filter_label to ensure multi-tenant isolation.
        Same entity name in different tenants will NOT match.

        Args:
            entities: List of entity names to search for
            k: Number of results per entity

        Returns:
            Dict mapping qdrant_point_id to relevance score
        """
        if not self.neo4j_driver or not entities:
            return {}

        if not self.filter_label:
            if self.debug:
                print("  ⚠️ No filter_label - skipping graph retrieval for safety")
            return {}

        entity_chunk_scores = {}

        try:
            with self.neo4j_driver.session() as session:
                for entity_name in entities:
                    # Query 1: Direct entity-chunk links within tenant
                    result = self._execute_neo4j_query(
                        session,
                        """
                        MATCH (e:Entity {filter_label: $filter_label})-[:APPEARS_IN]->(c:Chunk {filter_label: $filter_label})
                        WHERE toLower(e.name) CONTAINS toLower($entity_name)
                        WITH c, count(DISTINCT e) as entity_count
                        RETURN c.chunk_id as chunk_id,
                               c.qdrant_point_id as qdrant_point_id,
                               entity_count as relevance_score
                        ORDER BY relevance_score DESC
                        LIMIT $limit
                        """,
                        {
                            "filter_label": self.filter_label,
                            "entity_name": entity_name,
                            "limit": k,
                        },
                    )

                    if result:
                        for record in result:
                            qdrant_point_id = record.get("qdrant_point_id")
                            if qdrant_point_id is not None:
                                try:
                                    qdrant_id = int(qdrant_point_id)
                                    score = float(record["relevance_score"]) * 15
                                    entity_chunk_scores[qdrant_id] = (
                                        entity_chunk_scores.get(qdrant_id, 0) + score
                                    )
                                except (ValueError, TypeError):
                                    # Fallback to computing from chunk_id
                                    chunk_id = record.get("chunk_id")
                                    if chunk_id:
                                        qdrant_id = compute_point_id(chunk_id)
                                        if qdrant_id:
                                            score = float(record["relevance_score"]) * 15
                                            entity_chunk_scores[qdrant_id] = (
                                                entity_chunk_scores.get(qdrant_id, 0) + score
                                            )

                    # Query 2: Related entities through relationships (within tenant only)
                    # First get relationship types that exist in this tenant
                    rel_types_result = self._execute_neo4j_query(
                        session,
                        """
                        MATCH (e:Entity {filter_label: $filter_label})-[r]->(:Entity {filter_label: $filter_label})
                        WHERE r.filter_label = $filter_label
                        AND NOT type(r) IN ['APPEARS_IN', 'EXTRACTED_FROM', 'HAS_CHUNK']
                        RETURN DISTINCT type(r) as rel_type
                        LIMIT 10
                        """,
                        {"filter_label": self.filter_label},
                    )

                    rel_types = []
                    if rel_types_result:
                        rel_types = [r["rel_type"] for r in rel_types_result]

                    for rel_type in rel_types:
                        # Find chunks through related entities (strict tenant isolation)
                        result = self._execute_neo4j_query(
                            session,
                            f"""
                            MATCH (e1:Entity {{filter_label: $filter_label}})-[:APPEARS_IN]->(c1:Chunk {{filter_label: $filter_label}})
                            WHERE toLower(e1.name) CONTAINS toLower($entity_name)
                            WITH e1
                            MATCH (e1)-[r:{rel_type} {{filter_label: $filter_label}}]-(e2:Entity {{filter_label: $filter_label}})
                            MATCH (e2)-[:APPEARS_IN]->(c2:Chunk {{filter_label: $filter_label}})
                            WITH c2, count(DISTINCT e2) as rel_count
                            RETURN c2.chunk_id as chunk_id,
                                   c2.qdrant_point_id as qdrant_point_id,
                                   rel_count as relevance_score
                            ORDER BY relevance_score DESC
                            LIMIT $limit
                            """,
                            {
                                "filter_label": self.filter_label,
                                "entity_name": entity_name,
                                "limit": k // 2,
                            },
                        )

                        if result:
                            for record in result:
                                qdrant_point_id = record.get("qdrant_point_id")
                                if qdrant_point_id is not None:
                                    try:
                                        qdrant_id = int(qdrant_point_id)
                                        score = float(record["relevance_score"]) * 8
                                        entity_chunk_scores[qdrant_id] = (
                                            entity_chunk_scores.get(qdrant_id, 0) + score
                                        )
                                    except (ValueError, TypeError):
                                        pass

                    # Query 3: Cross-document entities within tenant (high value for knowledge synthesis)
                    result = self._execute_neo4j_query(
                        session,
                        """
                        MATCH (e:Entity {filter_label: $filter_label})-[:EXTRACTED_FROM]->(d:Document {filter_label: $filter_label})
                        WHERE toLower(e.name) CONTAINS toLower($entity_name)
                        WITH e, count(DISTINCT d) as doc_count
                        WHERE doc_count > 1
                        MATCH (e)-[:APPEARS_IN]->(c:Chunk {filter_label: $filter_label})
                        RETURN c.chunk_id as chunk_id,
                               c.qdrant_point_id as qdrant_point_id,
                               e.name as entity_name,
                               doc_count,
                               doc_count * 10 as relevance_score
                        ORDER BY relevance_score DESC
                        LIMIT $limit
                        """,
                        {
                            "filter_label": self.filter_label,
                            "entity_name": entity_name,
                            "limit": k // 3,
                        },
                    )

                    if result:
                        for record in result:
                            qdrant_point_id = record.get("qdrant_point_id")
                            if qdrant_point_id is not None:
                                try:
                                    qdrant_id = int(qdrant_point_id)
                                    score = float(record["relevance_score"]) * 12
                                    entity_chunk_scores[qdrant_id] = (
                                        entity_chunk_scores.get(qdrant_id, 0) + score
                                    )
                                except (ValueError, TypeError):
                                    pass

        except Exception as e:
            if self.debug:
                print(f"  ⚠️ Neo4j retrieval error: {e}")

        if self.debug and entity_chunk_scores:
            print(f"  🔗 Graph found {len(entity_chunk_scores)} chunks for entities: {entities}")

        return entity_chunk_scores

    def expand_query_with_cot(self, query: str) -> Dict:
        """Chain-of-Thought query expansion"""
        query_lower = query.lower()

        expanded = {
            "original": query,
            "keywords": [],
            "numbers": [],
            "years": [],
            "percentages": [],
            "expected_patterns": [],
        }

        # Extract European-format numbers (periods as thousand separators, comma as decimal)
        # Matches: 36.041,66 or €36.041,66 or 36.041 or 1.234.567,89
        euro_numbers = re.findall(r'€?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)', query)
        # Also extract plain numbers
        plain_numbers = re.findall(r'\b(\d+)\b', query)
        
        # Normalize European numbers for matching (add variants)
        normalized_numbers = []
        for num in euro_numbers:
            normalized_numbers.append(num)  # Original: 36.041,66
            normalized_numbers.append("€ " + num)  # With symbol and space: € 36.041,66
            normalized_numbers.append("€" + num)  # With symbol: €36.041,66
            # Also store without thousand separators for partial matching
            no_thousand = num.replace(".", "")  # 36041,66
            normalized_numbers.append(no_thousand)
        
        expanded["numbers"] = list(set(euro_numbers + plain_numbers + normalized_numbers))
        expanded["years"] = [n for n in plain_numbers if n.isdigit() and 1990 <= int(n) <= 2050]
        expanded["percentages"] = re.findall(r'\d+\s*%', query)

        words = re.findall(r"\b[A-Za-zÄÖÜäöüß]+\b", query)
        
        # Comprehensive stop words: German + English
        stop_words = {
            # German
            "der", "die", "das", "und", "oder", "ist", "sind", "war", "waren",
            "wird", "werden", "wurde", "wurden", "haben", "hat", "hatte", "hatten",
            "welche", "welcher", "welches", "wie", "was", "wann", "wo", "warum",
            "wer", "wem", "wen", "wessen", "bis", "für", "auf", "mit", "von", "zu",
            "bei", "nach", "über", "unter", "vor", "hinter", "neben", "zwischen",
            "durch", "gegen", "ohne", "um", "aus", "an", "in", "im", "am", "zum",
            "zur", "beim", "vom", "ins", "ans", "aufs", "durchs", "fürs", "ums",
            "ob", "wenn", "als", "weil", "dass", "damit", "obwohl", "obgleich",
            "während", "nachdem", "bevor", "sobald", "solange", "sofern", "falls",
            "auch", "noch", "schon", "erst", "nur", "bloß", "aber", "jedoch",
            "sondern", "denn", "doch", "also", "daher", "deshalb", "deswegen",
            "darum", "folglich", "somit", "trotzdem", "dennoch", "allerdings",
            "ein", "eine", "einer", "eines", "einem", "einen", "kein", "keine",
            "keiner", "keines", "keinem", "keinen", "mein", "meine", "dein", "deine",
            "sein", "seine", "ihr", "ihre", "unser", "unsere", "euer", "eure",
            "dieser", "diese", "dieses", "jener", "jene", "jenes", "solcher",
            "solche", "solches", "welch", "irgendein", "irgendeine", "alle", "jede",
            "jeder", "jedes", "manche", "mancher", "manches", "einige", "etliche",
            "mehrere", "viele", "wenige", "beide", "sämtliche", "jedwede",
            "ich", "du", "er", "sie", "es", "wir", "ihr", "Sie", "mich", "dich",
            "ihn", "uns", "euch", "mir", "dir", "ihm", "ihnen",
            "nicht", "nichts", "nie", "niemals", "nirgends", "nirgendwo",
            "sehr", "ganz", "recht", "ziemlich", "etwas", "wenig", "viel", "mehr",
            "genug", "fast", "beinahe", "kaum", "höchst", "äußerst", "besonders",
            "ja", "nein", "vielleicht", "wohl", "etwa", "ungefähr", "circa",
            # English
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
            "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "once", "here", "there", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "can",
            "will", "just", "don", "should", "now", "would", "could", "might",
            "must", "shall", "may", "need", "dare", "ought", "used", "had", "has",
            "have", "having", "do", "does", "did", "doing", "be", "been", "being",
            "am", "is", "are", "was", "were", "i", "me", "my", "myself", "we",
            "our", "ours", "ourselves", "you", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
            "herself", "it", "its", "itself", "they", "them", "their", "theirs",
            "themselves", "what", "which", "who", "whom", "this", "that", "these",
            "those", "any", "both", "each", "few", "many", "much", "neither",
            "either", "several", "enough", "another", "every", "nobody", "somebody",
            "anybody", "everybody", "nothing", "something", "anything", "everything",
            "see", "seeing", "seen", "saw", "remember", "document", "documents",
            "find", "finding", "found", "show", "showing", "shown", "tell", "told",
        }
        
        # Filter: not a stop word, at least 3 chars, and contains meaningful content
        expanded["keywords"] = [
            w for w in words 
            if w.lower() not in stop_words 
            and len(w) > 3  # Increased from 2 to 3
            and not w.lower() in ["money", "euro", "euros", "geld", "summe", "sum", "amount"]  # These are covered by number extraction
        ]

        if "anteil" in query_lower or "prozent" in query_lower:
            expanded["expected_patterns"] = [
                "prozent",
                "%",
                "drittel",
                "hälfte",
                "viertel",
                "anteil",
            ]
        if "betrag" in query_lower or "€" in query or "euro" in query_lower:
            expanded["expected_patterns"].extend(["mrd", "milliarden", "€", "euro"])
        if "ziel" in query_lower:
            expanded["expected_patterns"].extend(["bis", "2030", "2035", "2040", "2050", "%"])

        # === OPTIONAL: True LLM-based Chain-of-Thought for Complex Queries ===
        # Trigger conditions: Multi-part questions, analytical queries, or when fast extraction found little
        is_complex = (
            len(query.split()) > 8 or  # Long query
            any(word in query_lower for word in ['wie', 'warum', 'welche', 'relationship', 'connect', 'compare', 'analyze']) or
            (len(expanded["keywords"]) < 2 and len(query) > 20)  # Few keywords but long query
        )
        
        if is_complex:
            try:
                cot_prompt = f"""You are a Query Expansion AI. Given a user query, think step-by-step to generate related search terms that would help find relevant documents.

User Query: "{query}"

Think through:
1. What is the core intent?
2. What related terms, synonyms, or concepts should we search for?
3. What specific entities (names, IDs, organizations) are mentioned?

Output ONLY a JSON object with this structure:
{{
  "reasoning": "brief explanation of the query intent",
  "additional_keywords": ["term1", "term2", "term3"],
  "related_concepts": ["concept1", "concept2"],
  "entities": ["entity1", "entity2"]
}}"""

                messages = [
                    {"role": "system", "content": "You are a search query expansion expert. Output only valid JSON."},
                    {"role": "user", "content": cot_prompt}
                ]
                
                cot_response, _ = _invoke_llm(
                    messages,
                    model=self.pre_model,
                    max_tokens=300,
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                cot_data = json.loads(cot_response)
                
                # Merge LLM expansions with rule-based results
                if "additional_keywords" in cot_data:
                    expanded["keywords"].extend(cot_data["additional_keywords"])
                if "related_concepts" in cot_data:
                    expanded["keywords"].extend(cot_data["related_concepts"])
                if "entities" in cot_data:
                    expanded["keywords"].extend(cot_data["entities"])
                
                # Deduplicate
                expanded["keywords"] = list(set(expanded["keywords"]))
                
                if self.debug:
                    print(f"  🧠 LLM CoT Expansion: +{len(cot_data.get('additional_keywords', []))} keywords")
                    
            except Exception as e:
                if self.debug:
                    print(f"  ⚠️ LLM CoT expansion failed: {e}")

        return expanded

    def vector_search(self, query: str, k: int) -> Dict[int, float]:
        """Vector search using embeddings (Qdrant collection = tenant isolation)"""
        query_embedding = self.get_embedding(query)

        if query_embedding is None:
            print(f"  ⚠️ Vector search skipped: embedding failed for query '{query[:50]}...'")
            return {}

        import requests as req_lib

        q_url = os.getenv("QDRANT_URL", "https://qdrant.api.blaiq.ai").rstrip("/")
        api_key = os.getenv("QDRANT_API_KEY")
        headers = {"api-key": api_key} if api_key else {}

        # Strategy 1: REST API direct call
        try:
            search_url = f"{q_url}/collections/{self.collection_name}/points/search"
            payload = {
                "vector": query_embedding,
                "limit": k,
                "with_payload": False,
                "score_threshold": 0.1
            }

            if self.debug:
                print(f"DEBUG: Vector search to {search_url} (k={k})")

            resp = req_lib.post(search_url, json=payload, headers=headers, verify=False, timeout=15.0)

            if resp.status_code == 200:
                results = resp.json().get("result", [])
                if self.debug:
                    print(f"DEBUG: Vector returned {len(results)} results")
                return {
                    point.get("id"): float(point.get("score") or 0.0)
                    for point in results
                    if point.get("id") is not None
                }
            else:
                print(f"  ⚠️ Vector REST failed ({resp.status_code}): {resp.text[:200]}")

        except Exception as e:
            print(f"  ⚠️ Vector REST call failed: {e}")

        # Strategy 2: QdrantClient fallback
        try:
            print(f"  🔄 Falling back to QdrantClient for vector search...")
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=0.1,
            )
            return {
                point.id: point.score
                for point in results
            }
        except Exception as e:
            print(f"  ⚠️ QdrantClient fallback also failed: {e}")
            return {}

    def keyword_search(self, query: str, expanded_query: Dict, k: int) -> Dict[int, float]:
        """
        Optimized keyword search using Qdrant's native text matching.
        
        PERFORMANCE: 10-50x faster than scroll-based approach.
        Uses Qdrant's indexed text search instead of Python string matching.
        Enhanced to handle European number formats.
        """
        search_terms = expanded_query["keywords"] + expanded_query["numbers"]
        
        if not search_terms:
            return {}
        
        keyword_scores = {}
        
        if self.debug:
            print(f"  🔑 Keyword search terms: {search_terms[:15]}")
        
        # Strategy: Search for each term separately, then aggregate scores
        # This is much faster than scrolling all documents
        for term in search_terms[:15]:  # Increased limit to include number variants
            term_str = str(term).strip()
            if len(term_str) < 2:  # Skip very short terms
                continue
            
            # Clean the term for Qdrant matching
            # Remove € for cleaner matching, we'll search for the number part
            clean_term = term_str.replace("€", "").strip().lower()
            
            # Skip if it's just symbols
            if not any(c.isalnum() for c in clean_term):
                continue
            
            try:
                # Use Qdrant's MatchText filter for indexed search
                # Use Manual REST for scroll to avoid connection issues
                import requests
                
                q_url = os.getenv("QDRANT_URL", "https://qdrant.api.blaiq.ai").rstrip("/")
                scroll_url = f"{q_url}/collections/{self.collection_name}/points/scroll"
                
                headers = {}
                api_key = os.getenv("QDRANT_API_KEY") or (self.qdrant_client._api_key if hasattr(self.qdrant_client, "_api_key") else None)
                if api_key:
                    headers["api-key"] = api_key

                # For numeric terms with European formatting, extract just digits
                # 36.041,66 -> search for "36041" or "36.041"
                is_number = bool(re.match(r'^[\d.,]+$', clean_term))
                
                if is_number:
                    # For numbers, try multiple search strategies
                    # Strategy 1: Search for digits without separators
                    digits_only = re.sub(r'[^\d]', '', clean_term)
                    # Strategy 2: Search for first significant part (before comma)
                    main_part = clean_term.split(',')[0]
                    
                    search_variants = [clean_term, digits_only, main_part]
                    search_variants = list(set([v for v in search_variants if len(v) >= 3]))
                else:
                    search_variants = [clean_term]

                for variant in search_variants:
                    payload = {
                        "filter": {
                            "must": [
                                {
                                    "key": "text",
                                    "match": {"text": variant}
                                }
                            ]
                        },
                        "limit": min(k * 2, 100),
                        "with_payload": True
                    }

                    resp = requests.post(scroll_url, json=payload, headers=headers, verify=False, timeout=5.0)
                    
                    if resp.status_code == 200:
                        data = resp.json().get("result", {}).get("points", [])
                        # Lightweight object to mimic Qdrant PointStruct for downstream code
                        class SimpleRecord:
                            def __init__(self, d):
                                self.id = d.get("id")
                                self.payload = d.get("payload", {})
                        
                        records = [SimpleRecord(r) for r in data]
                        
                        if self.debug and records:
                            print(f"    📄 Found {len(records)} matches for '{variant}'")
                    else:
                        if self.debug: 
                            print(f"  ⚠️ Keyword scroll failed for '{variant}': {resp.status_code}")
                        records = []
                    
                    # Score based on term frequency in matched documents
                    for record in records:
                        text = record.payload.get("text", "").lower()
                        # For numbers, boost score if the exact amount appears
                        if is_number and clean_term in text:
                            tf = text.count(clean_term)
                            score = math.log(1 + tf) * 20  # Higher boost for exact number match
                        else:
                            tf = text.count(variant)
                            score = math.log(1 + tf) * 10
                        
                        # Aggregate scores for documents matching multiple terms
                        point_id = record.id
                        keyword_scores[point_id] = keyword_scores.get(point_id, 0) + score
                        
            except Exception as e:
                if self.debug:
                    print(f"  ⚠️ Keyword search error for term '{term_str}': {e}")
                # Fallback: continue with other terms
                continue
        
        # Sort by score and return top k
        sorted_results = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_results[:k])

    def get_adjacent_chunks(self, point_ids: List[int], window_size: int = 1) -> Dict[int, float]:
        """Get adjacent chunks for context expansion in parallel"""
        adjacent_results = {}

        if not point_ids:
            return adjacent_results

        import requests
        from concurrent.futures import ThreadPoolExecutor

        q_url = os.getenv("QDRANT_URL", "https://qdrant.api.blaiq.ai").rstrip("/")
        api_key = os.getenv("QDRANT_API_KEY") or (self.qdrant_client._api_key if hasattr(self.qdrant_client, "_api_key") else None)
        headers = {"api-key": api_key} if api_key else {}

        # 1. Fetch original blocks to get metadata
        try:
            resp = requests.post(
                f"{q_url}/collections/{self.collection_name}/points", 
                json={"ids": point_ids[:40], "with_payload": True}, 
                headers=headers, timeout=5.0
            )
            if resp.status_code != 200:
                return adjacent_results
            
            points_data = resp.json().get("result", [])
        except Exception:
            return adjacent_results

        # 2. Define worker for fetching neighbors
        def fetch_neighbor(doc_id, target_idx, offset):
            try:
                scroll_payload = {
                    "filter": {
                        "must": [
                            {"key": "doc_id", "match": {"text": doc_id}},
                            {"key": "chunk_index", "match": {"value": target_idx}}
                        ]
                    },
                    "limit": 1,
                    "with_payload": True
                }
                res = requests.post(f"{q_url}/collections/{self.collection_name}/points/scroll", 
                                   json=scroll_payload, headers=headers, timeout=2.0)
                if res.status_code == 200:
                    pts = res.json().get("result", {}).get("points", [])
                    if pts:
                        return pts[0].get("id"), 10.0 / (1 + abs(offset))
            except Exception:
                pass
            return None

        # 3. Queue neighbor tasks
        tasks = []
        for p in points_data:
            payload = p.get("payload", {})
            doc_id = payload.get("doc_id")
            chunk_index = payload.get("chunk_index")
            if doc_id is None or chunk_index is None:
                continue
                
            for offset in range(-window_size, window_size + 1):
                if offset == 0: continue
                target_idx = chunk_index + offset
                if target_idx < 0: continue
                tasks.append((doc_id, target_idx, offset))

        # 4. Execute in Parallel
        if tasks:
            with ThreadPoolExecutor(max_workers=min(len(tasks), 20)) as executor:
                futures = [executor.submit(fetch_neighbor, *t) for t in tasks]
                for f in futures:
                    res = f.result()
                    if res:
                        neighbor_id, score = res
                        adjacent_results[neighbor_id] = score

        return adjacent_results

    def weighted_rrf_fusion(
        self,
        rankings: Dict[str, Dict[int, float]],
        weights: Dict[str, float] = None,
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """Weighted Reciprocal Rank Fusion"""
        if weights is None:
            num_methods = len(rankings)
            weights = {method: 1.0 / num_methods for method in rankings}

        fused_scores = defaultdict(float)

        for method_name, ranking in rankings.items():
            if not ranking:
                continue

            method_weight = weights.get(method_name, 0.1)
            sorted_items = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

            for rank, (doc_id, original_score) in enumerate(sorted_items, 1):
                fused_scores[doc_id] += method_weight * (1 / (k + rank))

        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def _retrieve_chunks(self, ranked_results: List[Tuple[Any, float]]) -> List[Document]:
        """Retrieve actual chunk documents from ranked results"""
        chunks = []

        if not ranked_results:
            return chunks

        point_ids = [pid for pid, _ in ranked_results]

        try:
            # Manual REST call to bypass QdrantClient connection issues
            import requests
            
            q_url = os.getenv("QDRANT_URL", "https://qdrant.api.blaiq.ai").rstrip("/")
            retrieve_url = f"{q_url}/collections/{self.collection_name}/points"
            
            headers = {}
            api_key = os.getenv("QDRANT_API_KEY") or (self.qdrant_client._api_key if hasattr(self.qdrant_client, "_api_key") else None)
            if api_key:
                headers["api-key"] = api_key
            
            # Use requests with verify=False
            if self.debug:
                 print(f"DEBUG: Retrieving {len(point_ids)} chunks from {retrieve_url}")

            resp = requests.post(
                retrieve_url,
                json={"ids": point_ids, "with_payload": True},
                headers=headers,
                verify=False,
                timeout=5.0
            )

            if resp.status_code == 200:
                points_data = resp.json().get("result", [])
                
                # Create a map for order-preserving lookups
                # Normalize keys to string to handle UUID/Int mix
                points_map = {str(p.get("id")): p for p in points_data}

                for i, (pid, score) in enumerate(ranked_results):
                    point = points_map.get(str(pid))
                    
                    if not point:
                        continue

                    payload = point.get("payload", {})
                    
                    chunk = Document(
                        page_content=payload.get("text", ""),
                        metadata={
                            "qdrant_id": point.get("id"),
                            "doc_id": payload.get("doc_id", ""),
                            "chunk_id": payload.get("chunk_id", ""),
                            "chunk_index": payload.get("chunk_index", 0),
                            "fusion_score": score,
                            "retrieval_rank": i + 1,
                        },
                    )
                    chunks.append(chunk)
            else:
                 print(f"  ⚠️ Failed to retrieve chunks: Status {resp.status_code} - {resp.text}")

        except Exception as e:
            print(f"  ⚠️ Error retrieving chunks: {e}")

        return chunks

    def graphrag_retrieval(
        self, query: str, k: int = 20, debug: bool = None
    ) -> Tuple[List[Document], Dict]:
        """
        Strategic Context-Driven Retrieval.
        Instead of a fixed pipeline, we plan the search first.
        """
        if debug is None:
            debug = self.debug

        # --- STEP 1: STRATEGIC PLANNING ---
        print(f"🎯 Planning retrieval strategy for: '{query[:50]}...'")
        plan = self.plan_retrieval(query)
        search_mode = plan.get("mode", "LOCAL_SEARCH")
        
        # --- RESPONSE BRANCH A: SMALL TALK ---
        if search_mode == "SMALL_TALK":
            print("  ☕ Direct conversational reply (No DB search needed)")
            system_doc = Document(
                page_content=plan.get("direct_reply") or "Hallo! Wie kann ich Ihnen bei Ihren Dokumenten helfen?",
                metadata={"mode": "small_talk", "is_direct": True}
            )
            return [system_doc], {"mode": "small_talk", "plan": plan}

        # --- RESPONSE BRANCH B: GLOBAL HIVE ---
        if search_mode == "GLOBAL_SEARCH":
            print("  🕸️ Global Hive Mode: Strategic summary across corpus")
            summary = self.generate_global_hive_summary(query)
            if summary:
                global_doc = Document(
                    page_content=summary,
                    metadata={"mode": "global_hive", "is_direct": True}
                )
                return [global_doc], {"mode": "global_hive", "plan": plan}
            # If global summary fails, fall back to local search
            print("  ⚠️ Global summary failed, falling back to local search")

        # --- RESPONSE BRANCH C: LOCAL SEARCH (EVENT DRIVEN) ---
        print(f"🔍 Executing Local Search (tenant: {self.filter_label})...")

        # Use unified planner's entities and keywords (no separate LLM calls)
        entities = plan.get("entities", [])
        keywords = plan.get("keywords", [])
        search_plan = plan.get("search_plan", {"use_vector": True, "use_graph": True, "use_keyword": True})

        # Build expanded_query from planner keywords
        expanded_query = {
            "original": query,
            "keywords": keywords,
            "numbers": [k for k in keywords if any(c.isdigit() for c in k)],
            "years": [], "percentages": [], "expected_patterns": [],
        }

        broad_k = k * 10
        rankings = {}

        # 1. GRAPH (Conditional)
        if search_plan.get("use_graph") and self.neo4j_driver and entities and self.filter_label:
            graph_results = self.entity_based_retrieval(entities, k=broad_k)
            if graph_results:
                rankings["graph"] = graph_results
                print(f"  🔗 Graph used: {len(graph_results)} entity-linked chunks")

        # 2. VECTOR (Conditional)
        if search_plan.get("use_vector"):
            vector_results = self.vector_search(query, k=broad_k)
            if vector_results:
                rankings["vector"] = vector_results
            print(f"  📊 Vector used: {len(vector_results)} results")

        # 3. KEYWORD (Conditional)
        if search_plan.get("use_keyword"):
            keyword_results = self.keyword_search(query, expanded_query, k=broad_k)
            if keyword_results:
                rankings["keyword"] = keyword_results
            print(f"  🔑 Keyword used: {len(keyword_results)} results")

        # 4. ADJACENT CHUNKS (Context preservation)
        all_top = []
        for r_dict in rankings.values():
            all_top.extend(list(r_dict.keys())[:10])
        all_top = list(set(all_top[:30]))
        
        if all_top:
            adjacent_results = self.get_adjacent_chunks(all_top[:20], window_size=1)
            if adjacent_results:
                rankings["adjacent"] = adjacent_results

        # Stage 4: Fusion
        if not rankings:
            return [], {"mode": "error", "error": "No relevant data found"}

        # Dynamic weights based on planner context
        if "graph" in rankings:
            weights = {"graph": 0.40, "vector": 0.45, "keyword": 0.10, "adjacent": 0.05}
        else:
            weights = {"vector": 0.70, "keyword": 0.25, "adjacent": 0.05}

        fused_results = self.weighted_rrf_fusion(rankings, weights=weights, k=25)
        # Ensure we respect the user's requested k, but cap it at the fused limit
        final_k = min(k, 25)
        chunks = self._retrieve_chunks(fused_results[:final_k])

        stats = {
            "mode": "local_search",
            "planning": plan,
            "graph_used": "graph" in rankings,
            "vector_used": "vector" in rankings,
            "keyword_used": "keyword" in rankings,
            "chunks_retrieved": len(chunks),
            "filter_label": self.filter_label
        }

        return chunks, stats

    def get_graph_for_entities(
        self, entities: List[str], depth: int = 1, limit: int = 50
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Get graph data (entities and relationships) from Neo4j for given entity names.

        Strict filter_label isolation - only returns entities/relationships from current tenant.

        Args:
            entities: List of entity names to search for
            depth: How many hops to traverse (1-3)
            limit: Maximum number of entities to return

        Returns:
            Tuple of (entities_list, relationships_list)
        """
        if not self.neo4j_driver or not entities:
            return [], []

        if not self.filter_label:
            if self.debug:
                print("  ⚠️ No filter_label - skipping graph visualization for safety")
            return [], []

        found_entities = {}
        found_relationships = []

        try:
            with self.neo4j_driver.session() as session:
                for entity_name in entities:
                    # Build query based on depth - ALL queries enforce filter_label
                    if depth == 1:
                        query = """
                        MATCH (e:Entity {filter_label: $filter_label})
                        WHERE toLower(e.name) CONTAINS toLower($entity_name)
                        WITH e
                        LIMIT 10
                        OPTIONAL MATCH (e)-[r]-(e2:Entity {filter_label: $filter_label})
                        WHERE NOT type(r) IN ['APPEARS_IN', 'EXTRACTED_FROM', 'HAS_CHUNK']
                          AND (r.filter_label IS NULL OR r.filter_label = $filter_label)
                        RETURN e, r, e2
                        LIMIT $limit
                        """
                    elif depth == 2:
                        query = """
                        MATCH (e:Entity {filter_label: $filter_label})
                        WHERE toLower(e.name) CONTAINS toLower($entity_name)
                        WITH e
                        LIMIT 5
                        OPTIONAL MATCH path = (e)-[*1..2]-(e2:Entity {filter_label: $filter_label})
                        WHERE ALL(node IN nodes(path) WHERE node:Entity AND node.filter_label = $filter_label)
                          AND ALL(rel IN relationships(path) WHERE 
                              NOT type(rel) IN ['APPEARS_IN', 'EXTRACTED_FROM', 'HAS_CHUNK']
                              AND (rel.filter_label IS NULL OR rel.filter_label = $filter_label))
                        UNWIND relationships(path) as rel
                        WITH e, rel, startNode(rel) as src, endNode(rel) as tgt
                        RETURN DISTINCT e, rel as r, 
                               CASE WHEN src = e THEN tgt ELSE src END as e2
                        LIMIT $limit
                        """
                    else:  # depth == 3
                        query = """
                        MATCH (e:Entity {filter_label: $filter_label})
                        WHERE toLower(e.name) CONTAINS toLower($entity_name)
                        WITH e
                        LIMIT 3
                        OPTIONAL MATCH path = (e)-[*1..3]-(e2:Entity {filter_label: $filter_label})
                        WHERE ALL(node IN nodes(path) WHERE node:Entity AND node.filter_label = $filter_label)
                          AND ALL(rel IN relationships(path) WHERE 
                              NOT type(rel) IN ['APPEARS_IN', 'EXTRACTED_FROM', 'HAS_CHUNK']
                              AND (rel.filter_label IS NULL OR rel.filter_label = $filter_label))
                        UNWIND relationships(path) as rel
                        WITH e, rel, startNode(rel) as src, endNode(rel) as tgt
                        RETURN DISTINCT e, rel as r,
                               CASE WHEN src = e THEN tgt ELSE src END as e2
                        LIMIT $limit
                        """

                    result = self._execute_neo4j_query(
                        session,
                        query,
                        {
                            "filter_label": self.filter_label,
                            "entity_name": entity_name,
                            "limit": limit,
                        },
                    )

                    if result:
                        for record in result:
                            # Process source entity
                            e = record.get("e")
                            if e:
                                eid = e.get("global_id", str(e.id))
                                if eid not in found_entities:
                                    found_entities[eid] = {
                                        "id": eid,
                                        "name": e.get("name", "Unknown"),
                                        "type": e.get("type", "UNKNOWN"),
                                    }

                            # Process target entity
                            e2 = record.get("e2")
                            if e2:
                                e2id = e2.get("global_id", str(e2.id))
                                if e2id not in found_entities:
                                    found_entities[e2id] = {
                                        "id": e2id,
                                        "name": e2.get("name", "Unknown"),
                                        "type": e2.get("type", "UNKNOWN"),
                                    }

                            # Process relationship
                            r = record.get("r")
                            if r and e and e2:
                                rel_data = {
                                    "source_id": e.get("global_id", str(e.id)),
                                    "target_id": e2.get("global_id", str(e2.id)),
                                    "type": r.type,
                                }
                                if rel_data not in found_relationships:
                                    found_relationships.append(rel_data)

        except Exception as e:
            if self.debug:
                print(f"  ⚠️ Graph query error: {e}")

        return list(found_entities.values()), found_relationships

    def discover_communities(self, limit: int = 5) -> List[Dict]:
        """
        Global Hive: Discover structural communities in the graph.
        Identifies clusters of entities that are highly connected.
        """
        if not self.neo4j_driver or not self.filter_label:
            return []

        logger.info(f"🕸️ Discovering communities for tenant: {self.filter_label}")
        
        try:
            with self.neo4j_driver.session() as session:
                # Find entities that have the most relationships within this tenant
                query = """
                MATCH (e:Entity {filter_label: $filter_label})
                WITH e, count{(e)-[:RELATIONSHIP]-(:Entity)} as rel_count
                WHERE rel_count > 1
                MATCH (e)-[:RELATIONSHIP]-(neighbor:Entity {filter_label: $filter_label})
                WITH e, collect(DISTINCT neighbor.name) as connections, rel_count
                ORDER BY rel_count DESC
                LIMIT $limit
                RETURN e.name as community_root, connections, rel_count as density
                """
                result = session.run(query, {"filter_label": self.filter_label, "limit": limit})
                communities = [record.data() for record in result]
                return communities
        except Exception as e:
            logger.error(f"❌ Community discovery failed: {e}")
            return []

    def generate_global_hive_summary(self, query: str) -> Optional[str]:
        """
        Enterprise Hive Intelligence: Answers high-level 'Global' questions 
        by analyzing community structures instead of just local chunks.
        """
        communities = self.discover_communities(limit=10)
        if not communities:
            return None
            
        # Group community info for the LLM
        context = "ENTEPRISE HIVE INTELLIGENCE - GRAPH COMMUNITIES:\n"
        for i, comm in enumerate(communities):
            context += f"Community {i+1} (Root: {comm['community_root']}):\n"
            context += f" - Key Connections: {', '.join(comm['connections'][:10])}\n"
            context += f" - Connection Density: {comm['density']}\n\n"
            
        prompt = f"""
        You are the Enterprise Hive Intelligence. 
        Based on the structural relationships in the Knowledge Graph provided below, 
        answer the user's high-level strategic question.
        
        GRAPH CONTEXT:
        {context}
        
        USER QUESTION:
        {query}
        
        Focus on identifying patterns, shared risks, and commonalities across the entire dataset.
        If the data doesn't provide enough information, state that clearly.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a strategic intelligence layer for a Graph-based Knowledge Management system."},
                {"role": "user", "content": prompt}
            ]
            
            # Global summarization also benefits from the reasoning model
            content, used_model = _invoke_llm(
                messages,
                model=self.post_model,
                max_tokens=2000
            )
            return content
        except Exception as e:
            logger.error(f"❌ Global Hive Summary failed: {e}")
            return None

    def neo4j_to_mermaid(self, entities: List[Dict], relationships: List[Dict]) -> str:
        """
        Convert Neo4j graph data to Mermaid flowchart syntax.

        Args:
            entities: List of entity dicts with id, name, type
            relationships: List of relationship dicts with source_id, target_id, type

        Returns:
            Mermaid diagram code string
        """
        if not entities:
            return "graph LR\n    empty[No graph data]"

        lines = ["graph LR"]

        # Create node ID mapping (sanitize for Mermaid)
        node_ids = {}
        for i, entity in enumerate(entities):
            node_ids[entity["id"]] = f"E{i}"

        # Add nodes with shapes based on entity type
        type_shapes = {
            "PERSON": ('["', '"]'),  # Rectangle
            "ROLLE": ('["', '"]'),  # Rectangle
            "ORGANISATION": ('(["', '"])'),  # Stadium shape
            "ORT": ('{{"', '"}}'),  # Rhombus
            "IMMOBILIE": ('{{"', '"}}'),  # Rhombus
            "DOKUMENT": ('(["', '"])'),  # Stadium
            "KONZEPT": ('(("', '"))'),  # Circle
            "EVENT": ('(["', '"])'),  # Stadium
            "ZEIT": ('["', '"]'),  # Rectangle
            "FINANZIELL": ('["', '"]'),  # Rectangle
            "BESTAND": ('["', '"]'),  # Rectangle
            "SERVICE": ('(["', '"])'),  # Stadium
        }

        for entity in entities:
            node_id = node_ids.get(entity["id"])
            if not node_id:
                continue

            etype = entity.get("type", "UNKNOWN")
            name = entity.get("name", "?")

            # Escape special characters for Mermaid
            name = name.replace('"', "'").replace("\n", " ")[:40]
            if len(entity.get("name", "")) > 40:
                name += "..."

            # Get shape based on type
            prefix, suffix = type_shapes.get(etype, ('["', '"]'))
            lines.append(f"    {node_id}{prefix}{name}{suffix}")

        # Add relationships
        for rel in relationships:
            source_id = node_ids.get(rel["source_id"])
            target_id = node_ids.get(rel["target_id"])
            rel_type = rel.get("type", "RELATED")

            # Shorten relationship type for display
            rel_display = rel_type.replace("_", " ")[:15]

            if source_id and target_id:
                lines.append(f"    {source_id} -->|{rel_display}| {target_id}")

        # Add styling
        lines.append("")
        lines.append("    %% Styling")
        lines.append("    classDef person fill:#e1f5fe,stroke:#01579b")
        lines.append("    classDef org fill:#fff3e0,stroke:#e65100")
        lines.append("    classDef place fill:#e8f5e9,stroke:#2e7d32")
        lines.append("    classDef concept fill:#f3e5f5,stroke:#7b1fa2")
        lines.append("    classDef finance fill:#fff8e1,stroke:#f57f17")

        return "\n".join(lines)

    def get_graph_visualization(self, entities: List[str], depth: int = 1) -> Optional[Dict]:
        """
        Get complete graph visualization data for given entities.

        Args:
            entities: List of entity names to visualize
            depth: Graph traversal depth (1-3)

        Returns:
            Dict with mermaid_code, nodes, edges, entities, relationships
            or None if no graph data found
        """
        if not self.neo4j_driver:
            return None

        if not self.filter_label:
            return None

        # Get graph data from Neo4j (filter_label isolated)
        found_entities, found_relationships = self.get_graph_for_entities(
            entities=entities, depth=depth, limit=50
        )

        if not found_entities:
            return None

        # Generate Mermaid code
        mermaid_code = self.neo4j_to_mermaid(found_entities, found_relationships)

        return {
            "mermaid_code": mermaid_code,
            "nodes": len(found_entities),
            "edges": len(found_relationships),
            "entities": found_entities,
            "relationships": found_relationships,
            "filter_label": self.filter_label,
        }

    def get_tenant_stats(self) -> Optional[Dict]:
        """
        Get statistics for current tenant from Neo4j.

        Returns:
            Dict with entity/relationship counts or None if Neo4j unavailable
        """
        if not self.neo4j_driver or not self.filter_label:
            return None

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {filter_label: $filter_label})
                    WITH count(e) as entity_count
                    MATCH (d:Document {filter_label: $filter_label})
                    WITH entity_count, count(d) as doc_count
                    MATCH (c:Chunk {filter_label: $filter_label})
                    WITH entity_count, doc_count, count(c) as chunk_count
                    OPTIONAL MATCH (:Entity {filter_label: $filter_label})-[r]->(:Entity {filter_label: $filter_label})
                    WHERE r.filter_label = $filter_label
                    RETURN entity_count, doc_count, chunk_count, count(r) as rel_count
                    """,
                    {"filter_label": self.filter_label},
                )
                record = result.single()
                if record:
                    return {
                        "filter_label": self.filter_label,
                        "entities": record["entity_count"],
                        "documents": record["doc_count"],
                        "chunks": record["chunk_count"],
                        "relationships": record["rel_count"],
                    }
        except Exception as e:
            if self.debug:
                print(f"  ⚠️ Failed to get tenant stats: {e}")

        return None

    def close(self):
        """Clean up connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()


def format_chunks_for_context(chunks: List[Document]) -> str:
    """Format chunks for LLM context"""
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        doc_id = chunk.metadata.get("doc_id", "unknown")
        chunk_id = chunk.metadata.get("chunk_id", "unknown")
        context_parts.append(
            f"[CHUNK {i}]\n"
            f"Dokument: {doc_id}\n"
            f"Chunk-ID: {chunk_id}\n"
            f"Text:\n{chunk.page_content}\n"
            f"[ENDE CHUNK {i}]\n"
        )

    return "\n".join(context_parts)


def generate_answer(
    query: str,
    chunks: List[Document],
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Generate answer using LLM with configurable prompts.

    Args:
        query: User query
        chunks: Retrieved chunks
        system_prompt: Custom system prompt (uses default if None)
        user_prompt: Custom user prompt template with {context} and {query} placeholders
        model: LLM model name

    Returns:
        Generated answer string
    """
    # Post-Retrieval Synthesis with LiteLLM
    final_system_prompt = system_prompt or POST_RETRIEVAL_SYSTEM_PROMPT
    final_user_prompt = user_prompt or DEFAULT_USER_PROMPT

    # Use the specific POST model if not overridden by the call
    actual_model = model or LITELLM_POST_MODEL

    print(f"  🤖 Generating answer with reasoning model: {actual_model}")

    try:
        context = format_chunks_for_context(chunks)
        formatted_user_prompt = final_user_prompt.format(context=context, query=query)

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": formatted_user_prompt},
        ]

        # Invoke LLM via the unified handler
        content, used_model = _invoke_llm(
            messages,
            model=actual_model,
            max_tokens=LLM_MAX_OUTPUT_TOKENS
        )
        
        return content

    except Exception as e:
        return f"Fehler bei der Antwortgenerierung: {str(e)}"

def generate_answer_stream(
    query: str,
    chunks: List[Document],
    history: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    model: Optional[str] = None,
    content_mode: Optional[str] = None,
    format_template: Optional[str] = None,
):
    """
    Generate answer stream using LLM with conversation history and integrated formatting.

    When content_mode is TABLE/EMAIL/INVOICE, the formatting template is injected directly
    into the system prompt so the LLM generates the answer in the correct format in ONE shot.
    """
    final_system_prompt = system_prompt or POST_RETRIEVAL_SYSTEM_PROMPT
    final_user_prompt = user_prompt or DEFAULT_USER_PROMPT
    actual_model = model or LITELLM_POST_MODEL

    # Inject formatting template if content_mode is specified
    if content_mode and content_mode.upper() != "DEFAULT" and format_template:
        mode_upper = content_mode.upper()
        final_system_prompt += f"""

=== MANDATORY OUTPUT FORMAT: {mode_upper} ===
The user has requested their answer in {mode_upper} format. You MUST format your ENTIRE response according to these formatting rules:

{format_template}

IMPORTANT RULES FOR {mode_upper} FORMAT:
1. Your response MUST be in {mode_upper} format from the very first line. Do NOT write a regular answer first.
2. If the retrieved data does NOT contain enough information to create a complete {mode_upper}, respond with:
   - A brief explanation of what data IS available
   - A clear list of what ADDITIONAL information you need from the user to complete the {mode_upper}
   - Format this as a friendly request, e.g. "Um eine vollständige {mode_upper} zu erstellen, benötige ich noch folgende Informationen: ..."
3. Use the SAME LANGUAGE as the user's query for the formatted output.
4. Include source citations within the formatted output where applicable.
"""

    try:
        context = format_chunks_for_context(chunks)
        formatted_user_prompt = final_user_prompt.format(context=context, query=query)

        messages = [
            {"role": "system", "content": final_system_prompt},
        ]

        # Inject conversation history turns
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        # Add the current user query with context snippets
        messages.append({"role": "user", "content": formatted_user_prompt})

        return _invoke_llm_stream(
            messages,
            model=actual_model,
            max_tokens=LLM_MAX_OUTPUT_TOKENS
        )

    except Exception as e:
        print(f"Streaming Error: {e}")
        raise

    async def graphrag_retrieval_async(
        self, query: str, k: int = 20, debug: bool = None, status_callback=None
    ) -> Tuple[List[Document], Dict]:
        """
        Async version: Strategic Context-Driven Retrieval via asyncio.
        status_callback: Async function to receive progress updates {"step": str, "details": str}
        """
        if debug is None:
            debug = self.debug

        # --- STEP 1: STRATEGIC PLANNING (Async) ---
        print(f"🎯 Planning retrieval strategy (Async) for: '{query[:50]}...'")
        
        if status_callback: await status_callback({"step": "planning", "details": "Analyzing query intent..."})
        loop = asyncio.get_running_loop()
        plan = await loop.run_in_executor(None, self.plan_retrieval, query)
        if status_callback: await status_callback({"step": "planning_done", "details": f"Mode: {plan.get('mode')}"})
        
        search_mode = plan.get("mode", "LOCAL_SEARCH")

        # --- RESPONSE BRANCH A: SMALL TALK ---
        if search_mode == "SMALL_TALK":
            print("  ☕ Direct conversational reply (No DB search needed)")
            system_doc = Document(
                page_content=plan.get("direct_reply") or "Hallo! Wie kann ich Ihnen bei Ihren Dokumenten helfen?",
                metadata={"mode": "small_talk", "is_direct": True}
            )
            return [system_doc], {"mode": "small_talk", "plan": plan}

        # --- RESPONSE BRANCH B: GLOBAL HIVE ---
        if search_mode == "GLOBAL_SEARCH":
            print("  🕸️ Global Hive Mode: Strategic summary across corpus")
            summary = await loop.run_in_executor(None, self.generate_global_hive_summary, query)
            if summary:
                global_doc = Document(
                    page_content=summary,
                    metadata={"mode": "global_hive", "is_direct": True}
                )
                return [global_doc], {"mode": "global_hive", "plan": plan}
            print("  ⚠️ Global summary failed, falling back to local search")

        # --- RESPONSE BRANCH C: LOCAL SEARCH (Async Parallel) ---
        print(f"🔍 Executing Local Search (tenant: {self.filter_label})...")

        # Use unified planner's entities and keywords (no separate LLM calls)
        entities = plan.get("entities", [])
        keywords = plan.get("keywords", [])
        if status_callback: await status_callback({"step": "extraction_done", "details": f"Found {len(entities)} entities, {len(keywords)} keywords"})

        search_plan = plan.get("search_plan", {"use_vector": True, "use_graph": True, "use_keyword": True})
        broad_k = k * 10

        # Build expanded_query from planner keywords
        expanded_query = {
            "original": query,
            "keywords": keywords,
            "numbers": [k for k in keywords if any(c.isdigit() for c in str(k))],
            "years": [], "percentages": [], "expected_patterns": [],
        }

        # Launch tasks concurrently
        tasks = []
        task_types = []

        # 1. GRAPH
        if search_plan.get("use_graph") and self.neo4j_driver and entities and self.filter_label:
             tasks.append(loop.run_in_executor(None, self.entity_based_retrieval, entities, broad_k))
             task_types.append("graph")

        # 2. VECTOR
        if search_plan.get("use_vector"):
             tasks.append(loop.run_in_executor(None, self.vector_search, query, broad_k))
             task_types.append("vector")

        # 3. KEYWORD
        if search_plan.get("use_keyword"):
             tasks.append(loop.run_in_executor(None, self.keyword_search, query, expanded_query, broad_k))
             task_types.append("keyword")

        # Wait for all
        if status_callback: await status_callback({"step": "search_start", "details": f"Launching {len(tasks)} parallel searches..."})
        results = await asyncio.gather(*tasks) if tasks else []
        
        rankings = {}
        for type_name, res in zip(task_types, results):
            if res:
                rankings[type_name] = res
                print(f"  ✅ {type_name.capitalize()} search done: {len(res)} results")
                if status_callback: await status_callback({"step": "search_done", "details": f"{type_name.capitalize()} search found {len(res)} results"})

        # 4. ADJACENT CHUNKS (Requires I/O too)
        all_top = []
        for r_dict in rankings.values():
            all_top.extend(list(r_dict.keys())[:10])
        all_top = list(set(all_top[:30]))

        if all_top:
            adjacent_results = await loop.run_in_executor(None, self.get_adjacent_chunks, all_top[:20], 1)
            if adjacent_results:
                rankings["adjacent"] = adjacent_results

        # Stage 4: Fusion
        if not rankings:
             return [], {"mode": "error", "error": "No relevant data found"}

        if "graph" in rankings:
            weights = {"graph": 0.40, "vector": 0.45, "keyword": 0.10, "adjacent": 0.05}
        else:
            weights = {"vector": 0.70, "keyword": 0.25, "adjacent": 0.05}

        fused_results = self.weighted_rrf_fusion(rankings, weights=weights, k=60)
        
        # Retrieve final chunks (I/O bound)
        chunks = await loop.run_in_executor(None, self._retrieve_chunks, fused_results[:k])

        stats = {
            "mode": "local_search_async",
            "planning": plan,
            "graph_used": "graph" in rankings,
            "vector_used": "vector" in rankings,
            "keyword_used": "keyword" in rankings,
            "chunks_retrieved": len(chunks),
            "filter_label": self.filter_label
        }

        return chunks, stats
