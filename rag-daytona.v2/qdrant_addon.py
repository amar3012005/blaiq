import os
import time
import uuid
import logging
from typing import List, Dict, Optional, Any, Union

# Try to import qdrant_client - graceful degradation if missing
try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)

class QdrantAddon:
    """
    The 'Memory Add-on' for TARA.
    Handles 'Case Memory' and 'Knowledge Capsules'.
    """
    
    def __init__(self, 
                 embedding_dim: int = 1024, 
                 url: Optional[str] = None, 
                 api_key: Optional[str] = None, 
                 collection_name: Optional[str] = None):
        """Initialize Qdrant addon."""
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "tara_case_memory")
        self.embedding_dim = embedding_dim
        self.enabled = False
        self.client = None
        
        if not QDRANT_AVAILABLE:
            logger.warning("⚠️ qdrant-client not installed. Memory features disabled.")
            return

        if not self.url:
            logger.warning("⚠️ QDRANT_URL not set. Memory features disabled.")
            return
            
        try:
            # check_compatibility=False to avoid 403 errors during version check
            self.client = AsyncQdrantClient(
                url=self.url, 
                api_key=self.api_key,
                check_compatibility=False,
                timeout=0.4  # 400ms - fast fail for Hive Mind queries
            )
            # Sync check for startup
            sync_check = QdrantClient(url=self.url, api_key=self.api_key, check_compatibility=False)
            if sync_check.collection_exists(self.collection_name):
                self.enabled = True
            else:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                sync_check.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                self.enabled = True
            
            if self.enabled:
                logger.info(f"✅ Qdrant Memory initialized (URL: {self.url})")
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {e}")
            self.enabled = False

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if self.client is None: 
            return
        
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
            self.enabled = True  # Successfully connected and verified collection
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            self.enabled = False

    async def upsert_case(self, 
                    user_id: str, 
                    issue: str, 
                    solution: str, 
                    vector: List[float],
                    tenant_id: str = "demo",
                    domain: Optional[str] = None,
                    metadata: Optional[Dict] = None):
        """
        Store a resolved case in memory.
        
        Args:
            user_id: User identifier
            issue: Problem description
            solution: Resolution steps
            vector: Embedding vector
            tenant_id: Tenant identifier
            domain: Website domain (e.g., "groq.com") for domain-specific knowledge
            metadata: Additional metadata
        """
        if not self.enabled: return
        
        try:
            payload = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "issue": issue,
                "solution": solution,
                "timestamp": int(time.time()),
                "successful": True
            }
            
            # Add domain if provided
            if domain:
                payload["domain"] = domain
                logger.debug(f"💾 Storing case for domain: {domain}")
            
            if metadata:
                payload.update(metadata)
                
            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            logger.info(f"🧠 Learned new case from User {user_id}: {issue[:30]}...")
        except Exception as e:
            logger.error(f"Failed to upsert case: {e}")

    async def search_hive_mind(self, query_vector: List[float], tenant_id: str = "demo", domain: Optional[str] = None, limit: int = 3, score_threshold: float = 0.4) -> List[Dict]:
        """
        Search GLOBAL memory (Hive Mind) for similar issues within the same tenant and domain.
        Returns list of solutions.
        
        Args:
            query_vector: Embedding vector for semantic search
            tenant_id: Tenant identifier for multi-tenancy
            domain: Optional domain filter (e.g., "groq.com", "airbnb.com")
            limit: Maximum number of results
            score_threshold: Minimum similarity score
        """
        if not self.enabled: return []
        
        try:
            # Multi-tenant filter with optional domain filtering
            filter_conditions = [
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id)
                )
            ]
            
            # Add domain filter if provided (MAPPED MODE)
            if domain:
                filter_conditions.append(
                    models.FieldCondition(
                        key="domain",
                        match=models.MatchValue(value=domain)
                    )
                )
                logger.debug(f"🗺️ MAPPED MODE: Filtering HiveMind by domain={domain}")
            else:
                logger.debug("🧭 EXPLORER MODE: No domain filter (searching all domains)")
            
            query_filter = models.Filter(must=filter_conditions)

            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            hits = response.points
            
            if hits:
                domain_info = f" [domain={domain}]" if domain else " [all domains]"
                logger.info(f"🧠 Hive Mind{domain_info}: {len(hits)} matches (top score: {hits[0].score:.3f})")
            else:
                domain_info = f" for domain={domain}" if domain else ""
                logger.info(f"🧠 Hive Mind: No matches{domain_info} above threshold {score_threshold}")
            
            results = []
            for hit in hits:
                results.append({
                    "solution": hit.payload.get("solution"),
                    "issue": hit.payload.get("issue"),
                    "user_id": hit.payload.get("user_id"), # Attribution
                    "confidence": hit.score,
                    "timestamp": hit.payload.get("timestamp")
                })
            return results
        except Exception as e:
            logger.error(f"Hive Mind search failed: {e}")
            return []
    
    async def check_domain_has_knowledge(self, domain: str, tenant_id: str = "demo") -> bool:
        """
        Check if HiveMind has any knowledge for a specific domain.
        Used to determine MAPPED vs EXPLORER mode.
        
        Args:
            domain: Domain to check (e.g., "groq.com")
            tenant_id: Tenant identifier
            
        Returns:
            True if domain has HiveMind knowledge (MAPPED MODE)
            False if domain is unknown (EXPLORER MODE)
        """
        if not self.enabled:
            return False
        
        try:
            # Count points for this domain
            result = await self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id",
                            match=models.MatchValue(value=tenant_id)
                        ),
                        models.FieldCondition(
                            key="domain",
                            match=models.MatchValue(value=domain)
                        )
                    ]
                )
            )
            
            has_knowledge = result.count > 0
            mode = "MAPPED" if has_knowledge else "EXPLORER"
            logger.info(f"🗺️ Domain '{domain}': {result.count} HiveMind entries → {mode} MODE")
            return has_knowledge
            
        except Exception as e:
            logger.error(f"Failed to check domain knowledge: {e}")
            return False

    async def search_user_history(self, user_id: str, query_vector: List[float], tenant_id: str = "demo", limit: int = 3) -> List[Dict]:
        """
        Search PERSONAL memory for recall within the same tenant.
        """
        if not self.enabled: return []
        
        try:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id)
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            )

            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            )
            hits = response.points
            
            results = []
            for hit in hits:
                results.append({
                    "suggestion": f"On {time.strftime('%Y-%m-%d', time.localtime(hit.payload.get('timestamp', 0)))}, you fixed '{hit.payload.get('issue')}' by: {hit.payload.get('solution')}",
                    "confidence": hit.score
                })
            return results
        except Exception as e:
            logger.error(f"User history search failed: {e}")
            return []

    def get_user_context(self, user_id: str, limit: int = 1) -> List[Dict]:
        """
        Get most recent interaction for this user (for greeting).
        """
        if not self.enabled: return []
        
        try:
            # Scroll/Query by user_id sorted by timestamp desc
            # Note: Qdrant scroll doesn't sort by payload field easily efficiently without index
            # Ideally we would use filter + dummy vector or scroll. 
            # Simplified: Just return empty or implement simple scroll if supported.
            # actually better to use filter
            
            # Using scroll for retrieving latest by filter is tricky.
            # Using search with zero vector works if we just want "any" or we can search with a generic "problem" vector.
            pass 
        except Exception:
            pass
        return []

    async def delete_user_memory(self, user_id: str, tenant_id: str = "demo"):
        """GDPR Compliance: Delete all vectors for a user within a tenant."""
        if not self.enabled: return
        
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_id",
                                match=models.MatchValue(value=tenant_id)
                            ),
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"🗑️ Deleted memory for User {user_id} (GDPR)")
        except Exception as e:
            logger.error(f"Failed to delete user memory: {e}")
