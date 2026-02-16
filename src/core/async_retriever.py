"""
Async Parallel Retrieval for GraphRAG
Runs Vector, Graph, and Keyword searches concurrently.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor


class AsyncRetriever:
    """
    Wrapper to run synchronous retrieval methods in parallel using asyncio.
    
    This converts the sequential GraphRAG pipeline into concurrent execution:
    - Vector search (Qdrant)
    - Graph search (Neo4j)
    - Keyword search (Qdrant)
    
    Expected speedup: 2-3x for retrieval phase
    """
    
    def __init__(self, retriever, max_workers: int = 3):
        """
        Initialize async wrapper.
        
        Args:
            retriever: GraphRAGRetriever instance
            max_workers: Max concurrent threads
        """
        self.retriever = retriever
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def run_vector_search(self, query: str, k: int) -> Tuple[Dict[int, float], float]:
        """Run vector search in thread pool."""
        start = time.time()
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            self.executor,
            self.retriever.vector_search,
            query,
            k
        )
        
        elapsed = time.time() - start
        return results, elapsed
    
    async def run_graph_search(
        self,
        entities: List[str],
        k: int
    ) -> Tuple[Dict[int, float], float]:
        """Run graph search in thread pool."""
        if not entities or not self.retriever.neo4j_driver:
            return {}, 0.0
        
        start = time.time()
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            self.executor,
            self.retriever.entity_based_retrieval,
            entities,
            k
        )
        
        elapsed = time.time() - start
        return results, elapsed
    
    async def run_keyword_search(
        self,
        query: str,
        expanded_query: Dict,
        k: int
    ) -> Tuple[Dict[int, float], float]:
        """Run keyword search in thread pool."""
        start = time.time()
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            self.executor,
            self.retriever.keyword_search,
            query,
            expanded_query,
            k
        )
        
        elapsed = time.time() - start
        elapsed = time.time() - start
        return results, elapsed
    
    async def run_hive_mind_search(self, query: str) -> Tuple[Optional[str], float]:
        """Run Hive Integrity search (Global Summary)."""
        start = time.time()
        loop = asyncio.get_event_loop()
        
        summary = await loop.run_in_executor(
            self.executor,
            self.retriever.generate_global_hive_summary,
            query
        )
        
        elapsed = time.time() - start
        return summary, elapsed
    
    async def parallel_retrieval(
        self,
        query: str,
        entities: List[str],
        expanded_query: Dict,
        k: int,
        plan: Optional[Dict] = None
    ) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float], Dict[str, Any]]:
        """
        Run all retrieval methods in parallel, filtered by plan.
        
        Args:
            query: User query
            entities: Extracted entities
            expanded_query: Query expansion results
            k: Number of results per method
            plan: Dict with keys 'use_vector', 'use_graph', 'use_keyword'
            
        Returns:
            (rankings dict, timing dict)
        """
        if plan is None:
            # Default to all enabled
            plan = {"use_vector": True, "use_graph": True, "use_keyword": True}
        # Launch all searches concurrently
        # Launch searches concurrently based on plan
        tasks = []
        task_map = {} # Map index to type
        
        idx = 0
        if plan.get("use_vector", True):
            tasks.append(self.run_vector_search(query, k))
            task_map[idx] = "vector"
            idx += 1
            
        if plan.get("use_graph", True):
            tasks.append(self.run_graph_search(entities, k))
            task_map[idx] = "graph"
            idx += 1
            
        if plan.get("use_keyword", True):
            tasks.append(self.run_keyword_search(query, expanded_query, k))
            task_map[idx] = "keyword"
            idx += 1
            
        if plan.get("use_hive_mind", False):
            # HiveMind needs raw logic or specific query
            tasks.append(self.run_hive_mind_search(query))
            task_map[idx] = "hive_mind"
            idx += 1
        
        # Wait for all to complete with robustness
        if not tasks:
            return {}, {}, {}
            
        # Use return_exceptions=True to ensure one failure doesn't crash the whole pipeline
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Unpack results dynamically
        rankings = {}
        timings = {}
        special_results = {}
        
        max_time = 0.0
        
        for i, task_result in enumerate(results):
            type_key = task_map.get(i, "unknown")
            
            # Check if task raised an exception
            if isinstance(task_result, Exception):
                print(f"  ❌ Parallel task '{type_key}' failed: {task_result}")
                timings[f"{type_key}_search"] = 0.0
                continue
                
            # Otherwise unpack the normal (res, elapsed) tuple
            res, elapsed = task_result
            
            if type_key == "hive_mind":
                if res:
                    special_results["hive_mind"] = res
            else:
                if res:
                    rankings[type_key] = res
            
            timings[f"{type_key}_search"] = elapsed
            if elapsed > max_time:
                max_time = elapsed
                
        timings["total_parallel"] = max_time
        
        return rankings, timings, special_results
    
    def shutdown(self):
        """Shutdown thread pool."""
        self.executor.shutdown(wait=True)
