
    async def graphrag_retrieval_async(
        self, query: str, k: int = 20, debug: bool = None
    ) -> Tuple[List[Document], Dict]:
        """Async version: Strategic Context-Driven Retrieval via asyncio."""
        if debug is None:
            debug = self.debug

        # --- STEP 1: STRATEGIC PLANNING (Async) ---
        print(f"🎯 Planning retrieval strategy (Async) for: '{query[:50]}...'")
        
        loop = asyncio.get_running_loop()
        plan = await loop.run_in_executor(None, self.plan_retrieval, query)
        
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

        entities = plan.get("entities_german", [])
        if not entities:
            entities = await loop.run_in_executor(None, self.extract_entities_with_llm, query)

        search_plan = plan.get("search_plan", {"use_vector": True, "use_graph": True, "use_keyword": True})
        broad_k = k * 10
        
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
             expanded_query = self.expand_query_with_cot(query) # Fast enough
             tasks.append(loop.run_in_executor(None, self.keyword_search, query, expanded_query, broad_k))
             task_types.append("keyword")

        # Wait for all
        results = await asyncio.gather(*tasks) if tasks else []
        
        rankings = {}
        for type_name, res in zip(task_types, results):
            if res:
                rankings[type_name] = res
                print(f"  ✅ {type_name.capitalize()} search done: {len(res)} results")

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
