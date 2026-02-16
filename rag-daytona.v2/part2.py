    async def gemini_only_query(
        self,
        query_text: str,
        context: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[str, bool], None]] = None
    ) -> Dict[str, Any]:
        """Fallback when embeddings/vector store unavailable."""
        start_time = time.time()
        
        try:
            if not self.documents:
                # Standalone Gemini mode
                prompt = f"""You are Lexi, a helpful university assistant for Daytona.
Answer this question to the best of your ability:

Question: {query_text}

Provide a helpful response even without specific knowledge base context."""
            else:
                # Keyword-based selection with improved relevance filtering
                query_lower = query_text.lower()
                
                # Check for vague queries that shouldn't use RAG
                vague_indicators = [
                    "what does that mean", "what do you mean", "what does this mean",
                    "what is that", "what is this", "what are you", "who are you",
                    "how does it work", "how do you work", "tell me about yourself"
                ]
                
                if any(indicator in query_lower for indicator in vague_indicators):
                    # For vague queries, provide a direct helpful response
                    prompt = f"""You are Lexi, a helpful university assistant for Daytona.

A user asked: "{query_text}"

This seems like a vague or general question. Provide a brief, helpful response that acknowledges the question and offers assistance. Keep it under 200 characters.

Response:"""
                else:
                    # Normal RAG processing
                    keywords = query_text.lower().split()
                    if context and 'key_entities' in context:
                        for entity_value in context['key_entities'].values():
                            if isinstance(entity_value, str):
                                keywords.extend(entity_value.lower().split())
                    
                    # Improved relevance scoring
                    doc_scores = []
                    for i, doc_text in enumerate(self.documents):
                        doc_lower = doc_text.lower()
                        # More sophisticated scoring: exact phrases, proper nouns, etc.
                        score = 0
                        for keyword in keywords:
                            if len(keyword) > 3:  # Only count meaningful keywords
                                score += doc_lower.count(keyword)
                        
                        # Boost score for documents with multiple keywords
                        unique_keywords = set(keywords)
                        keyword_matches = sum(1 for kw in unique_keywords if len(kw) > 3 and kw in doc_lower)
                        if keyword_matches >= 2:  # Require at least 2 meaningful keyword matches
                            score *= 1.5
                        
                        if score > 0:
                            doc_meta = self.doc_metadata[i] if i < len(self.doc_metadata) else {}
                            doc_scores.append({'text': doc_text, 'metadata': doc_meta, 'score': score})
                    
                    doc_scores.sort(key=lambda x: -x['score'])
                    relevant_docs = doc_scores[:self.config.top_n] if doc_scores else []
                    
                    if not relevant_docs:
                        # No relevant documents found
                        prompt = f"""You are Lexi, a helpful university assistant for Daytona.

A user asked: "{query_text}"

I couldn't find specific information about this in my knowledge base. Provide a helpful response that acknowledges this and offers alternative assistance.

Response:"""
                    else:
                        context_text = "\n\n".join([doc['text'] for doc in relevant_docs])
                        sources = list(set([doc['metadata'].get('source', 'Unknown') for doc in relevant_docs]))
                        
                        prompt = f"""Answer this question using the provided context:

Context: {context_text}

Question: {query_text}

Response:"""
            
            gen_start = time.time()
            response = self.gemini_model.generate_content(prompt) if self.gemini_model else None
            raw_response = response.text.strip() if response else "Service unavailable."
            
            timing = {
                'generation_ms': (time.time() - gen_start) * 1000,
                'total_ms': (time.time() - start_time) * 1000
            }
            
            return {
                'answer': raw_response,
                'sources': sources if self.documents else [],
                'confidence': 0.5,
                'timing_breakdown': timing,
                'metadata': {'fallback_mode': 'gemini_only'}
            }
        
        except Exception as e:
            logger.error(f"Gemini-only query error: {e}")
            return {
                'answer': "I'm having trouble generating a response. Please try again.",
                'sources': [],
                'confidence': 0.0,
                'timing_breakdown': {'total_ms': (time.time() - start_time) * 1000},
                'metadata': {'error': str(e)}
            }
    
    def humanize_response(self, response: str, query: str, context: Optional[Dict[str, Any]] = None, is_first_turn: bool = True) -> str:
        """Make responses conversational."""
        if not response:
            return response
        
        # Remove formal prefixes
        formal_prefixes = [
            "According to the context",
            "Based on the information provided",
            "The knowledge base states",
            "As per the document"
        ]
        for prefix in formal_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].lstrip(' ,:-')
        
        # Add conversational starters for first turn
        if is_first_turn and self.config.response_style == "friendly_casual":
            query_lower = query.lower()
            
            # Use hash for deterministic selection
            hash_val = int(hashlib.md5(query.encode()).hexdigest(), 16)
            
            starters_how = ["Here's how you can", "Here's the process", "This is how it works"]
            starters_what = ["Here's what you need to know", "Basically", "Let me explain"]
            starters_where = ["You can find it at", "The location is", "It's located at"]
            starters_when = ["The schedule is", "It happens", "Timing-wise"]
            starters_why = ["The reason is", "This is because", "Here's why"]
            starters_default = ["Sure", "Absolutely", "Good question"]
            
            if query_lower.startswith('how'):
                starter = starters_how[hash_val % len(starters_how)]
            elif query_lower.startswith('what'):
                starter = starters_what[hash_val % len(starters_what)]
            elif query_lower.startswith('where'):
                starter = starters_where[hash_val % len(starters_where)]
            elif query_lower.startswith('when'):
                starter = starters_when[hash_val % len(starters_when)]
            elif query_lower.startswith('why'):
                starter = starters_why[hash_val % len(starters_why)]
            else:
                starter = starters_default[hash_val % len(starters_default)]
            
            if not response.startswith(starter):
                response = f"{starter}: {response[0].lower()}{response[1:]}"
        
        # Ensure proper capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Ensure proper ending
        if response and response[-1] not in '.!?':
            response += '.'
        
        # Add helpful ending only if it won't exceed max length
        max_length = self.config.max_response_length
        if len(response) > 150 and len(response) < max_length - 50:
            helpful_endings = [
                " Let me know if you need more details!",
                " Feel free to ask if you have more questions.",
                " Hope that helps!"
            ]
            hash_val = int(hashlib.md5(response.encode()).hexdigest(), 16)
            ending = helpful_endings[hash_val % len(helpful_endings)]
            if not any(response.endswith(e.strip()) for e in helpful_endings):
                if len(response) + len(ending) <= max_length:
                    response += ending
        
        # Enforce max length
        if len(response) > self.config.max_response_length:
            sentences = re.split(r'[.!?]\s+', response)
            truncated = ""
            for sentence in sentences:
                if len(truncated) + len(sentence) < self.config.max_response_length:
                    truncated += sentence + '. '
                else:
                    break
            response = truncated.strip()
        
        return response
    
    def validate_response_quality(self, response: str) -> Dict[str, Any]:
        """Validate response quality."""
        issues = []
        quality_score = 1.0
        
        # Check for formal language
        formal_words = ['pursuant', 'aforementioned', 'hereby', 'henceforth', 'notwithstanding']
        if any(word in response.lower() for word in formal_words):
            issues.append("overly_formal")
            quality_score -= 0.3
        
        # Check for jargon
        jargon_words = ['matriculation', 'pedagogy', 'curriculum vitae']
        if any(word in response.lower() for word in jargon_words):
            issues.append("unexplained_jargon")
            quality_score -= 0.2
        
        # Check length
        if len(response) < 30:
            issues.append("too_short")
            quality_score -= 0.3
        elif len(response) > 450:
            issues.append("too_long")
            quality_score -= 0.1
        
        # Check for unhelpfulness
        unhelpful_phrases = ["i don't know", "not sure", "can't help"]
        if any(phrase in response.lower() for phrase in unhelpful_phrases):
            if "but" not in response.lower() and "however" not in response.lower():
                issues.append("unhelpful_without_alternative")
                quality_score -= 0.4
        
        retry = quality_score < 0.5
        
        return {
            'issues': issues,
            'quality_score': max(0.0, quality_score),
            'retry': retry
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Return performance metrics."""
        return {
            'total_queries': self.query_count,
            'average_query_time': self.total_query_time / self.query_count if self.query_count > 0 else 0.0,
            'vector_store_size': len(self.documents),
            'categories_loaded': len(set(m.get('category', '') for m in self.doc_metadata)),
            'embeddings_available': self.embeddings is not None,
            'gemini_available': self.gemini_model is not None
        }
