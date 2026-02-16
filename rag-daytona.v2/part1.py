"""
RAG Engine Core Logic

Ported from leibniz_rag.py for microservice deployment.

Reference:
    - leibniz_rag.py (lines 62-1262) - Original LeibnizRAG class
"""

import os
import json
import time
import logging
import hashlib
import re
import asyncio
import httpx
from typing import Dict, Any, Optional, List, Callable
import numpy as np
import faiss
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings

from daytona_agent.services.rag.config import RAGConfig
from daytona_agent.services.rag.prompts import prompt_manager

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Core RAG engine with FAISS retrieval and Gemini generation.
    
    Attributes:
        config: RAG configuration
        embeddings: HuggingFace embeddings model
        gemini_model: Gemini model instance
        vector_store: FAISS index
        documents: Document chunks
        doc_metadata: Chunk metadata
        query_count: Query counter
        total_query_time: Cumulative query time
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initialize RAG engine with configuration.
        
        Args:
            config: RAG configuration instance
        """
        self.config = config
        
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Gemini model
        self.gemini_model = None
        if config.gemini_api_key:
            try:
                genai.configure(api_key=config.gemini_api_key)
                self.gemini_model = genai.GenerativeModel(config.gemini_model)
                logger.info(f" Gemini model initialized: {config.gemini_model}")
            except Exception as e:
                logger.error(f" Gemini initialization failed: {e}")
        else:
            logger.warning("️ No Gemini API key - response generation unavailable")
        
        # Storage
        self.vector_store = None
        self.documents: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        
        # HYBRID APPROACH: Rule-based patterns for instant context reduction
        # Reduces Gemini token count by 50-70% for common queries
        self.quick_answer_patterns = {
            "office_hours": {
                "keywords": ["office hours", "opening hours", "working hours", "open time", "office time", "hours of operation"],
                "response_template": "The {department} is open {hours}. {additional_info}",
                "faiss_boost": ["office_hours", "contact_information"],
                "max_context_chars": 2000,
                "priority": 1
            },
            "contact_info": {
                "keywords": ["contact", "email", "phone", "call", "reach", "get in touch", "contact information"],
                "response_template": "You can contact {department} via {contact_methods}.",
                "faiss_boost": ["contact_information"],
                "max_context_chars": 1500,
                "priority": 1
            },
            "admission_requirements": {
                "keywords": ["admission", "requirements", "eligibility", "entry", "apply", "application"],
                "response_template": "For admission to {program}, you need: {requirements}.",
                "faiss_boost": ["admission", "masters_admission", "bachelors_admission"],
                "max_context_chars": 2500,
                "priority": 2
            },
            "appointment_scheduling": {
                "keywords": ["appointment", "schedule", "book", "meeting", "reservation", "slot"],
                "response_template": "To schedule an appointment: {steps}",
                "faiss_boost": ["appointment_scheduling", "administrative_procedures"],
                "max_context_chars": 1800,
                "priority": 1
            },
            "tuition_fees": {
                "keywords": ["tuition", "fees", "cost", "price", "payment", "semester fee"],
                "response_template": "The {program} tuition is {amount}. {payment_info}",
                "faiss_boost": ["tuition", "fees", "financial"],
                "max_context_chars": 1800,
                "priority": 2
            },
            "academic_calendar": {
                "keywords": ["semester", "calendar", "academic year", "term dates", "exam schedule"],
                "response_template": "The academic calendar shows: {calendar_info}",
                "faiss_boost": ["academic_calendar", "academic_policies"],
                "max_context_chars": 2200,
                "priority": 2
            },
            "ambiguous_question": {
                # Full phrases only - NOT single words that match too broadly
                "keywords": ["what does that mean", "what do you mean", "what does this mean", "what is that", "what are you talking about", "i don't understand", "what do you mean by that", "can you explain that", "what does it mean", "what's that", "huh", "was bedeutet das", "was meinst du", "was bedeutet das hier", "was ist das", "wovon redest du", "ich verstehe nicht", "was meinst du damit", "kannst du das erklären", "was bedeutet es", "häh", "wie meinst du das", "was soll das heißen", "ich versteh nicht", "was meinst du genau"],
                "response_template": {
                    "en": "I'm not sure I understand your question. Could you please rephrase it or be more specific?",
                    "de": "Ich bin mir nicht sicher, ob ich Ihre Frage verstehe. Könnten Sie sie bitte umformulieren oder genauer sein?"
                },
                "faiss_boost": [],
                "max_context_chars": 0,
                "priority": 0  # Lowest priority - only match when nothing else does
            }
        }
        
        # Load index
        self.load_index()
    
    def load_index(self) -> bool:
        """
        Load pre-built FAISS index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            index_path = os.path.join(self.config.vector_store_path, "index.faiss")
            metadata_path = os.path.join(self.config.vector_store_path, "metadata.json")
            texts_path = os.path.join(self.config.vector_store_path, "texts.json")
            
            # Validate files exist
            if not all(os.path.exists(p) for p in [index_path, metadata_path, texts_path]):
                logger.error(f" Index files not found at {self.config.vector_store_path}")
                return False
            
            # Load FAISS index
            self.vector_store = faiss.read_index(index_path)
            
            # Load metadata and texts
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.doc_metadata = json.load(f)
            
            with open(texts_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            logger.info(f" Loaded FAISS index: {len(self.documents)} documents")
            return True
        
        except Exception as e:
            logger.error(f" Error loading index: {e}", exc_info=True)
            return False
    
    def _detect_query_pattern(self, query: str) -> Optional[Dict]:
        """
        Detect if query matches known patterns for hybrid retrieval (1-5ms)
        
        Args:
            query: User query text
            
        Returns:
            Pattern config dict with:
                - name: str
                - keywords: list
                - response_template: str
                - faiss_boost: list
                - max_context_chars: int
                - priority: int
            or None if no pattern match
        """
        query_lower = query.lower()
        
        # Add synonym mappings for fuzzy matching
        synonyms = {
            "office_hours": [
                "office hours", "opening hours", "working hours", "open time", "office time", 
                "hours of operation", "when open", "timing", "slot availability", "available hours",
                "office timing", "work hours", "working time"
            ],
            "contact_info": [
                "contact", "email", "phone", "call", "reach", "get in touch", "contact information",
                "contact details", "how to reach", "reach out", "speak to", "talk to"
            ],
            "admission_requirements": [
                "admission", "requirements", "eligibility", "entry", "apply", "application",
                "admission criteria", "entry requirements", "qualification", "prerequisites"
            ],
            "appointment_scheduling": [
                "appointment", "schedule", "book", "meeting", "reservation", "slot",
                "book slot", "make appointment", "reserve", "schedule meeting"
            ],
            "tuition_fees": [
                "tuition", "fees", "cost", "price", "payment", "semester fee",
                "tuition cost", "fee structure", "charges", "how much"
            ],
            "academic_calendar": [
                "semester", "calendar", "academic year", "term dates", "exam schedule",
                "semester dates", "academic schedule", "exam dates"
            ],
            "ambiguous_question": [
                # Full phrases only - NOT single words like "what" that match too broadly
                "what does that mean", "what do you mean", "what does this mean", "what is that", 
                "what are you talking about", "i don't understand", "what do you mean by that",
                "can you explain that", "what does it mean", "what's that", "huh",
                # German equivalents - full phrases only
                "was bedeutet das", "was meinst du", "was bedeutet das hier", "was ist das",
                "wovon redest du", "ich verstehe nicht", "was meinst du damit",
                "kannst du das erklären", "was bedeutet es", "häh",
                "wie meinst du das", "was soll das heißen", "ich versteh nicht",
                "was meinst du genau"
            ]
        }
        
        # Find best matching pattern (highest priority + fuzzy threshold)
        best_match = None
        best_priority = 0
        best_match_count = 0
        
        for pattern_name, pattern_config in self.quick_answer_patterns.items():
            # Use extended keywords from synonyms
            extended_keywords = synonyms.get(pattern_name, pattern_config["keywords"])
            
            # Count keyword matches for fuzzy threshold
            match_count = sum(1 for keyword in extended_keywords if keyword in query_lower)
            
            # Fuzzy regex patterns for hours/timing
            if pattern_name == "office_hours":
                # Check for time-related patterns
                time_pattern = r'\d{1,2}:\d{2}|\d{1,2}\s?(am|pm)|hours?|timing|schedule|open|available'
                if re.search(time_pattern, query_lower):
                    match_count += 1
            
            # Lower threshold - require at least 1 match
            if match_count > 0:
                # Priority tie-breaker: more matches = better
                if (pattern_config.get("priority", 0) > best_priority or 
                    (pattern_config.get("priority", 0) == best_priority and match_count > best_match_count)):
                    best_match = {"name": pattern_name, **pattern_config}
                    best_priority = pattern_config.get("priority", 0)
                    best_match_count = match_count
        
        if best_match:
            logger.debug(f" Pattern detected: {best_match['name']} (match_count: {best_match_count})")
        
        return best_match
    
    def _retrieve_with_boosting(self, query: str, context: Dict, boost_categories: list, max_context_chars: int) -> tuple:
        """
        FAISS retrieval with category boosting and context truncation
        
        Args:
            query: User query text
            context: Structured context from intent parser
            boost_categories: List of categories/keywords to boost
            max_context_chars: Maximum total character count for context
            
        Returns:
            tuple: (relevant_docs: list, timing_dict: dict)
        """
        timing = {}
        
        # ENHANCED: Enrich query with entities AND specific keywords from query
        enriched_query = query
        
        # Extract specific department/office mentions from query
        dept_keywords = {
            "registrar": "registrar office registry student records enrollment verification transcript certificate exmatriculation",
            "admissions": "admissions office admission requirements application enroll apply eligibility",
            "academic": "academic affairs dean faculty program curriculum course",
            "student services": "student services counseling support advisory wellness",
            "international": "international office exchange study abroad visa foreign",
            "financial": "financial aid bafög scholarship tuition fees payment funding",
            "transcript": "transcript records registry registrar student services academic records",
            "certificate": "certificate certification registrar student services verification document",
        }
        
        # Add department-specific keywords if mentioned
        query_lower = query.lower()
        for dept, keywords in dept_keywords.items():
            if dept in query_lower:
                enriched_query = f"{query} {keywords}"
                logger.debug(f" Query enriched with department keywords: {dept}")
                break
        
        # Add entities from context
        if context and 'key_entities' in context:
            entities = context['key_entities']
            entity_terms = ' '.join([f"{k} {v}" for k, v in entities.items()])
            enriched_query = f"{enriched_query} {entity_terms}"
        
        # Embed query
        embed_start = time.time()
        query_embedding = self.embeddings.embed_query(enriched_query)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        timing['embedding_ms'] = (time.time() - embed_start) * 1000
        
        # FAISS search (retrieve more candidates for filtering)
        search_start = time.time()
        distances, indices = self.vector_store.search(query_embedding, k=self.config.top_k + 5)
        timing['search_ms'] = (time.time() - search_start) * 1000
        
        # Build candidates with boosting
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                distance = float(distances[0][i])
                similarity = 1.0 - (distance * distance / 2.0)
                
                # Skip low-similarity docs
                if similarity < self.config.similarity_threshold:
                    continue
                
                doc_text = self.documents[idx]
                doc_meta = self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
                
                # Apply category boosting
                boosted_similarity = similarity
                doc_category = doc_meta.get('category', '').lower()
                doc_source = doc_meta.get('source', '').lower()
                
                # Check if doc matches any boost category
                for boost_cat in boost_categories:
                    boost_cat_lower = boost_cat.lower()
                    if boost_cat_lower in doc_category or boost_cat_lower in doc_source:
                        boosted_similarity *= 1.5  # 50% boost!
                        logger.debug(f" Boosted {doc_source} (matched '{boost_cat}')")
                        break  # Only boost once
                
                candidates.append({
                    'text': doc_text,
                    'metadata': doc_meta,
                    'distance': distance,
                    'similarity': similarity,
                    'boosted_similarity': boosted_similarity
                })
        
        # Sort by boosted similarity
        candidates.sort(key=lambda x: x['boosted_similarity'], reverse=True)
        
        # Truncate to max_context_chars (keeps highest-scoring docs)
        total_chars = 0
        final_docs = []
        
        for doc in candidates:
            doc_length = len(doc['text'])
            if total_chars + doc_length > max_context_chars:
                # Try to include partial doc if it fits
                remaining = max_context_chars - total_chars
                if remaining > 100:  # Only if meaningful chunk remains
                    doc_partial = {**doc, 'text': doc['text'][:remaining] + '...'}
                    final_docs.append(doc_partial)
                break
            
            final_docs.append(doc)
            total_chars += doc_length
        
        logger.info(f" Hybrid retrieval: {len(final_docs)} docs ({total_chars} chars, limit: {max_context_chars})")
        
        return final_docs, timing
    
    def _extract_template_fields(self, docs: list, pattern: Dict) -> Dict:
        """
        Extract structured fields from retrieved docs for template filling
        
        Args:
            docs: Retrieved document dicts
            pattern: Pattern configuration dict
            
        Returns:
            dict: Extracted fields for template (e.g., {department, hours, ...})
        """
        extracted = {}
        pattern_name = pattern.get("name", "")
        
        # Combine all doc texts for extraction
        combined_text = "\n".join([doc.get('text', '') for doc in docs])
        
        # Pattern-specific extraction logic
        if pattern_name == "office_hours":
            # Extract hours pattern (e.g., "Monday-Friday 9:00-15:00")
            hours_match = re.search(r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*[-–]\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?\s*\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})', combined_text)
            if hours_match:
                extracted["hours"] = hours_match.group(1)
            
            # Find department mention
            dept_keywords = ["admissions", "academic", "student services", "registry", "enrollment"]
            for dept in dept_keywords:
                if dept in combined_text.lower():
                    extracted["department"] = dept.title()
                    break
            
            # Extract additional info (walk-in, appointment, etc.)
            if "walk-in" in combined_text.lower():
                walk_in_match = re.search(r'walk-in.*?(\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})', combined_text, re.IGNORECASE)
                if walk_in_match:
                    extracted["additional_info"] = f"Walk-in hours: {walk_in_match.group(1)}"
        
        elif pattern_name == "admission_requirements":
            # ENHANCED: More specific extraction for admission requirements
            # Extract program mention (with context)
            programs = {
                "master": ["master", "master's", "msc", "m.sc"],
                "bachelor": ["bachelor", "bachelor's", "bsc", "b.sc"],
                "phd": ["phd", "doctoral", "doctorate"],
                "mba": ["mba", "business administration"]
            }
            for prog_key, prog_variants in programs.items():
                if any(variant in combined_text.lower() for variant in prog_variants):
                    extracted["program"] = prog_key.upper() if prog_key == "mba" else prog_key.title()
                    break
            
            # Extract specific requirements with categories
            requirements_found = []
            
            # Check for GPA/grades
            gpa_match = re.search(r'(GPA|grade|average).*?(\d+\.?\d*)', combined_text, re.IGNORECASE)
            if gpa_match:
                requirements_found.append(f"GPA requirement: {gpa_match.group(2)}")
            
            # Check for language requirements
            if re.search(r'(TOEFL|IELTS|language|English|German)', combined_text, re.IGNORECASE):
                lang_match = re.search(r'(TOEFL.*?(?:\d+)|IELTS.*?(?:\d+\.?\d*))', combined_text, re.IGNORECASE)
                if lang_match:
                    requirements_found.append(f"Language: {lang_match.group(1)}")
                else:
                    requirements_found.append("Language proficiency required")
            
            # Check for degree requirements
            degree_match = re.search(r'(bachelor[\'s]?|undergraduate)\s+degree', combined_text, re.IGNORECASE)
            if degree_match:
                requirements_found.append(f"Previous {degree_match.group(1)} degree required")
            
            # Extract bulleted/numbered requirements
            req_lines = [line.strip() for line in combined_text.split('\n') 
                        if line.strip() and (line.strip().startswith(('-', '•', '*')) or re.match(r'^\d+\.', line.strip()))]
            if req_lines:
                # Filter out too-long lines (likely not actual requirements)
                clean_reqs = [r for r in req_lines if len(r) < 150][:3]
                requirements_found.extend(clean_reqs)
            
            if requirements_found:
                extracted["requirements"] = "; ".join(requirements_found[:4])  # Top 4 requirements
            
            # Extract application link/portal
            link_match = re.search(r'(https?://[^\s]+(?:application|apply|admission)[^\s]*)', combined_text, re.IGNORECASE)
            if link_match:
                extracted["application_link"] = link_match.group(1)
        
        elif pattern_name == "contact_info":
            # ENHANCED: More aggressive rule-based extraction for contact info
            # Extract ALL emails (prioritize .uni-hannover.de)
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', combined_text)
            uni_emails = [e for e in emails if 'uni-hannover' in e.lower() or 'leibniz' in e.lower()]
            if uni_emails:
                extracted["email"] = uni_emails[0]  # Prioritize university email
            elif emails:
                extracted["email"] = emails[0]
            
            # Extract ALL phones (prioritize German format)
            phones = re.findall(r'(\+?\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4})', combined_text)
            if phones:
                extracted["phone"] = phones[0]
            
            # Extract office/department names
            dept_patterns = [
                r'(Registrar[\'s]?\s+Office)',
                r'(Student\s+Services)',
                r'(Academic\s+Affairs)',
                r'(Admissions\s+Office)',
                r'(International\s+Office)',
                r'(Central\s+Student\s+Advisory\s+Service)',
            ]
            for pattern_regex in dept_patterns:
                dept_match = re.search(pattern_regex, combined_text, re.IGNORECASE)
                if dept_match:
                    extracted["department"] = dept_match.group(1)
                    break
            
            # Extract office hours if mentioned
            hours_match = re.search(r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Mon|Tue|Wed|Thu|Fri)\s*[-–]\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Mon|Tue|Wed|Thu|Fri)?\s*\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})', combined_text, re.IGNORECASE)
            if hours_match:
                extracted["hours"] = hours_match.group(1)
            
            # Extract building/room numbers
            room_match = re.search(r'((?:Building|Room|Office)\s+[A-Z]?\d+[A-Za-z]?(?:\s*,\s*Room\s+\d+)?)', combined_text, re.IGNORECASE)
            if room_match:
                extracted["location"] = room_match.group(1)
            
            # Build contact summary
            contact_parts = []
            if "department" in extracted:
                contact_parts.append(f"the {extracted['department']}")
            if "email" in extracted:
                contact_parts.append(f"email: {extracted['email']}")
            if "phone" in extracted:
                contact_parts.append(f"phone: {extracted['phone']}")
            if "location" in extracted:
                contact_parts.append(f"location: {extracted['location']}")
            if "hours" in extracted:
                contact_parts.append(f"hours: {extracted['hours']}")
                
            if contact_parts:
                extracted["contact_summary"] = " | ".join(contact_parts)
        
        elif pattern_name == "appointment_scheduling":
            # Extract steps/instructions
            step_lines = [line.strip() for line in combined_text.split('\n') 
                         if line.strip() and (re.match(r'^\d+\.', line.strip()) or 'step' in line.lower())]
            if step_lines:
                extracted["steps"] = " ".join(step_lines[:3])
        
        return extracted
    
    def _build_hybrid_prompt(self, query: str, extracted_info: Dict, pattern: Dict, context: Optional[Dict], retrieved_docs: list, history_context: str = "") -> str:
        """
        Build SHORT prompt with template-based structure (reduces tokens by 50-70%)
        
        Args:
            query: User query
            extracted_info: Extracted template fields
            pattern: Pattern configuration
            context: Structured context from intent parser
            retrieved_docs: Retrieved documents
            
        Returns:
            str: Compact Gemini prompt
        """
        template = pattern["response_template"]
        pattern_name = pattern["name"]
        
        # Build compact context summary (only essential info)
        context_summary = ""
        if retrieved_docs:
            # Take first 300 chars of top doc (increased for better context)
            top_doc_text = retrieved_docs[0].get('text', '')[:300]
            context_summary = f"\nKey information: {top_doc_text}..."
        
        # ENHANCED: Pattern-specific prompt templates for better accuracy
        language = os.getenv("RAG_LANGUAGE", "english").lower()
        is_german = language == "german"
        if pattern_name == "contact_info":
            # Force direct extraction-based response
            if extracted_info:
                prompt = f"""You are Lexi, a helpful university assistant for Daytona.

{"You must respond ONLY in GERMAN, regardless of the input language." if is_german else "You must respond ONLY in ENGLISH, regardless of the input language."}

CONVERSATION HISTORY:
{history_context if history_context else "(No previous conversation)"}

Give them the contact information directly - email, phone, location, hours, whatever's relevant. Be direct and concise. Maximum 450-500 characters.

QUESTION: {query}

EXTRACTED CONTACT INFO: {json.dumps(extracted_info, ensure_ascii=False)}
CONTEXT: {context_summary}

ANSWER:"""
            else:
                # Fallback if extraction failed
                prompt = f"""You are Lexi, a helpful university assistant for Daytona.

{"You must respond ONLY in GERMAN, regardless of the input language." if is_german else "You must respond ONLY in ENGLISH, regardless of the input language."}

CONVERSATION HISTORY:
{history_context if history_context else "(No previous conversation)"}

Give them the contact information from what you know. Be specific and direct. Maximum 450-500 characters.

QUESTION: {query}
CONTEXT: {context_summary}

ANSWER:"""
        
        elif pattern_name == "ambiguous_question":
            # Return fixed response for ambiguous questions - no retrieval needed
            return pattern["response_template"]
        
        elif pattern_name == "admission_requirements":
            # Force specific, structured requirements response
            if extracted_info:
                prompt = f"""You are Lexi, a helpful university assistant for Daytona.

{"You must respond ONLY in GERMAN, regardless of the input language." if is_german else "You must respond ONLY in ENGLISH, regardless of the input language."}

CONVERSATION HISTORY:
{history_context if history_context else "(No previous conversation)"}

List out the admission requirements clearly. Mention the program if you know it. Be direct and structured. Maximum 450-500 characters. Only suggest contacting admissions if you're missing key info.

QUESTION: {query}

EXTRACTED REQUIREMENTS: {json.dumps(extracted_info, ensure_ascii=False)}
CONTEXT: {context_summary}

ANSWER:"""
            else:
                prompt = f"""You are Lexi, a helpful university assistant for Daytona.

{"You must respond ONLY in GERMAN, regardless of the input language." if is_german else "You must respond ONLY in ENGLISH, regardless of the input language."}

CONVERSATION HISTORY:
{history_context if history_context else "(No previous conversation)"}

Give them the admission requirements based on what you know. Be direct and clear. Maximum 450-500 characters.

QUESTION: {query}
CONTEXT: {context_summary}

ANSWER:"""
        
        else:
            # Generic template for other patterns
            prompt = f"""You are Lexi, a helpful university assistant for Daytona.

{"You must respond ONLY in GERMAN, regardless of the input language." if is_german else "You must respond ONLY in ENGLISH, regardless of the input language."}

CONVERSATION HISTORY:
{history_context if history_context else "(No previous conversation)"}

Be concise and to the point. Maximum 450-500 characters. If you don't have complete info, suggest who they should contact.

QUESTION: {query}

PATTERN TYPE: {pattern_name.replace('_', ' ')}
DETAILS: {json.dumps(extracted_info, ensure_ascii=False)}
CONTEXT: {context_summary}

ANSWER:"""
        
        return prompt
    
    def _construct_enriched_search_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        history_context: Optional[str] = None,
        relevant_docs: Optional[List[Dict]] = None
    ) -> str:
        """
        Construct a context-enriched search query for web search.
        
        Enriches the user query with:
        - Organization name and location (default context)
        - Key entities from context (if available)
        - Disambiguated query from history (if pronouns detected)
        
        Args:
            query: Original user query
            context: Optional context from intent service
            history_context: Optional conversation history
            relevant_docs: Retrieved local documents (for entity extraction)
            
        Returns:
            Enriched search query string
        """
        # Start with the original query
        enriched_parts = [query]
        
        # Use extracted_meaning or user_goal if available (these are often disambiguated)
        if context:
            if 'extracted_meaning' in context and context['extracted_meaning']:
                # Use the disambiguated version if it's different
                extracted = context['extracted_meaning']
                if extracted.lower() != query.lower():
                    enriched_parts[0] = extracted  # Replace with disambiguated version
            
            # Add key entities (e.g., "program", "department", "location")
            if 'key_entities' in context:
                entities = context['key_entities']
                # Add entity values that are meaningful (not generic)
                for key, value in entities.items():
                    if isinstance(value, str) and len(value) > 2:
                        # Skip generic values like "yes", "no", etc.
                        if value.lower() not in ['yes', 'no', 'true', 'false', 'ok']:
                            enriched_parts.append(value)
        
        # Always add organization context for specificity
        enriched_parts.append(self.config.organization_name)
        enriched_parts.append(self.config.organization_location)
        
        # Construct final query
        enriched_query = " ".join(enriched_parts)
        
        logger.debug(f" Enriched search query: '{query}' -> '{enriched_query}'")
        return enriched_query
    
    async def _fetch_web_results(self, query: str) -> str:
        """
        Fetch top search results from Google Programmable Search Engine.
        Returns formatted string of snippets or empty string on failure.
        """
        if not self.config.google_search_api_key or not self.config.google_cse_id:
            logger.warning(" Web search enabled but credentials missing")
            return ""
            
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.config.google_search_api_key,
                "cx": self.config.google_cse_id,
                "q": query,
                "num": 3,  # Top 3 results
                "safe": "active"
            }
            
            # Strict 2s timeout for low latency
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(search_url, params=params)
                
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                
                if not items:
                    return ""
                    
                formatted_results = ["--- WEB SEARCH RESULTS ---"]
                for item in items:
                    title = item.get("title", "No Title")
                    snippet = item.get("snippet", "No Snippet")
                    link = item.get("link", "")
                    formatted_results.append(f"Source: {title} ({link})\nContent: {snippet}")
                    
                return "\n\n".join(formatted_results)
            else:
                logger.error(f" Google Search API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f" Web search failed: {e}")
            return ""
    
