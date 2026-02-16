import json
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SessionAnalytics:
    """
    DavinciAI Sentiment & Reasoning Pipeline.
    Implements a 5-phase analysis for post-session business intelligence.
    """

    def __init__(self, llm_provider: Any, model_name: str = "qwen/qwen3-32b"):
        self.llm_provider = llm_provider
        self.model_name = model_name

    def format_transcript(self, raw_logs: List[Dict[str, Any]]) -> str:
        """
        Phase 1: Data Ingestion & Serialization
        Normalizes raw logs into a clean, context-aware format including latency.
        """
        formatted_transcript = ""
        prev_timestamp = None
        
        for index, log in enumerate(raw_logs):
            curr_ts = log.get('timestamp', time.time())
            latency = 0
            if prev_timestamp is not None:
                latency = round(curr_ts - prev_timestamp, 2)
            
            role = log.get('role', 'unknown').capitalize()
            content = log.get('content', '') or log.get('text', '')
            
            formatted_transcript += f"[Turn {index+1}] [{role}] (Latency: {latency}s): {content}\n"
            prev_timestamp = curr_ts
            
        return formatted_transcript

    async def analyze_session(self, raw_logs: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        """
        Executes the full analytics pipeline.
        """
        start_time = time.time()
        
        # 1. Serialize
        transcript = self.format_transcript(raw_logs)
        
        # 2. Reasoning (LLM)
        reasoning_output = await self._run_reasoning_engine(transcript, session_id)
        
        # 3. Calculate Deterministic Metrics
        davinci_metrics = self._calculate_davinci_metrics(reasoning_output.get('turns', []), raw_logs)
        
        # 4. Classify Business Signals
        business_signals = self._classify_business_signals(reasoning_output.get('session_summary', {}), davinci_metrics)
        
        # 5. Extract Final Report for Orchestrator/Backend
        report = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": davinci_metrics,
            "business_signals": business_signals,
            "analysis": reasoning_output.get('session_summary', {}),
            "distilled_knowledge": reasoning_output.get('distilled_knowledge', []),
            "processing_time": round(time.time() - start_time, 2)
        }
        
        logger.info(f"Analytics complete for {session_id} in {report['processing_time']}s")
        return report

    async def _run_reasoning_engine(self, transcript: str, session_id: str) -> Dict[str, Any]:
        """
        Phase 2: Reasoning Engine (LLM Interaction)
        Uses CoT prompting to extract structured sentiment and intent.
        """
        system_prompt = """
You are the DavinciAI Sentiment Engine. Your goal is to analyze the provided [Transcript] using Chain-of-Thought reasoning.

## ANALYSIS STEPS (Perform for EVERY User Turn)
1. **Context Check:** Look at the Agent's *previous* message. Did the Agent answer the question?
2. **Implicit Signal Detection:**
   - Did the user correct the agent? (Signal: Negative)
   - Did the user ignore the Agent's suggestion? (Signal: Disengagement)
   - Did the user thank the Agent specifically for *solving* the problem? (Signal: Resolution)
3. **Aspect Separation:** Distinguish between the User's mood vs. the Agent's performance.
4. **Knowledge Distillation (New):** Identify specific technical problems that were successfully resolved. Extract them as standalone knowledge units (Issue/Solution pairs) for the team's Hive Mind.

## OUTPUT SCHEMA (JSON)
{
  "session_id": "string",
  "turns": [
    {
      "turn_number": int,
      "speaker": "user",
      "thought_trace": "Detailed thought process about sentiment and intent",
      "sentiment_score": float (-1.0 to 1.0),
      "emotional_labels": ["label1", "label2"],
      "intent_type": "Complaint | Inquiry | Purchase | Tech_Support"
    }
  ],
  "session_summary": {
    "overall_sentiment": float,
    "resolution_status": "Resolved | Unresolved | Escalated",
    "customer_pain_points": ["point1", "point2"]
  },
  "distilled_knowledge": [
    {
      "issue": "A concise description of the specific problem solved",
      "solution": "The definitive technical answer or steps to resolve it",
      "category": "e.g. installation, shell_config, pricing, sdk_usage",
      "reliability_score": float (0.0 to 1.0)
    }
  ]
}
Return ONLY valid JSON.
"""
        user_prompt = f"Analyze the following transcript for session {session_id}:\n\n{transcript}"
        
        try:
            # Note: We use the Qwen reasoning model as requested
            response = await self.llm_provider.generate_messages(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                response_format={"type": "json_object"}
            )
            return json.loads(response)
        except Exception as e:
            logger.error(f"Reasoning engine failed: {e}")
            return {"turns": [], "session_summary": {"overall_sentiment": 0, "resolution_status": "Unknown", "customer_pain_points": []}}

    def _calculate_davinci_metrics(self, analyzed_turns: List[Dict[str, Any]], raw_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Phase 3: Davinci Post-Processing Logic
        Deterministic metrics like Frustration Velocity and Agent IQ.
        """
        if not analyzed_turns:
            return {"frustration_velocity": "STABLE", "agent_iq": 1.0, "avg_sentiment": 0.0}
            
        scores = [t.get('sentiment_score', 0) for t in analyzed_turns]
        
        # Velocity
        start_avg = sum(scores[:2]) / len(scores[:2]) if scores else 0
        end_avg = sum(scores[-2:]) / len(scores[-2:]) if scores else 0
        velocity_val = end_avg - start_avg
        
        velocity = "STABLE"
        if velocity_val < -0.4: velocity = "CRITICAL_DEGRADATION"
        elif velocity_val > 0.4: velocity = "SUCCESSFUL_RECOVERY"
        
        # Agent IQ (Correction Rate)
        correction_keywords = ["no", "not what i meant", "wrong", "listen", "again", "stop"]
        corrections = 0
        for turn in analyzed_turns:
            trace = turn.get('thought_trace', '').lower()
            if any(word in trace for word in correction_keywords):
                corrections += 1
        
        agent_iq = 1.0 - (corrections / len(analyzed_turns)) if analyzed_turns else 1.0
        
        return {
            "frustration_velocity": velocity,
            "agent_iq": round(agent_iq, 2),
            "avg_sentiment": round(sum(scores) / len(scores), 2) if scores else 0,
            "correction_count": corrections
        }

    def _classify_business_signals(self, summary: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 4: Predictive Classification
        Convert math into business value.
        """
        overall_sentiment = summary.get('overall_sentiment', 0)
        resolution = summary.get('resolution_status', 'Unknown')
        
        is_churn_risk = (overall_sentiment < -0.5 and resolution != "Resolved") or (metrics['frustration_velocity'] == "CRITICAL_DEGRADATION")
        
        # Simple lead detection (this could be improved by context)
        is_hot_lead = False # Placeholder for logic related to intent or keywords
        
        return {
            "is_churn_risk": is_churn_risk,
            "is_hot_lead": is_hot_lead,
            "priority_level": "HIGH" if is_churn_risk else "NORMAL"
        }
