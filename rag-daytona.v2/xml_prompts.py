"""
XML Prompt System for RAG-XML Architecture (Daytona V2).
Minified version for ultra-low latency.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import datetime
import re

def escape_xml(text: str) -> str:
    """Safely escape text for XML inclusion."""
    if not text: return ""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')

class XMLPromptManager:
    """
    Manages minified XML construction to reduce token count and TTFC.
    """

    @staticmethod
    def render_zone_a(agent_name: str, persona_anchor: str, constraints: List[str]) -> str:
        constraints_xml = "".join([f"<li>{escape_xml(c)}</li>" for c in constraints])
        
        # High-density examples with acknowledgments
        shots = (
            '<ex id="1"><q>How do I install Daytona?</q><a>Understood. You can install it by running `curl -sfL https://download.daytona.io/daytona/install.sh | sudo bash`. Then, just start the server with `daytona server`.</a></ex>'
            '<ex id="2"><q>Was kostet Daytona?</q><a>Ich verstehe. Daytona ist Open Source und komplett kostenlos für die lokale Nutzung. Enterprise-Lizenzen gibt es für Teams, aber für dich als Einzelentwickler ist es frei.</a></ex>'
            '<ex id="3"><q>AWS support?</q><a>Right, it definitely supports AWS. You just need to run `daytona target set` to configure your AWS, Azure, or DigitalOcean targets.</a></ex>'
        )
        
        rules = (
            "Acknowledge naturally first (e.g., 'Got it', 'Right'). "
            "Then provide a direct, high-utility, human-sounding response as a helpful senior dev. "
            "Match the user's language naturally. Brief: 1-3 sentences unless complex."
        )
        
        return f'<sys><id><name>{escape_xml(agent_name)}</name><persona>{escape_xml(persona_anchor)}</persona><rules>{rules}</rules><cons><ul>{constraints_xml}</ul></cons>{shots}</id></sys>'

    @staticmethod
    def render_zone_b(hive_mind_state: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
        now = datetime.datetime.now().isoformat()
        
        insights = ""
        if "insights" in hive_mind_state:
            for topic, insight in hive_mind_state["insights"].items():
                insights += f'<insight t="{escape_xml(topic)}">{escape_xml(insight)}</insight>'
        
        profile = "".join([f'<{escape_xml(k)}>{escape_xml(str(v))}</{escape_xml(k)}>' for k,v in user_profile.items()])

        return f'<mem><hive ts="{now}">{insights}</hive><store><prof>{profile}</prof></store></mem>'

    @staticmethod
    def render_zone_c(history: List[Dict[str, str]], retrieved_context: List[Dict[str, Any]], user_input: str) -> str:
        hist = "".join([f'<t s="{i}" r="{escape_xml(t.get("role","u"))}">{escape_xml(t.get("content",""))}</t>' for i, t in enumerate(history)])

        docs = ""
        for d in retrieved_context:
            src = d.get("metadata", {}).get("source_type", "f")
            id_ = str(d.get("metadata", {}).get("id", "unk"))
            cont = d.get("text", d.get("content", d.get("page_content", "")))[:1200]
            docs += f'<src t="{escape_xml(src)}" i="{escape_xml(id_)}">{escape_xml(cont)}</src>'

        return f'<ctxt><hist>{hist}</hist><docs>{docs}</docs><turn><in>{escape_xml(user_input)}</in><resp>'

    @classmethod
    def assemble_static_prompt(cls, agent_name: str, persona: str, constraints: List[str], hive_mind: Dict, profile: Dict) -> str:
        return cls.render_zone_a(agent_name, persona, constraints) + cls.render_zone_b(hive_mind, profile)
