import os
import re

class PromptLoader:
    """Loads and manages XML-based prompts."""
    
    def __init__(self, prompt_dir: str = None):
        if not prompt_dir:
            # Default to current dir's /xml folder
            base = os.path.dirname(os.path.abspath(__file__))
            prompt_dir = os.path.join(base, "xml")
        self.prompt_dir = prompt_dir
        self._cache = {}

    def get_prompt_content(self, prompt_name: str) -> str:
        """Reads the raw XML content of a prompt file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        path = os.path.join(self.prompt_dir, f"{prompt_name}.xml")
        if not os.path.exists(path):
            return ""
            
        with open(path, "r") as f:
            content = f.read()
            self._cache[prompt_name] = content
            return content

    def load_planner_prompt(self) -> str:
        """Returns the content for the Strategic Planner."""
        return self.get_prompt_content("planner_prompt")

    def load_entity_prompt(self) -> str:
        """Returns the content for Entity Extraction."""
        return self.get_prompt_content("entity_prompt")

    def load_response_generator(self) -> str:
        """Returns the content for the Final BLAIQ Persona."""
        return self.get_prompt_content("response_generator")

    def load_template(self, template_name: str) -> str:
        """Loads a specific formatting template from the templates/ directory."""
        path = os.path.join(self.prompt_dir, "templates", f"{template_name.lower()}_formatter.xml")
        if not os.path.exists(path):
            return ""
            
        with open(path, "r") as f:
            return f.read()
