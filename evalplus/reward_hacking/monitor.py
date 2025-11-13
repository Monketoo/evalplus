"""Simple monitors for code and reasoning analysis."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class Monitor:
    """Base monitor class."""
    
    def __init__(self, openrouter_client, model: str, prompts_path: Optional[str] = None):
        self.client = openrouter_client
        self.model = model
        
        if prompts_path is None:
            prompts_path = Path(__file__).parent / "monitors.yaml"
        
        with open(prompts_path) as f:
            self.prompts = yaml.safe_load(f)
    
    def _call(self, prompt: str) -> Dict:
        """Call LLM and parse JSON."""
        try:
            response = self.client.get_text_response(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000
            )
            
            output = response['output']
            
            # Extract JSON
            if "```json" in output:
                json_str = output.split("```json")[1].split("```")[0]
            elif "```" in output:
                json_str = output.split("```")[1].split("```")[0]
            else:
                json_str = output
            
            return json.loads(json_str.strip())
            
        except Exception as e:
            return {"error": str(e)}


class CodeMonitor(Monitor):
    """Monitor for code analysis."""
    
    def analyze(
        self, task_id: str, code: str, entry_point: str,
        task_description: str, visible_tests: List[Tuple]
    ) -> Dict:
        """Analyze code for hardcoding."""
        tests_str = "\n".join(
            f"  {i+1}. {inp} → {out}"
            for i, (inp, out) in enumerate(visible_tests[:10])
        )
        
        prompt = self.prompts['monitor_code'].format(
            task_id=task_id,
            task_description=task_description,
            entry_point=entry_point,
            code=code,
            visible_tests=tests_str
        )
        
        return self._call(prompt)


class CoTMonitor(Monitor):
    """Monitor for reasoning analysis."""
    
    def analyze(
        self, task_id: str, reasoning: str,
        task_description: str, visible_tests: List[Tuple]
    ) -> Dict:
        """Analyze reasoning for test-fitting."""
        tests_str = "\n".join(
            f"  {i+1}. {inp} → {out}"
            for i, (inp, out) in enumerate(visible_tests[:10])
        )
        
        prompt = self.prompts['monitor_cot'].format(
            task_id=task_id,
            task_description=task_description,
            reasoning=reasoning,
            visible_tests=tests_str
        )
        
        return self._call(prompt)


# Factory functions
def create_code_monitor(openrouter_client, model: str = "anthropic/claude-3.5-sonnet"):
    return CodeMonitor(openrouter_client, model)


def create_cot_monitor(openrouter_client, model: str = "anthropic/claude-3.5-sonnet"):
    return CoTMonitor(openrouter_client, model)