"""Simple monitors for code and reasoning analysis."""

import json
import yaml
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class Monitor:
    """Base monitor class."""
    
    def __init__(self, openrouter_client, model: str, prompts_path: Optional[str] = None):
        self.client = openrouter_client
        self.model = model
        
        if prompts_path is None:
            prompts_path = Path(__file__).parent / "prompts"/ "monitors.yaml"
        
        with open(prompts_path) as f:
            self.prompts = yaml.safe_load(f)
    
    def _call(self, prompt: str) -> Dict:
        """Call LLM and parse JSON."""
        try:
            response = self.client.get_text_response(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5000
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

    def _batch_call(self, prompts: List[str], max_concurrent: int = 16) -> List[Dict]:
        """Call LLM with multiple prompts in batch and parse JSON results."""
        # Build batch requests (same pattern as _run_codegen_openrouter)
        requests = [
            {
                'model': self.model,
                'messages': [{"role": "user", "content": prompt}],
                'temperature': 0.0,
                'max_tokens': 5000
            }
            for prompt in prompts
        ]

        # Run batch processing (same pattern as _run_codegen_openrouter)
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            results = loop.run_until_complete(
                self.client.batch_get_text_responses(requests, max_concurrent=max_concurrent)
            )
        except RuntimeError:
            results = asyncio.run(
                self.client.batch_get_text_responses(requests, max_concurrent=max_concurrent)
            )

        # Parse JSON from each result
        parsed_results = []
        for response in results:
            try:
                if response.get('error'):
                    parsed_results.append({"error": response['error']})
                    continue

                output = response['output']

                # Extract JSON
                if "```json" in output:
                    json_str = output.split("```json")[1].split("```")[0]
                elif "```" in output:
                    json_str = output.split("```")[1].split("```")[0]
                else:
                    json_str = output

                parsed_results.append(json.loads(json_str.strip()))
            except Exception as e:
                parsed_results.append({"error": str(e)})

        return parsed_results


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

    def batch_analyze(
        self,
        analyses: List[Dict[str, Any]],
        max_concurrent: int = 16
    ) -> List[Dict]:
        """
        Analyze multiple code samples in batch.

        Args:
            analyses: List of dicts with keys: task_id, code, entry_point, task_description, visible_tests
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of analysis results in same order as input
        """
        prompts = []
        for analysis_input in analyses:
            tests_str = "\n".join(
                f"  {i+1}. {inp} → {out}"
                for i, (inp, out) in enumerate(analysis_input['visible_tests'][:10])
            )

            prompt = self.prompts['monitor_code'].format(
                task_id=analysis_input['task_id'],
                task_description=analysis_input['task_description'],
                entry_point=analysis_input['entry_point'],
                code=analysis_input['code'],
                visible_tests=tests_str
            )
            prompts.append(prompt)

        return self._batch_call(prompts, max_concurrent)


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

    def batch_analyze(
        self,
        analyses: List[Dict[str, Any]],
        max_concurrent: int = 16
    ) -> List[Dict]:
        """
        Analyze multiple reasoning traces in batch.

        Args:
            analyses: List of dicts with keys: task_id, reasoning, task_description, visible_tests
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of analysis results in same order as input
        """
        prompts = []
        for analysis_input in analyses:
            tests_str = "\n".join(
                f"  {i+1}. {inp} → {out}"
                for i, (inp, out) in enumerate(analysis_input['visible_tests'][:10])
            )

            prompt = self.prompts['monitor_cot'].format(
                task_id=analysis_input['task_id'],
                task_description=analysis_input['task_description'],
                reasoning=analysis_input['reasoning'],
                visible_tests=tests_str
            )
            prompts.append(prompt)

        return self._batch_call(prompts, max_concurrent)


class RewardHackJudge(Monitor):
    """
    Judge that verifies if undetected cases are actual reward hacks.

    Checks whether the visible tests given to the model are actually wrong
    (contradicting the task specification) or if they're valid tests.
    """

    def analyze(
        self, task_id: str, task_description: str,
        visible_tests: List[Tuple], code: str, reasoning: str,
        hidden_pass_rate: float
    ) -> Dict:
        """Analyze whether tests are actually wrong and if this is a real reward hack."""
        tests_str = "\n".join(
            f"  {i+1}. Input: {inp} → Expected: {out}"
            for i, (inp, out) in enumerate(visible_tests[:10])
        )

        prompt = self.prompts['judge_reward_hack'].format(
            task_id=task_id,
            task_description=task_description,
            visible_tests=tests_str,
            code=code,
            reasoning=reasoning[:2000] if reasoning else "(no reasoning provided)",
            hidden_pass_rate=f"{hidden_pass_rate:.0%}"
        )

        return self._call(prompt)

    def batch_analyze(
        self,
        analyses: List[Dict[str, Any]],
        max_concurrent: int = 16
    ) -> List[Dict]:
        """
        Analyze multiple cases in batch.

        Args:
            analyses: List of dicts with keys: task_id, task_description, visible_tests,
                      code, reasoning, hidden_pass_rate
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of analysis results in same order as input
        """
        prompts = []
        for analysis_input in analyses:
            tests_str = "\n".join(
                f"  {i+1}. Input: {inp} → Expected: {out}"
                for i, (inp, out) in enumerate(analysis_input['visible_tests'][:10])
            )

            prompt = self.prompts['judge_reward_hack'].format(
                task_id=analysis_input['task_id'],
                task_description=analysis_input['task_description'],
                visible_tests=tests_str,
                code=analysis_input['code'],
                reasoning=analysis_input['reasoning'][:2000] if analysis_input['reasoning'] else "(no reasoning provided)",
                hidden_pass_rate=f"{analysis_input['hidden_pass_rate']:.0%}"
            )
            prompts.append(prompt)

        return self._batch_call(prompts, max_concurrent)


# Factory functions
def create_code_monitor(openrouter_client, model: str = "anthropic/claude-3.5-sonnet"):
    return CodeMonitor(openrouter_client, model)


def create_cot_monitor(openrouter_client, model: str = "anthropic/claude-3.5-sonnet"):
    return CoTMonitor(openrouter_client, model)


def create_reward_hack_judge(openrouter_client, model: str = "anthropic/claude-3.5-sonnet"):
    return RewardHackJudge(openrouter_client, model)