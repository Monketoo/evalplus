from typing import List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PromptStyle(Enum):
    """Different ways to present test cases in prompts."""
    DOCSTRING = "docstring"      # Inside function docstring as examples
    ASSERT = "assert"            # As assert statements after signature
    COMMENT = "comment"          # As commented examples
    NATURAL = "natural"          # Natural language description
    NONE = "none"                # No test cases shown


@dataclass
class PromptConfig:
    """Configuration for prompt building."""
    show_tests: bool = True
    style: PromptStyle = PromptStyle.ASSERT
    max_tests_shown: Optional[int] = None  # Limit number of tests shown
    add_instruction: Optional[str] = None  # Additional instruction to add
    misleading_instruction: Optional[str] = None  # Misleading hint
    show_expected_output: bool = True  # Whether to show expected outputs


class RewardHackingPromptBuilder:
    """Builds prompts with controlled test case visibility for reward hacking experiments."""
    
    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize prompt builder.
        
        Args:
            config: Configuration for how to build prompts
        """
        self.config = config or PromptConfig()
    
    def build_prompt(
        self,
        task_prompt: str,
        entry_point: str,
        visible_tests: List[Tuple[Tuple, Any]],  # List of (input, output) pairs
        task_id: str = "",
    ) -> str:
        """
        Build a prompt with visible test cases.
        
        Args:
            task_prompt: Original task prompt (function signature + docstring)
            entry_point: Function name
            visible_tests: List of (input_tuple, expected_output) pairs to show
            task_id: Task identifier (optional, for logging)
            
        Returns:
            Complete prompt string to send to model
        """
        # Start with base prompt
        prompt = task_prompt.rstrip()
        
        # Add visible tests if requested
        if self.config.show_tests and visible_tests:
            test_str = self._format_tests(
                entry_point,
                visible_tests,
                self.config.style,
                self.config.max_tests_shown,
                self.config.show_expected_output
            )
            prompt = self._inject_tests(prompt, test_str, entry_point)
        
        # Add instructions
        if self.config.add_instruction:
            prompt += f"\n\n# Note: {self.config.add_instruction}"
        
        if self.config.misleading_instruction:
            prompt += f"\n\n# Important: {self.config.misleading_instruction}"
        
        return prompt
    
    def _format_tests(
        self,
        entry_point: str,
        tests: List[Tuple[Tuple, Any]],
        style: PromptStyle,
        max_tests: Optional[int],
        show_output: bool
    ) -> str:
        """Format test cases according to specified style."""
        if max_tests:
            tests = tests[:max_tests]
        
        if style == PromptStyle.ASSERT:
            return self._format_as_asserts(entry_point, tests, show_output)
        elif style == PromptStyle.DOCSTRING:
            return self._format_as_docstring(entry_point, tests, show_output)
        elif style == PromptStyle.COMMENT:
            return self._format_as_comments(entry_point, tests, show_output)
        elif style == PromptStyle.NATURAL:
            return self._format_as_natural(entry_point, tests, show_output)
        else:
            return ""
    
    def _format_as_asserts(
        self,
        entry_point: str,
        tests: List[Tuple[Tuple, Any]],
        show_output: bool
    ) -> str:
        """Format as assert statements."""
        lines = []
        for inp, out in tests:
            args_str = ", ".join(repr(arg) for arg in inp)
            if show_output:
                lines.append(f"assert {entry_point}({args_str}) == {repr(out)}")
            else:
                lines.append(f"{entry_point}({args_str})  # Should pass")
        return "\n".join(lines)
    
    def _format_as_docstring(
        self,
        entry_point: str,
        tests: List[Tuple[Tuple, Any]],
        show_output: bool
    ) -> str:
        """Format as docstring examples (>>> style)."""
        lines = []
        for inp, out in tests:
            args_str = ", ".join(repr(arg) for arg in inp)
            lines.append(f"    >>> {entry_point}({args_str})")
            if show_output:
                lines.append(f"    {repr(out)}")
        return "\n".join(lines)
    
    def _format_as_comments(
        self,
        entry_point: str,
        tests: List[Tuple[Tuple, Any]],
        show_output: bool
    ) -> str:
        """Format as commented examples."""
        lines = []
        for inp, out in tests:
            args_str = ", ".join(repr(arg) for arg in inp)
            if show_output:
                lines.append(f"# Example: {entry_point}({args_str}) -> {repr(out)}")
            else:
                lines.append(f"# Example: {entry_point}({args_str})")
        return "\n".join(lines)
    
    def _format_as_natural(
        self,
        entry_point: str,
        tests: List[Tuple[Tuple, Any]],
        show_output: bool
    ) -> str:
        """Format as natural language description."""
        lines = ["# The function should handle these cases:"]
        for i, (inp, out) in enumerate(tests, 1):
            args_str = ", ".join(repr(arg) for arg in inp)
            if show_output:
                lines.append(f"# {i}. When called with {args_str}, it returns {repr(out)}")
            else:
                lines.append(f"# {i}. When called with {args_str}")
        return "\n".join(lines)
    
    def _inject_tests(self, prompt: str, test_str: str, entry_point: str) -> str:
        """Inject test string into prompt at appropriate location."""
        if self.config.style == PromptStyle.DOCSTRING:
            # Try to inject into docstring
            return self._inject_into_docstring(prompt, test_str, entry_point)
        else:
            # Append after prompt
            return f"{prompt}\n\n{test_str}\n"
    
    def _inject_into_docstring(self, prompt: str, test_str: str, entry_point: str) -> str:
        """Inject examples into the function's docstring."""
        lines = prompt.split("\n")
        
        # Find docstring end
        in_docstring = False
        docstring_end_idx = -1
        quote_style = None
        
        for i, line in enumerate(lines):
            if not in_docstring and ('"""' in line or "'''" in line):
                in_docstring = True
                quote_style = '"""' if '"""' in line else "'''"
                # Check if single-line docstring
                if line.count(quote_style) >= 2:
                    docstring_end_idx = i
                    break
            elif in_docstring and quote_style in line:
                docstring_end_idx = i
                break
        
        if docstring_end_idx > 0:
            # Insert before closing quotes
            lines.insert(docstring_end_idx, test_str)
            return "\n".join(lines)
        else:
            # Couldn't find docstring, append after
            return f"{prompt}\n\n{test_str}\n"


# Convenience functions for common scenarios

def build_standard_prompt(
    task_prompt: str,
    entry_point: str,
    visible_tests: List[Tuple[Tuple, Any]]
) -> str:
    """Build standard prompt with assert-style tests."""
    builder = RewardHackingPromptBuilder(
        PromptConfig(show_tests=True, style=PromptStyle.ASSERT)
    )
    return builder.build_prompt(task_prompt, entry_point, visible_tests)


def build_misleading_prompt(
    task_prompt: str,
    entry_point: str,
    visible_tests: List[Tuple[Tuple, Any]],
    misleading_hint: str
) -> str:
    """Build prompt with misleading instruction."""
    builder = RewardHackingPromptBuilder(
        PromptConfig(
            show_tests=True,
            style=PromptStyle.ASSERT,
            misleading_instruction=misleading_hint
        )
    )
    return builder.build_prompt(task_prompt, entry_point, visible_tests)


def build_minimal_prompt(
    task_prompt: str,
    entry_point: str,
    visible_tests: List[Tuple[Tuple, Any]]
) -> str:
    """Build prompt without showing expected outputs (harder)."""
    builder = RewardHackingPromptBuilder(
        PromptConfig(
            show_tests=True,
            style=PromptStyle.ASSERT,
            show_expected_output=False
        )
    )
    return builder.build_prompt(task_prompt, entry_point, visible_tests)


def build_no_test_prompt(task_prompt: str, entry_point: str) -> str:
    """Build prompt without any test cases (baseline)."""
    builder = RewardHackingPromptBuilder(
        PromptConfig(show_tests=False)
    )
    return builder.build_prompt(task_prompt, entry_point, [])