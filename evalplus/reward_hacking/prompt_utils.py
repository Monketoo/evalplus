"""Utilities for cleaning and processing prompts."""

import re


def remove_examples_from_prompt(prompt: str) -> str:
    """
    Remove existing test examples from prompt to have full control.
    Removes docstring examples (>>> style) while keeping the rest.
    
    Args:
        prompt: Original prompt with potential examples
        
    Returns:
        Cleaned prompt without examples
    """
    lines = prompt.split('\n')
    cleaned = []
    in_docstring = False
    skip_example = False
    quote_style = None
    
    for line in lines:
        # Detect docstring start/end
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                quote_style = '"""' if '"""' in line else "'''"
                cleaned.append(line)
                # Check if single-line docstring
                if line.count(quote_style) >= 2:
                    in_docstring = False
            elif quote_style in line:
                in_docstring = False
                cleaned.append(line)
            else:
                cleaned.append(line)
            continue
        
        # Inside docstring - check for examples
        if in_docstring:
            stripped = line.strip()
            # >>> indicates example
            if stripped.startswith('>>>'):
                skip_example = True
                continue
            # Example output line (indented after >>>)
            elif skip_example and stripped and not stripped.startswith('>>>'):
                # This is the output of the example
                continue
            else:
                skip_example = False
                cleaned.append(line)
        else:
            cleaned.append(line)
    
    return '\n'.join(cleaned)