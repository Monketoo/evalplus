def strip_think_tags(code: str) -> str:
    """Remove <think> tags and content from code if accidentally included."""
    if '<think>' not in code:
        return code
    
    # Remove everything between <think> and </think>
    while '<think>' in code:
        start = code.find('<think>')
        end = code.find('</think>')
        if end == -1:
            # Unclosed tag, remove everything after <think>
            code = code[:start]
            break
        code = code[:start] + code[end + 8:]  # 8 = len('</think>')
    
    return code.strip()