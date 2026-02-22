"""
CodeDenoise (Plan §2.6): Syntax–Semantik-Filter für Rohdaten.
Stage-2-Stream in prepare_data nutzt clean_syntax_semantics/code_denoise_filter
bei run_code_denoise in config_data.yaml.
"""

import ast

def clean_syntax_semantics(code: str, docstring: str) -> bool:
    """True wenn Code parsbare AST hat; bei docstring mind. eine FunctionDef."""
    try:
        tree = ast.parse(code)
        # Verify function defs exist if a docstring is provided
        if docstring:
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            if not functions:
                return False
        return True
    except SyntaxError:
        return False

def code_denoise_filter(dataset_stream):
    """Generator applying CodeDenoise onto a streaming dataset."""
    for item in dataset_stream:
        if clean_syntax_semantics(item.get("code", ""), item.get("docstring", "")):
            yield item

if __name__ == "__main__":
    valid_code = "def add(a, b):\n    '''Adds two nums'''\n    return a + b"
    invalid_code = "def add(a, b) return a + b" # Syntax Error
    print(f"Valid code passes: {clean_syntax_semantics(valid_code, 'Adds two nums')}")
    print(f"Invalid code passes: {clean_syntax_semantics(invalid_code, '')}")
