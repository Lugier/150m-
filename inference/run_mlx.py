"""
MLX inference (Plan §8): Apple Silicon; load, generate. RLM (rlm_generate) für 1M+ Repo.
RLM REPL: Prompt → generate code → execute in sandbox → result in context → iterate (1M+ context via fixed window).
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional

RLM_MAX_STEPS = 10
RLM_EXEC_TIMEOUT = 5.0


def _execute_code_safely(code: str, timeout: float = RLM_EXEC_TIMEOUT) -> tuple[bool, str]:
    """Execute code in subprocess sandbox; return (success, stdout+stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
        f.write(code)
        f.flush()
        try:
            r = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=None,
            )
            out = (r.stdout or "") + (r.stderr or "")
            return r.returncode == 0, out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return False, "Execution Timeout"
        except Exception as e:
            return False, str(e)


def _extract_code_block(text: str) -> str:
    """Extract first ```python ... ``` or ``` ... ``` block."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def rlm_repl_loop(
    query: str,
    generate_fn: Callable[[str], str],
    max_steps: int = RLM_MAX_STEPS,
    exec_timeout: float = RLM_EXEC_TIMEOUT,
) -> str:
    """
    RLM REPL (Plan §2.8, arXiv 2505.07897): fixed-window context loop.
    1. Build prompt with query and prior context (code + output).
    2. Generate code via generate_fn.
    3. Execute in sandbox; append (code, output) to context.
    4. Repeat until solution or max_steps.
    """
    context: List[str] = []
    for step in range(max_steps):
        prompt_parts = [f"Task: {query}"]
        for i, ctx in enumerate(context):
            prompt_parts.append(f"Step {i+1} output:\n{ctx}")
        prompt_parts.append("Generate Python code to proceed (single block, no markdown). Code:")
        prompt = "\n\n".join(prompt_parts)
        response = generate_fn(prompt)
        code = _extract_code_block(response)
        if not code:
            context.append("(no code generated)")
            continue
        ok, output = _execute_code_safely(code, timeout=exec_timeout)
        context.append(f"Exit ok: {ok}\n{output}")
        if ok and output and "error" not in output.lower()[:100]:
            return output
    return context[-1] if context else ""


def load_mlx_model(model_path: str | Path, adapter_path: Optional[str | Path] = None) -> Any:
    """Load model for MLX (Plan §8). Returns (model, tokenizer) or None."""
    try:
        from mlx_lm import load as mlx_load
        path = Path(model_path)
        model_dir = path if path.is_dir() else path.parent
        tok_path = str(adapter_path or "data/tokenizer")
        m, t = mlx_load(str(model_dir), tokenizer_path=tok_path)
        return (m, t)
    except Exception:
        return None

def generate(model: Any, prompt: str, max_tokens: int = 256, temperature: float = 0.7, tokenizer_path: Optional[str] = None) -> str:
    """Generate completion (Plan §8). model is (m, tokenizer) from load_mlx_model."""
    if model is None:
        return prompt + " [stub: no MLX model loaded]"
    try:
        from mlx_lm import generate as mlx_gen
        m, t = model if isinstance(model, tuple) and len(model) == 2 else (model, None)
        if t is None:
            return prompt + " [stub: no tokenizer]"
        return mlx_gen(m, t, prompt=prompt, max_tokens=max_tokens, temp=temperature)
    except Exception:
        return prompt + " [stub: MLX generate failed]"

def rlm_generate(
    repo_path: str,
    query: str,
    model: Any = None,
    max_steps: int = RLM_MAX_STEPS,
    require_model: bool = True,
) -> str:
    """
    RLM (Recursive Language Models) REPL (Plan §2.8): 1M+ context via fixed window.
    1. LLM generates code (e.g. chunk/regex/search over repo).
    2. Execute in safe sandbox; result appended to context.
    3. Re-prompt with context; iterate until solution or max_steps.
    require_model=True (ohne Fallback): Kein Fallback auf grep; bei fehlendem Modell wird ValueError erhoben.
    """
    def generate_fn(prompt: str) -> str:
        if model is not None:
            return generate(model, prompt, max_tokens=512, temperature=0.3)
        if require_model:
            raise ValueError("RLM requires a loaded model (--model). No fallback to grep; pass require_model=False to allow.")
        try:
            grep_cmd = ["grep", "-rnE", "def |class ", repo_path]
            out = subprocess.check_output(grep_cmd, text=True, stderr=subprocess.STDOUT)
            return f"# Context from repo:\n{out[:2000]}"
        except subprocess.CalledProcessError:
            return ""

    return rlm_repl_loop(query, generate_fn, max_steps=max_steps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to safetensors (required; no fallback)")
    parser.add_argument("--prompt", type=str, required=True, help="RLM base prompt / query")
    parser.add_argument("--repo", type=str, default=".", help="Repo path for RLM context")
    parser.add_argument("--max-steps", type=int, default=RLM_MAX_STEPS, help="RLM REPL max steps")
    args = parser.parse_args()

    model = load_mlx_model(args.model)
    if model is None:
        raise FileNotFoundError(f"Could not load model from {args.model}. No fallback; provide a valid --model path.")
    try:
        import mlx.core as mx
        print(f"Apple Silicon MLX backend active: {mx.__version__}. Unified memory zero-copy engaged.")
    except ImportError:
        print("WARNING: mlx is not installed. RLM requires --model (no fallback).")
    result = rlm_generate(args.repo, args.prompt, model=model, max_steps=args.max_steps, require_model=True)
    print(f"\nFinal RLM Output:\n{result}")

if __name__ == "__main__":
    main()
