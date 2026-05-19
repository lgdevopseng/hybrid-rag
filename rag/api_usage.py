import sys
from typing import Any


# Formats OpenAI usage metrics for logging.
def log_api_usage(function_name: str, usage: Any, model: str = "") -> None:
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(prompt_details, "cached_tokens", None) if prompt_details else None
    parts = [f"[ai usage] fn={function_name}"]
    if model:
        parts.append(f"model={model}")
    if prompt_tokens is not None:
        parts.append(f"input={prompt_tokens}")
    elif input_tokens is not None:
        parts.append(f"input={input_tokens}")
    if completion_tokens is not None:
        parts.append(f"output={completion_tokens}")
    elif output_tokens is not None:
        parts.append(f"output={output_tokens}")
    if cached_tokens is not None:
        parts.append(f"cached={cached_tokens}")
    if total_tokens is not None:
        parts.append(f"total={total_tokens}")
    print(" ".join(parts), file=sys.stderr)
