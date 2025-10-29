# Agent Notes

## Instructions
- Follow PEP 8 style guidelines and keep line length at or below 100 characters when editing Python files.
- Prefer explicit relative imports within this repository (e.g., `from .module import name`).
- Include or preserve module and function docstrings when modifying Python code.
- Ensure any new Python functions or methods have type hints for parameters and return values when practical.
- Keep code paths device agnostic: never assume CUDA is available and always provide a CPU fallback.

## Context
- ComfyUI nodes must remain friendly to downstream data storage in PostgreSQL-driven flows; surface structured metadata when possible.
- Hugging Face segmenter usage should avoid downloading weights at import time—perform heavy work lazily in node execution paths or caches.
