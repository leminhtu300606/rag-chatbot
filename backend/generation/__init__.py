"""
backend/generation/__init__.py - Tầng sinh câu trả lời (Generation)
==================================================================
Nhiệm vụ:
- Nhận context đã retrieve và sinh câu trả lời bằng LLM.

Thành phần:
- prompts.py   : định nghĩa SYSTEM_PROMPT và hàm dựng messages/prompt
- generator.py : gọi LLM backend (Ollama mặc định, Transformers tùy chọn)
- reranker.py  : cross-encoder rerank context trước khi dựng prompt
- rag.py       : điều phối retrieve -> rerank -> generate và trả về answer + sources
"""

from .generator import generate
from .prompts import SYSTEM_PROMPT, build_messages, build_prompt_text
from .rag import answer
from .reranker import rerank, RERANK_MODEL

__all__ = ["generate", "SYSTEM_PROMPT", "build_messages", "build_prompt_text", "answer", "rerank", "RERANK_MODEL"]
