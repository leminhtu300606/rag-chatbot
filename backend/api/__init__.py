"""Giao dien HTTP (FastAPI).

Canonical: ``backend.api.app`` (truoc day ``backend.api`` file don).
Giu tuong thich: ``uvicorn backend.api:app`` va ``python -m backend.api`` van chay.
"""
from .app import app

__all__ = ["app"]
