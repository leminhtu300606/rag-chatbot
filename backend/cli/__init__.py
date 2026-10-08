"""Giao dien dong lenh (clean / index).

Canonical: ``backend.cli.main`` (truoc day ``backend.cli`` file don).
Giu tuong thich: ``python -m backend.cli <clean|index>`` va
``from backend.cli import build`` van chay.
"""
from .main import *  # noqa: F401,F403
from .main import __all__ as __all__  # noqa: F401
