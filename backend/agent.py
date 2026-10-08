"""Shim tuong thich: ``backend.agent`` -> ``backend.generation.agent``.

Giu import cu van chay (khong doi logic). Code moi nen import tu
``backend.generation.agent``.
"""
from backend.generation.agent import *  # noqa: F401,F403
