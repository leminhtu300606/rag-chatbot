"""Shim tuong thich: ``backend.core`` -> ``backend.config.settings``.

Giu import cu van chay (khong doi logic). Code moi nen import tu
``backend.config.settings``.
"""
from backend.config.settings import *  # noqa: F401,F403
