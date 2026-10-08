"""Shim tuong thich: ``backend.cache`` -> ``backend.infra.cache``.

Giu import cu van chay (khong doi logic). Code moi nen import tu
``backend.infra.cache``.
"""
from backend.infra.cache import *  # noqa: F401,F403
