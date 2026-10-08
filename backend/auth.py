"""Shim tuong thich: ``backend.auth`` -> ``backend.security.auth``.

Giu import cu van chay (khong doi logic). Code moi nen import tu
``backend.security.auth``.
"""
from backend.security.auth import *  # noqa: F401,F403
