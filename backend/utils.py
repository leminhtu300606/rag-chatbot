"""Shim tuong thich: ``backend.utils`` -> ``backend.common.utils``.

Giu import cu van chay (khong doi logic). Code moi nen import tu
``backend.common.utils``.
"""
from backend.common.utils import *  # noqa: F401,F403
