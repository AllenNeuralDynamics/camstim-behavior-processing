"""Path bootstrap for reusing the live Stage-1 extractor.

The trials/events builder reuses ``build_events_and_intervals`` from the
existing ``data_loading_and_formatting_code/`` directory. Those modules use
bare (non-package) imports of their siblings (e.g. ``task_parameters``), so
that directory must be on ``sys.path`` before they can be imported. This
helper adds it idempotently.
"""

from __future__ import annotations

import sys
from pathlib import Path

# code/change_detection_pipeline/ -> code/data_loading_and_formatting_code/
_LEGACY_DIR = (
    Path(__file__).resolve().parent.parent / "data_loading_and_formatting_code"
)


def add_legacy_to_path() -> Path:
    """Put the legacy extractor directory on ``sys.path`` (idempotent).

    Returns the resolved path so callers can assert/log it.
    """
    p = str(_LEGACY_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return _LEGACY_DIR
