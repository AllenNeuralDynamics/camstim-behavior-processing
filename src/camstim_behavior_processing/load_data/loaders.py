"""Load the two raw camstim session files (names vary per session).

A session is identified by a ``*_stim.pkl`` (camstim behavior pickle) and a
``*_sync.h5`` (sync file). These helpers just resolve/read the files; all
downstream builders take the loaded objects (or the paths) so callers can swap
in their own data.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_stim_pkl(pkl_path: str | Path) -> dict[str, Any]:
    """Load a camstim ``*_stim.pkl`` into its raw dict.

    The encoder traces used by the wheel builder live under
    ``pkl['items']['behavior']['encoders'][0]``.
    """
    with open(Path(pkl_path), "rb") as f:
        return pickle.load(f, encoding="latin1")
