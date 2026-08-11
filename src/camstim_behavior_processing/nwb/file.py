"""Build the NWBFile shell (identity + subject) for a session.

Consumes the raw behavior pickle (loaded dict) to populate the NWB identity
fields (``session_start_time`` and the ``Subject``) and returns a
:class:`~pynwb.NWBFile`. An optional ``metadata`` override dict supplies
fields the pkl does not carry (experimenter, institution, subject
age/sex/genotype, ...).
"""

from __future__ import annotations

import datetime
from typing import Any
from uuid import uuid4

from pynwb import NWBFile
from pynwb.file import Subject


def build_subject(pkl: dict, metadata: dict[str, Any]) -> Subject:
    """Build a :class:`~pynwb.file.Subject` from the pkl, with overrides.

    Parameters
    ----------
    pkl : loaded behavior pickle. ``subject_id`` comes from
        ``items.behavior.params.mouse_id``.
    metadata : optional override dict. ``age`` is an ISO-8601 duration string
        (e.g. ``"P142D"``); ``date_of_birth`` is ``"YYYY-MM-DD"``. Other
        keys: ``species``, ``sex``, ``genotype``, ``strain``,
        ``subject_description``.

    Returns
    -------
    The populated ``Subject``.
    """
    mouse_id = str(
        pkl["items"]["behavior"]["params"].get("mouse_id", "unknown")
    )
    dob = metadata.get("date_of_birth")
    if isinstance(dob, str):
        try:
            dob = datetime.date.fromisoformat(dob)
        except ValueError:
            dob = None
    # pynwb Subject.date_of_birth requires a (tz-aware) datetime, not a date.
    if isinstance(dob, datetime.date) and not isinstance(
        dob, datetime.datetime
    ):
        dob = datetime.datetime(
            dob.year, dob.month, dob.day, tzinfo=datetime.timezone.utc
        )
    return Subject(
        subject_id=mouse_id,
        species=metadata.get("species", "Mus musculus"),
        age=metadata.get("age"),
        sex=metadata.get("sex", "U"),  # U = unknown
        genotype=metadata.get("genotype"),
        strain=metadata.get("strain"),
        date_of_birth=dob,
        description=metadata.get("subject_description"),
    )


def build_nwbfile(
    pkl: dict, metadata: dict[str, Any] | None = None
) -> NWBFile:
    """Create an empty :class:`~pynwb.NWBFile` with identity + subject.

    Parameters
    ----------
    pkl : loaded behavior pickle. ``session_start_time`` is taken from
        ``pkl['start_time']`` (naive datetimes are assumed UTC); the session
        description defaults to the behavior ``params.stage``.
    metadata : optional override dict — keys ``session_description``,
        ``identifier``, ``experimenter``, ``lab``, ``institution``, ``notes``
        plus the subject keys consumed by :func:`build_subject`.

    Returns
    -------
    A :class:`~pynwb.NWBFile` with no data yet.
    """
    metadata = metadata or {}
    params = pkl["items"]["behavior"]["params"]
    start_time = pkl.get("start_time")
    if start_time is None:
        start_time = datetime.datetime.now()
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=datetime.timezone.utc)

    nwb = NWBFile(
        session_description=metadata.get(
            "session_description", params.get("stage", "change detection")
        ),
        identifier=metadata.get("identifier", str(uuid4())),
        session_start_time=start_time,
        experimenter=metadata.get("experimenter"),
        lab=metadata.get("lab"),
        institution=metadata.get("institution"),
        notes=metadata.get("notes"),
    )
    nwb.subject = build_subject(pkl, metadata)
    return nwb
