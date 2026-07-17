# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Turns a raw camstim change-detection session (`*_stim.pkl` + `*_sync.h5`) into
analysis-ready pandas tables and a packaged NWB file. The code is split into two
composable layers; any intermediate can be built on its own.

### Layer 1 — `load_data/` (pure DataFrames, no NWB)

- **`trials_events.build_trials_and_events`** → a **`SessionData`** (see
  below), holding `events_df`, `intervals_df`, `timestamp_data`, and
  `task_parameters`. numpy/pandas/h5py.
- **`session_data.SessionData`** — the typed intermediate structure the loader
  returns and the `nwb/` layer consumes (a frozen dataclass; replaced the old
  loose `dict[str, Any]`). Holds only pandas/plain objects, so it stays on the
  eager import path (no scipy/pynwb). The **trials table is a derived view**:
  `SessionData.trials_df` = the `interval_type=='trial'` rows of `intervals_df`
  — `build_all` does **not** emit a standalone trials frame.
- **`running_speed.compute_running_speed`** → the wheel df (`speed`, `dx`,
  `v_sig`, `v_in` indexed by `timestamps`). Adds scipy. Takes the loaded pkl
  plus the sync `stim_vsync_fall` times (from `SessionData.timestamp_data`).
- **`loaders.load_stim_pkl`** → reads the raw behavior pickle.

No NWB packaging lives in `load_data` — keep it that way.

### Layer 2 — `nwb/` (DataFrames → NWB containers)

Produces a **complete** change-detection NWB (an `ndx_events.NdxEventsNWBFile`),
at content parity with the capsule's `package_to_nwb`. The writers each take a
Layer-1 DataFrame (or the loaded pkl) and add a container in place:

- `file.build_nwbfile(pkl, metadata=None)` / `file.build_subject` → the
  `NdxEventsNWBFile` shell with identity + `Subject`, all from the **pkl**
  (not session.json). `metadata` is an optional override dict.
- `hed_tags.py` — single source of truth for every HED fragment + description
  dict, plus `stim_presentation_hed` / `trial_hed` and `HED_SCHEMA_VERSION`.
- `acquisition.add_running_speed(nwb, wheel_df)` → `acquisition/{v_sig,v_in}` +
  `processing/running/{speed,dx}` (AllenSDK layout).
- `events.build_events_table` / `add_events` → the **ndx-events `EventsTable`**
  of discrete events (lick/reward/image_change/image_omission) with four
  `MeaningsTable`s (HED PASS design). Onsets/offsets and `miss` are dropped
  here (see `hed_tags.DROP_EVENT_TYPES`).
- `stimulus.add_stimulus_presentations` → `stimulus_presentations` and (if a
  movie is present) `natural_movie_one_presentations` `TimeIntervals`.
- `intervals.add_trials(nwb, intervals_df, warm_up_n, epoch_list)` → the trials
  table (booleans + timing + orientations + warm-up + epoch + HED outcome);
  `intervals.add_intervals` / `build_intervals_table` → the canonical flat
  `intervals` table (every interval type, timing + FKs + HED).
- `epochs.build_epoch_lookup` / `epoch_name_at` → the canonical epoch list
  (warm_up folded into change_detection; spontaneous gaps filled). `nwb.epochs`
  is intentionally **not** set — epochs live as `interval_type='epoch'` rows.
- `sidecar.build_events_sidecar` → the BIDS-style column sidecar dict.
- `nwb.assemble_nwbfile(pkl, session, wheel_df, *, metadata=...)` assembles the
  full NWB from already-loaded intermediates — `session` is the `SessionData`
  from `build_trials_and_events`; `pkl` (raw identity) and `wheel_df`
  (scipy-computed) stay separate. No legacy dir needed — this is what the tests
  drive.
- `nwb.package_nwb(pkl, sync, *, output_path=None, metadata=None, fmt="zarr",
  write_sidecar=True)` is the one-call path: raw files → intermediates →
  assembled NWB, written as **NWB-Zarr** (default; `fmt="hdf5"` also supported)
  with a `*.events.json` sidecar alongside.

Builders are split into small helpers deliberately to keep flake8
`max-complexity ≤ 10`; the null-coalescing `_f/_i/_s` helpers in `intervals.py`
exist so the per-row `add_trial` call stays under that limit.

### Lazy-import boundary (preserve it)

The package root and `load_data/__init__` re-export everything, but scipy- and
pynwb-dependent names (`compute_running_speed`, `package_nwb`, `build_nwbfile`)
are loaded **lazily** via `__getattr__`. So `import camstim_behavior_processing`
and building only the trials/events tables do not require scipy or pynwb. When
adding code, keep scipy/pynwb imports out of any module reachable from
`trials_events` / the eager import paths.

## Critical: the legacy-extractor dependency

`trials_events` does **not** reimplement the Stage-1 extractor — it wraps
`build_events_and_intervals.build_all`. That extractor lives in the
`data_loading_and_formatting_code/` directory inside the package at
`src/camstim_behavior_processing/data_loading_and_formatting_code/`. It is
declared as `package-data` in `pyproject.toml`, so it ships with the wheel.

This directory originated as a wholesale copy of the capsule's `code/` tree,
but has been **adopted into this library**: it is now owned, editable code
here, trimmed to only `build_all`'s real transitive closure — three items:
`build_events_and_intervals.py`, `task_parameters.py`, and the
`ndx_change_detection_task/` extension package (its `__init__.py` + `_specs/`
YAMLs). The ~110 unrelated capsule files (notebooks, session logs,
`package_to_nwb.py`, `sweepstim_packaging/`, `behavior_nwb_updates/`,
`run_capsule.py`, the sync/pkl loaders, etc.) were removed — they were never
reachable from `build_all`. Edits to event/interval/HED logic are now made
here directly (the capsule is no longer treated as upstream for this library).

Those legacy modules use bare (non-package) imports of their siblings (e.g.
`import task_parameters`), so the directory must be on `sys.path` first.
`load_data/_bootstrap.add_legacy_to_path()` does this idempotently; it resolves
the dir as `<package_root>/data_loading_and_formatting_code`. The
`from build_events_and_intervals import build_all` is done **inside**
`build_trials_and_events` (not at module load) so that importing the package, or
building only the wheel df, does not pull the legacy modules (and their heavier
deps) into scope.

History note: earlier revisions of this doc described the extractor as living in
a sibling dir *outside* the repo (supplied via `sys.path`), then as a wholesale
read-only vendoring of the capsule's `code/` tree. It is now adopted and trimmed
(see above), so `build_trials_and_events` / `package_nwb` work out of the box —
no external directory needed. If `ModuleNotFoundError: build_events_and_intervals`
appears, the dir is missing at the package root. The `trials_events` **unit** tests still
inject a fake `build_events_and_intervals` module (see `tests/test_load_data.py`
`setUp`) to keep those tests independent of the real extractor and its deps;
`test_legacy_extractor_module_exists` asserts the vendored file is present.

## Commands

This project uses **uv**. The CI (`.github/workflows/test_and_lint.yml`) runs:

```bash
uv sync                                              # install (incl. dev group)
uv run flake8 . && uv run interrogate --verbose .    # lint + docstring coverage
uv run coverage run -m unittest discover && uv run coverage report
```

Run a single test:

```bash
uv run python -m unittest tests.test_example.ExampleTest.test_assert_example
```

Formatting (not enforced in CI but configured): `uv run black .` and
`uv run isort .`. `.flake8` ignores **E203** for black slice-spacing compat.

## Enforced standards (all gates fail the build)

- **black** line length **79**, target py310 (`isort` uses the black profile).
- **interrogate** requires **100%** docstring coverage (incl. private helpers
  and module-level `__getattr__`).
- **coverage** `fail_under = 100` — every line must be covered (note `__init__`
  files are omitted from coverage, so `nwb/__init__.py`'s orchestration is
  exempt). The NWB writers + wheel builder are testable with synthetic
  pkl/DataFrames (see `tests/fixtures.py` — no legacy dir or real files
  needed); the `trials_events` wrapper needs the legacy extractor mocked.
  Note: single-row `HedTags` columns don't round-trip through Zarr, so test
  fixtures keep every HED-tagged table at ≥2 rows (real sessions always are).
- **flake8** `max-complexity = 10`.
- Python **>= 3.10**; CI matrix tests 3.10 / 3.11 / 3.12.

## Dependencies

Runtime deps are declared in `pyproject.toml`: `numpy`, `pandas`, `h5py`,
`scipy`, `pynwb`, `hdmf-zarr` (NWB-Zarr output), `ndx-events` (the
`EventsTable`/`MeaningsTable`/`NdxEventsNWBFile` — needs **>= 0.4.0**), and
`ndx-hed` (`HedTags` + `HedLabMetaData`). The `task_parameters` lab metadata is
a `ChangeDetectionTaskParameters` object built by the legacy extractor's
`ndx_change_detection_task` extension, so its namespace is only registered
once `build_trials_and_events` has run (the legacy dir is on `sys.path`).
