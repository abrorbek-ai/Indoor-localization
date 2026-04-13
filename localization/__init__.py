"""Compatibility import for the clean root-level localization.py module.

The project now keeps the MVP pipeline in root files:
io_colmap.py, features.py, retrieval.py, localization.py, visualize.py.

This package remains only so old `import localization` calls resolve to the new
pipeline instead of the previous experimental package internals.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT_MODULE_PATH = Path(__file__).resolve().parent.parent / "localization.py"
_SPEC = importlib.util.spec_from_file_location("_menda_root_localization", _ROOT_MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load {_ROOT_MODULE_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

PipelineConfig = _MODULE.PipelineConfig
ReferenceEntry = _MODULE.ReferenceEntry
LocalizationContext = _MODULE.LocalizationContext
load_localization_context = _MODULE.load_localization_context
localize_query_image = _MODULE.localize_query_image
run_primary_localization = _MODULE.run_primary_localization
try_optional_pnp = _MODULE.try_optional_pnp

__all__ = [
    "PipelineConfig",
    "ReferenceEntry",
    "LocalizationContext",
    "load_localization_context",
    "localize_query_image",
    "run_primary_localization",
    "try_optional_pnp",
]
