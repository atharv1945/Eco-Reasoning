"""
tests/conftest.py — pytest session configuration for Eco-Reasoning
==================================================================
Runs before any test collection or execution.

Disables third-party telemetry background threads (PostHog from ChromaDB)
that are known to conflict with LightGBM's C++ memory allocator on Windows,
causing Fatal Python error: Aborted during model loading.
"""

import os

# ── Suppress PostHog telemetry (ChromaDB) ────────────────────────────────────
# PostHog spawns daemon threads at import time; these threads can race with
# LightGBM's LGBM_BoosterLoadModelFromString on Windows, triggering an abort.
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # ChromaDB toggle
os.environ.setdefault("CHROMA_TELEMETRY", "False")

# ── Force LightGBM single-threaded during test collection ────────────────────
os.environ.setdefault("LGBM_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
