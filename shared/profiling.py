"""Opt-in per-stage timing for Warp training steps.

Kept out of the trainer modules because every trainer wants it and none of them owns it. Set
`WARP_3DGS_PROFILE_STEPS=1` to switch it on; it is off by default and costs nothing when off.

The companion for tile construction is `tile_builder.PROFILE_TILES`, which reads its own
environment variable and stores into `builder.stage_times`.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from time import perf_counter

import warp as wp

PROFILE_STEPS = os.environ.get("WARP_3DGS_PROFILE_STEPS", "") not in ("", "0", "false")


@contextmanager
def profile_stage(store, name, device):
    """Time one stage of a training step, synchronising so async GPU work lands in its own bucket.

    Off unless WARP_3DGS_PROFILE_STEPS is set, because the synchronise serialises work that
    normally overlaps. Without it, a timer around a kernel launch measures only the launch, and
    the first sync point in the iteration silently absorbs every earlier queued kernel -- which is
    how `tiles=` in the training log came to look like 79% of a step it barely contributes to.
    """
    if not PROFILE_STEPS:
        yield
        return
    wp.synchronize_device(device)
    started = perf_counter()
    yield
    wp.synchronize_device(device)
    store[name] = store.get(name, 0.0) + perf_counter() - started
