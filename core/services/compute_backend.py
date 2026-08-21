"""
Compute Backend Access

The two things every blocked, GPU-capable service needs, in one neutral place:
how to get its backend, and how many points to carry at once.

Both used to live in ``core.services.screen_selection``. That made the spatial
grid import from the *screen* module just to find out whether a graphics card
was available, which is backwards — the generic thing was filed inside the
specific one, and nobody looking for "how do I get a backend" would think to
open a file about lassos.
"""

import logging

logger = logging.getLogger(__name__)

# Points carried through a pipeline at once. Sized so the scratch arrays stay in
# the hundreds of MB rather than scaling with the cloud: a block is taken all the
# way to its answer before the next block starts, so peak memory is a property of
# this constant and not of the cloud.
DEFAULT_BLOCK = 8_000_000

# Registry getter per backend kind, and what to fall back to when the registry is
# not up — which is the normal state under unit tests. Imported lazily so that
# importing a service never needs a GPU, or even the plugins package.
_KINDS = {
    "selection": ("get_selection",
                  "plugins.backends.selection_backends", "NumpySelection"),
    "grid": ("get_grid",
             "plugins.backends.grid_backends", "NumpyGrid"),
}


def resolve_backend(kind, backend=None):
    """The backend to use for *kind*: the caller's, else the registry's, else CPU.

    Args:
        kind: which family of backend — see ``_KINDS``.
        backend: an explicit backend to use, or None to ask the registry. Tests
            pass one to compare the CPU and GPU paths against each other.

    Returns:
        A backend instance. Never raises: a missing registry means the CPU
        implementation, which is always importable.
    """
    if backend is not None:
        return backend

    try:
        getter_name, module_name, fallback_name = _KINDS[kind]
    except KeyError:
        raise ValueError(f"Unknown backend kind {kind!r}; "
                         f"expected one of {sorted(_KINDS)}")

    try:
        from config.config import global_variables
        registry = global_variables.global_backend_registry
        if registry is not None:
            return getattr(registry, getter_name)()
    except Exception as exc:                       # registry not up yet (tests)
        logger.debug(f"{kind} backend registry unavailable ({exc}); using CPU")

    import importlib
    return getattr(importlib.import_module(module_name), fallback_name)()
