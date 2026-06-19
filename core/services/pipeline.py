"""
Pipeline (macro) data model + capture / save / load.

A *pipeline* is the ordered sequence of analysis steps (plugin name + the
parameters it ran with) that produced a branch. Because every analysis result
node records ``tags=[plugin_name, params]`` and a ``parent_uid``, the sequence
can be reconstructed by walking the parent chain from a branch back to its root
-- no extra bookkeeping required.

This module is pure logic (no Qt, no globals). Capture takes the set of
replayable (analysis) plugin names from the caller so it can flag steps it
cannot reproduce. Replay orchestration lives in
``application/pipeline_runner.py``.

See ``docs/pipeline_macro_design.md`` for the full design and scope.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

# Bump when the on-disk schema changes in a backward-incompatible way.
PIPELINE_VERSION = 1


@dataclass
class PipelineStep:
    """One recorded step: a plugin and the parameters it ran with.

    ``produces`` is the result data_type, kept for human readability of the
    saved file; replay does not depend on it (the plugin emits its own type).

    ``bindings`` maps a param name to a human-readable *hint* (the original
    referenced branch's display name) for params that pointed at another branch
    by UUID. The UUID itself is deliberately NOT stored -- it is session-specific
    and meaningless on replay -- so these params are dropped from ``params`` at
    capture and re-bound to a live branch before the run starts.
    """
    plugin: str
    params: Dict[str, Any] = field(default_factory=dict)
    produces: str = ""
    bindings: Dict[str, str] = field(default_factory=dict)


@dataclass
class Pipeline:
    name: str = ""
    steps: List[PipelineStep] = field(default_factory=list)
    version: int = PIPELINE_VERSION


def _branch_ref_hint(value, data_nodes):
    """If ``value`` is a branch-UUID string, return a display-name hint for it;
    otherwise return None. Any UUID-shaped param is treated as a branch
    reference -- no legitimate plugin param is a bare UUID, and the point is to
    never persist a session UUID."""
    if not isinstance(value, str):
        return None
    try:
        u = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None
    node = data_nodes.get_node(u) if data_nodes is not None else None
    if node is None:
        return "(unknown branch)"
    return getattr(node, "alias", "") or getattr(node, "params", "") or "(branch)"


def _split_branch_refs(params, data_nodes):
    """Separate plain params from branch-UUID references.

    Returns ``(clean_params, bindings)`` -- ``clean_params`` keeps everything
    that is safe to persist; ``bindings`` maps each stripped param name to its
    hint (so it can be re-bound at replay)."""
    clean, bindings = {}, {}
    for name, value in params.items():
        hint = _branch_ref_hint(value, data_nodes)
        if hint is None:
            clean[name] = value
        else:
            bindings[name] = hint
    return clean, bindings


def capture_pipeline(branch_uid,
                     data_nodes,
                     supported_plugins: Iterable[str]) -> Tuple[Pipeline, List[str]]:
    """Reconstruct the analysis chain that produced ``branch_uid``.

    Walks ``parent_uid`` from the branch up to (but not including) the root,
    then orders the steps root -> branch. The root point cloud is the input,
    not a step, so it is excluded.

    Args:
        branch_uid: UID (str or UUID) of the outcome branch to capture.
        data_nodes: The ``DataNodes`` collection (``get_node(UUID)``).
        supported_plugins: Names of plugins that can be replayed (the analysis
            plugins). Steps whose plugin is not in this set are omitted and
            reported back so the caller can warn the user.

    Returns:
        (pipeline, unsupported) -- the capturable Pipeline (name left blank for
        the caller to fill) and the list of omitted (non-replayable) plugin
        names encountered, in chain order.
    """
    supported = set(supported_plugins)

    node = data_nodes.get_node(uuid.UUID(str(branch_uid)))
    chain = []
    while node is not None and node.parent_uid is not None:
        chain.append(node)
        node = data_nodes.get_node(node.parent_uid)
    chain.reverse()

    steps: List[PipelineStep] = []
    unsupported: List[str] = []
    for n in chain:
        plugin = n.tags[0] if n.tags else None
        if not plugin:
            # Node not produced by a recordable plugin (e.g. an action-plugin
            # edit) -- nothing to replay here.
            continue
        if plugin not in supported:
            unsupported.append(plugin)
            continue
        raw = n.tags[1] if len(n.tags) >= 2 else {}
        if not isinstance(raw, dict):
            raw = {}
        params, bindings = _split_branch_refs(raw, data_nodes)
        steps.append(PipelineStep(plugin=plugin, params=params,
                                  produces=n.data_type or "", bindings=bindings))

    return Pipeline(name="", steps=steps), unsupported


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Write the pipeline to ``path`` as JSON."""
    data = {
        "version": pipeline.version,
        "name": pipeline.name,
        "steps": [
            {"plugin": s.plugin, "params": s.params, "produces": s.produces,
             "bindings": s.bindings}
            for s in pipeline.steps
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_pipeline(path: str) -> Pipeline:
    """Read a pipeline JSON file written by :func:`save_pipeline`."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = [
        PipelineStep(
            plugin=s["plugin"],
            params=s.get("params", {}) or {},
            produces=s.get("produces", ""),
            bindings=s.get("bindings", {}) or {},
        )
        for s in data.get("steps", [])
    ]
    return Pipeline(
        name=data.get("name", ""),
        steps=steps,
        version=data.get("version", PIPELINE_VERSION),
    )
