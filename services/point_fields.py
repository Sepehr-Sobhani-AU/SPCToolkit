"""
Per-point field discovery and resolution, shared across plugins.

A "field" is any 1-D scalar defined per point on a branch: geometry (X/Y/Z),
normal components (nx/ny/nz), intensity, distance-to-ground, and every key in
the cloud's ``attributes`` dict (e.g. ``values`` from projected_distance).

Both ``color_by_value`` (map a field through a colormap) and ``query_select``
(test a field against a value) need the same two operations -- enumerate the
available fields, and resolve a field key to its scalar array -- so they live
here once instead of being duplicated per plugin.

Field keys are stable strings safe to persist in plugin params (and therefore
in saved pipelines): geometry/normal/intensity/distToGround use their bare key,
attribute fields use the ``attr:<name>`` prefix.
"""

from collections import OrderedDict

import numpy as np

from core.entities.values import Values


def enumerate_fields(node, pc) -> "OrderedDict[str, str]":
    """Build the ``{field_key: label}`` options offered for a branch.

    ``pc`` is a representative point cloud (cached or reconstructed) read for the
    presence of normals/intensity/etc.; ``node`` is the DataNode, used only to
    surface a ``values`` attribute that a Values branch exposes after
    reconstruction and to qualify it with the producing analysis' tag.
    """
    fields: "OrderedDict[str, str]" = OrderedDict()
    # Geometry is always available; Z leads since it is the common case.
    fields["z"] = "Z (height)"
    fields["x"] = "X"
    fields["y"] = "Y"

    if pc is not None:
        normals = getattr(pc, "normals", None)
        if normals is not None and len(normals) > 0:
            fields["nz"] = "Normal Z"
            fields["nx"] = "Normal X"
            fields["ny"] = "Normal Y"
        intensity = getattr(pc, "intensity", None)
        if intensity is not None and len(intensity) > 0:
            fields["intensity"] = "Intensity"
        d2g = getattr(pc, "distToGround", None)
        if d2g is not None and len(d2g) > 0:
            fields["distToGround"] = "Distance to ground"
        for key in (getattr(pc, "attributes", None) or {}):
            fields[f"attr:{key}"] = key

    # A values branch exposes its 'values' attribute only after reconstruction,
    # so offer it explicitly when the node carries Values.
    data = getattr(node, "data", None)
    is_values = isinstance(data, Values) or getattr(node, "data_type", "") == "values"
    if is_values and "attr:values" not in fields:
        fields["attr:values"] = "values"

    # Qualify the otherwise-generic 'values' entry with the analysis that
    # produced it (the node's first tag), e.g. "values (projected_distance)".
    if "attr:values" in fields:
        tags = getattr(node, "tags", None) or []
        if tags and tags[0]:
            fields["attr:values"] = f"values ({tags[0]})"

    return fields


def resolve_field(pc, key: str):
    """Return the 1-D scalar array for ``key`` from ``pc``, or None if the field
    is absent. Multi-column attributes use their first column."""
    if pc is None or not hasattr(pc, "points") or pc.points is None:
        return None
    pts = np.asarray(pc.points)

    if key in ("x", "y", "z"):
        return pts[:, {"x": 0, "y": 1, "z": 2}[key]]
    if key in ("nx", "ny", "nz"):
        normals = getattr(pc, "normals", None)
        if normals is None or len(normals) == 0:
            return None
        return np.asarray(normals)[:, {"nx": 0, "ny": 1, "nz": 2}[key]]
    if key == "intensity":
        v = getattr(pc, "intensity", None)
        return np.asarray(v) if v is not None and len(v) > 0 else None
    if key == "distToGround":
        v = getattr(pc, "distToGround", None)
        return np.asarray(v) if v is not None and len(v) > 0 else None
    if key.startswith("attr:"):
        v = (getattr(pc, "attributes", None) or {}).get(key[len("attr:"):])
        if v is None:
            return None
        v = np.asarray(v)
        return v[:, 0] if v.ndim > 1 else v
    return None


def representative_cloud(node):
    """A point cloud to read fields from WITHOUT forcing a rebuild: the cached
    reconstruction if present, else the node's own data when it is already a
    point cloud. Returns None when neither is cheaply available."""
    if node is None:
        return None
    cached = getattr(node, "cached_point_cloud", None)
    if getattr(node, "is_cached", False) and cached is not None and hasattr(cached, "points"):
        return cached
    data = getattr(node, "data", None)
    if hasattr(data, "points") and getattr(data, "points", None) is not None:
        return data
    return None
