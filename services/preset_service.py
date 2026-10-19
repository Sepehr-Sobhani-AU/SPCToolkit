"""
Preset Service

Named sets of a plugin's settings, saved and loaded from its parameter
dialog. One JSON file per preset, in ``presets/<plugin name>/<preset>.json``
under the project folder, so presets travel with the project and are backed
up with it. Presets that ship with the project (e.g. Trace Lines'
"Overhead cable") live there too.

File format::

    {"plugin": "trace_lines", "name": "Overhead cable",
     "params": {"max_gap": 20.0, ...}}

A preset holds only setting values. Loading one fills the dialog; settings it
does not mention keep what the dialog shows, and keys the plugin no longer has
are ignored - so a preset survives a plugin gaining or losing a setting.
"""

import json
import os
import re

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESETS_ROOT = os.path.join(_PROJECT_ROOT, "presets")


def preset_dir(plugin_name: str, root: str = None) -> str:
    return os.path.join(root or PRESETS_ROOT, _safe(plugin_name))


def list_presets(plugin_name: str, root: str = None) -> list:
    """Preset names for *plugin_name*, sorted."""
    folder = preset_dir(plugin_name, root)
    if not os.path.isdir(folder):
        return []
    names = []
    for entry in os.listdir(folder):
        if entry.endswith(".json"):
            try:
                with open(os.path.join(folder, entry), encoding="utf-8") as f:
                    names.append(json.load(f).get("name") or entry[:-5])
            except (OSError, ValueError):
                continue          # an unreadable file is not a preset
    return sorted(names, key=str.lower)


def load_preset(plugin_name: str, name: str, root: str = None) -> dict:
    """The setting values saved under *name*."""
    with open(_path(plugin_name, name, root), encoding="utf-8") as f:
        return dict(json.load(f).get("params", {}))


def save_preset(plugin_name: str, name: str, params: dict, root: str = None) -> str:
    """Save *params* under *name* (replacing a preset of that name). Returns
    the file written."""
    name = name.strip()
    if not name:
        raise ValueError("a preset needs a name")
    path = _path(plugin_name, name, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"plugin": plugin_name, "name": name, "params": params},
                  f, indent=2, sort_keys=False)
    return path


def delete_preset(plugin_name: str, name: str, root: str = None) -> None:
    path = _path(plugin_name, name, root)
    if os.path.exists(path):
        os.remove(path)


def _path(plugin_name, name, root):
    return os.path.join(preset_dir(plugin_name, root), _safe(name) + ".json")


def _safe(text: str) -> str:
    """A file name from *text*: letters, digits, spaces, dots, dashes and
    underscores kept, anything else replaced."""
    return re.sub(r"[^\w .-]", "_", str(text)).strip() or "_"
