"""Config loading: YAML -> merged dict -> typed dataclass.

Every run in this project is defined by config, not code, so this module is how a
run becomes reproducible. It:

  * loads one or more YAML files and deep-merges them (base + overrides),
  * applies Hydra-style dotlist overrides ("train.learning_rate=1e-4"),
  * instantiates a typed dataclass (QLoRAConfig, TrainConfig, ...) from the
    result, tolerating unknown keys, and
  * can dump the fully-resolved config back to YAML so the exact settings of a
    run are saved next to its outputs.

Instantiation prefers a dataclass's own `from_dict` (several define one) and
falls back to filtering by declared fields — so it works for any of them.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml

T = TypeVar("T")

__all__ = [
    "load_yaml",
    "save_yaml",
    "deep_merge",
    "apply_overrides",
    "merge_configs",
    "instantiate",
    "load_config",
    "to_dict",
]


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_dict(obj: Any) -> dict:
    """dataclass instance or dict -> plain dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError(f"cannot convert {type(obj)} to dict")


def save_yaml(obj: Any, path: str | Path) -> None:
    """Dump a resolved config (dict or dataclass) to YAML for the run record."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(obj), f, sort_keys=False, default_flow_style=False)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override wins). Non-mutating."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"override path {dotted_key!r} traverses a non-dict")
    node[keys[-1]] = value


def apply_overrides(d: dict, overrides: list[str] | None) -> dict:
    """Apply "a.b.c=value" strings. RHS is parsed as YAML so types are natural
    (1e-4 -> float, true -> bool, [1,2] -> list)."""
    if not overrides:
        return d
    result = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"bad override (expected key=value): {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        _set_dotted(result, key.strip(), value)
    return result


def merge_configs(*paths: str | Path, overrides: list[str] | None = None) -> dict:
    """Deep-merge YAML files left-to-right, then apply dotlist overrides."""
    merged: dict = {}
    for p in paths:
        merged = deep_merge(merged, load_yaml(p))
    return apply_overrides(merged, overrides)


def instantiate(cls: Type[T], d: dict) -> T:
    """Build a dataclass from a dict via its from_dict, or field-filtering."""
    from_dict = getattr(cls, "from_dict", None)
    if callable(from_dict):
        return from_dict(d)  # type: ignore[return-value]
    field_names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in d.items() if k in field_names})  # type: ignore[call-arg]


def load_config(
    cls: Type[T],
    *paths: str | Path,
    overrides: list[str] | None = None,
    key: str | None = None,
) -> T:
    """Load + merge YAML into a typed config.

    `key` selects a nested section (e.g. load_config(TrainConfig, cfg, key="train")
    when the file namespaces settings under a `train:` block).
    """
    d = merge_configs(*paths, overrides=overrides)
    if key is not None:
        d = d.get(key, {})
    return instantiate(cls, d)