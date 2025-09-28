import os, json
from typing import Any, Dict
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

def load_config(path_or_none: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(defaults)
    if not path_or_none:
        return cfg
    path = path_or_none
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.endswith(('.yml','.yaml')):
        if not _HAS_YAML:
            raise RuntimeError("pyyaml required for YAML configs")
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f) or {}
    cfg.update(user)
    return cfg
