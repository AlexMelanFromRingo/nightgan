"""Exported generators: saving, loading and generating images."""

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from .config import to_dict

MODEL_FORMAT = 1


def save_model(path, G: torch.nn.Module, model_cfg, meta: dict):
    """Writes the generator weights to `path` (.safetensors) and its configuration next to it (.json)."""
    path = Path(path)
    state = {k: v.detach().cpu().contiguous() for k, v in G.state_dict().items()}
    tmp = path.with_name(path.name + '.tmp')
    save_file(state, tmp)
    os.replace(tmp, path)
    info = dict(format=MODEL_FORMAT, model=to_dict(model_cfg), **meta)
    json_path = path.with_suffix('.json')
    tmp = json_path.with_name(json_path.name + '.tmp')
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, json_path)
