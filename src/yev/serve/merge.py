"""Merge a LoRA adapter into its base for engines that serve plain weights (vLLM, SGLang). Local only."""

from __future__ import annotations

import glob
import json
import os

MULTIMODAL_PREFIX, TEXT_PREFIX = "model.language_model.", "model."


def export_merged(adapter_dir: str, base: str, out: str, save_tokenizer: bool = True) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, dtype=dtype), adapter_dir)
    model.merge_and_unload().save_pretrained(out, safe_serialization=True)
    _text_only_names(out)
    if save_tokenizer:
        AutoTokenizer.from_pretrained(base).save_pretrained(out)


def _rename(key: str) -> str:
    return TEXT_PREFIX + key[len(MULTIMODAL_PREFIX):] if key.startswith(MULTIMODAL_PREFIX) else key


def _text_only_names(out: str) -> None:
    """Qwen3.5 saves as ``Qwen3_5ForCausalLM`` but keeps the multimodal ``model.language_model.`` prefix. vLLM and
    transformers remap it; SGLang loads none of those names and serves NaN logits. Rename to the text layout."""
    from safetensors.torch import load_file, save_file
    for path in glob.glob(os.path.join(out, "*.safetensors")):
        sd = load_file(path)
        if any(k.startswith(MULTIMODAL_PREFIX) for k in sd):
            save_file({_rename(k): v for k, v in sd.items()}, path, metadata={"format": "pt"})
    index = os.path.join(out, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            data = json.load(f)
        data["weight_map"] = {_rename(k): v for k, v in data["weight_map"].items()}
        with open(index, "w") as f:
            json.dump(data, f, indent=2)
