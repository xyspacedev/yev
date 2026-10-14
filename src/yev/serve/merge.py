"""Merge a LoRA adapter into its base for engines that serve plain weights (vLLM, SGLang). Local only."""

from __future__ import annotations


def export_merged(adapter_dir: str, base: str, out: str, save_tokenizer: bool = True) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, dtype=dtype), adapter_dir)
    model.merge_and_unload().save_pretrained(out, safe_serialization=True)
    if save_tokenizer:
        AutoTokenizer.from_pretrained(base).save_pretrained(out)
