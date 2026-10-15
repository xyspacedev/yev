"""LoRA merge for the sidecars (Plan 7 Task 3)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from transformers import AutoModelForCausalLM  # noqa: E402

from yev.serve.merge import export_merged  # noqa: E402


def test_merged_matches_adapter(tmp_path, tiny_model):
    from peft import LoraConfig, get_peft_model
    base_dir, adapter_dir, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "merged"
    tiny_model.save_pretrained(base_dir)
    peft = get_peft_model(tiny_model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                 init_lora_weights=False))
    peft.save_pretrained(adapter_dir)
    x = torch.tensor([[3, 5, 7, 9]])
    with torch.no_grad():
        want = peft(input_ids=x).logits
    export_merged(str(adapter_dir), str(base_dir), str(out), save_tokenizer=False)
    merged = AutoModelForCausalLM.from_pretrained(out)
    assert not (out / "adapter_config.json").exists()
    with torch.no_grad():
        got = merged(input_ids=x).logits
    assert torch.allclose(got, want, atol=1e-4)


def test_merged_weights_use_text_only_names(tmp_path):
    # Qwen3.5 saves as Qwen3_5ForCausalLM but keeps the multimodal "model.language_model." prefix; SGLang loads
    # none of those names and serves NaN logits, so the export renames them to the text layout.
    import json

    from safetensors.torch import load_file, save_file

    from yev.serve.merge import _text_only_names
    save_file({"model.language_model.layers.0.w": torch.zeros(2), "lm_head.weight": torch.ones(2)}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {"model.language_model.layers.0.w": "model.safetensors",
                                        "lm_head.weight": "model.safetensors"}}))
    _text_only_names(str(tmp_path))
    assert set(load_file(tmp_path / "model.safetensors")) == {"model.layers.0.w", "lm_head.weight"}
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"model.layers.0.w", "lm_head.weight"}
