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
