import json, pytest, torch

class FakeTok:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1}
        for L in "ABCDEF":
            self.vocab[L] = len(self.vocab)
        self.pad_token_id = 0
        self.eos_token_id = 1
    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab) % 2000 + 8 if len(self.vocab) >= 2000 else len(self.vocab)
        return self.vocab[w]
    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.split()]
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        # accepts enable_thinking like the real Qwen template
        words = []
        for m in messages:
            words += ["<|" + m["role"] + "|>"] + m["content"].replace('"', " ").split()
        if add_generation_prompt:
            words.append("<|assistant|>")
        return [self._id(w) for w in words]

@pytest.fixture
def tok():
    return FakeTok()

@pytest.fixture
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=2048, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return LlamaForCausalLM(cfg)

def chat_row(id_, letters, gold_letter, type_="choice", cluster=None, examples=0, state="the state text"):
    opts = [{"label": L, "key": k, "description": f"desc {k}"} for L, k in letters.items()]
    user = json.dumps({"state": state, "question": "q?", "options": opts})
    msgs = [{"role": "system", "content": "sys"}]
    for _ in range(examples):
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": "A"}]
    msgs.append({"role": "user", "content": user})
    return {"id": id_, "messages": msgs, "answer": gold_letter, "letters": letters,
            "target": {L: float(L == gold_letter) for L in letters}, "weight": 1.0, "type": type_,
            "family": "f", "source": "s", "cluster_id": cluster, "edit_type": None, "split": "train"}

@pytest.fixture
def make_row():
    return chat_row
