"""Letter log-probs from vLLM or SGLang sidecars, for ``yev serve --backend vllm|sglang``.

Each question is one generation call of one token, restricted to (vLLM) or scored on (SGLang) its option letters.
Prompts are token ids, so the sidecar sees exactly the prompt the transformers readout scores. The returned scores
are log-probs, which differ from raw logits by a per-question constant, so ``readout.probs`` gives the same
calibrated probabilities. All questions of a request go out together so the engine can share the state's prefix.
"""

from __future__ import annotations

import asyncio

import httpx

TIMEOUT_S = 600.0


class EngineError(Exception):
    """The sidecar engine is unreachable or its answer can't be used; served as a 503."""


class _Backend:
    name = "engine"

    def __init__(self, url: str, client: httpx.AsyncClient | None = None):
        self.url = url.rstrip("/")
        self.client = client or httpx.AsyncClient(
            timeout=TIMEOUT_S, limits=httpx.Limits(max_connections=512, max_keepalive_connections=512))

    async def _post(self, path: str, body: dict) -> dict:
        try:
            r = await self.client.post(f"{self.url}{path}", json=body)
        except httpx.HTTPError as exc:
            raise EngineError(f"{self.name} engine at {self.url}: {type(exc).__name__}: {exc}") from exc
        if r.status_code != 200:
            raise EngineError(f"{self.name} engine at {self.url} returned {r.status_code}: {r.text[:500]}")
        return r.json()

    async def letter_logprobs(self, prompts: list[list[int]], letters: list[list[int]]) -> list[list[float]]:
        if len(prompts) != len(letters):
            raise ValueError(f"{len(prompts)} prompts but {len(letters)} letter lists")
        return list(await asyncio.gather(*(self._one(p, ls) for p, ls in zip(prompts, letters))))

    async def _one(self, prompt: list[int], letters: list[int]) -> list[float]:
        raise NotImplementedError

    def _pick(self, by_id: dict[int, float], letters: list[int]) -> list[float]:
        missing = [t for t in letters if t not in by_id]
        if missing:
            raise EngineError(f"{self.name} engine returned no log-prob for letter token ids {missing}")
        return [float(by_id[t]) for t in letters]


class VLLMBackend(_Backend):
    name = "vllm"

    def __init__(self, url: str, client: httpx.AsyncClient | None = None, model: str = "yev"):
        super().__init__(url, client)
        self.model = model

    async def _one(self, prompt: list[int], letters: list[int]) -> list[float]:
        body = {"model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                "allowed_token_ids": letters, "logprobs": len(letters), "return_tokens_as_token_ids": True}
        data = await self._post("/v1/completions", body)
        try:
            top = data["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError(f"vllm engine returned no top_logprobs: {str(data)[:300]}") from exc
        by_id = {}
        for key, lp in top.items():
            if not key.startswith("token_id:"):
                raise EngineError(f"vllm top_logprobs key {key!r} is not 'token_id:N' "
                                  "(the server must honour return_tokens_as_token_ids)")
            by_id[int(key.split(":", 1)[1])] = lp
        return self._pick(by_id, letters)


class SGLangBackend(_Backend):
    name = "sglang"

    async def _one(self, prompt: list[int], letters: list[int]) -> list[float]:
        body = {"input_ids": prompt, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True, "token_ids_logprob": letters}
        data = await self._post("/generate", body)
        try:
            entries = data["meta_info"]["output_token_ids_logprobs"][0]
            by_id = {int(e[1]): e[0] for e in entries}
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EngineError(f"sglang engine returned no output_token_ids_logprobs: {str(data)[:300]}") from exc
        return self._pick(by_id, letters)


def make_backend(name: str, url: str) -> VLLMBackend | SGLangBackend:
    if name == "vllm":
        return VLLMBackend(url)
    if name == "sglang":
        return SGLangBackend(url)
    raise ValueError(f"unknown backend {name!r} (vllm or sglang)")
