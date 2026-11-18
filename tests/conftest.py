"""Offline deterministic fixtures and mock SentenceTransformer models for embedxray tests."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
import torch
import torch.nn as nn

from embedxray.utils.tensor_ops import safe_l2_normalize


class OfflineMockTokenizer:
    """Deterministic whitespace and punctuation tokenizer for offline testing."""

    def tokenize(self, text: str) -> list[str]:
        return text.strip().split()

    def convert_ids_to_tokens(self, ids: list[int] | torch.Tensor) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [f"tok_{i}" for i in ids]


class OfflineMockSentenceTransformer(nn.Module):
    """Deterministic offline SentenceTransformer mock supporting output_value=None and standard encode."""

    def __init__(self, embedding_dim: int = 64, boilerplate_boost: float = 4.0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.boilerplate_boost = boilerplate_boost
        self.tokenizer = OfflineMockTokenizer()
        self.encoder_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            self.encoder_layer.weight.copy_(torch.eye(embedding_dim))

    def _token_to_vector(self, token: str) -> torch.Tensor:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        gen = torch.Generator(device="cpu")
        seed = int.from_bytes(digest[:8], byteorder="little") % (2**63 - 1)
        gen.manual_seed(seed)
        vec = torch.randn(self.embedding_dim, generator=gen, dtype=torch.float32)
        vec = vec / (torch.linalg.vector_norm(vec) + 1e-12)
        if token.startswith("[") or token.startswith("#") or token.startswith("<"):
            boiler_dir = torch.zeros(self.embedding_dim, dtype=torch.float32)
            boiler_dir[0] = self.boilerplate_boost
            vec = vec + boiler_dir
        return vec

    def encode_tokens_internal(self, texts: list[str]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            if not tokens:
                tokens = ["<empty>"]
            token_vecs = torch.stack([self._token_to_vector(t) for t in tokens], dim=0)
            token_vecs = self.encoder_layer(token_vecs)
            mask = torch.ones(len(tokens), dtype=torch.int64)
            pooled = token_vecs.mean(dim=0)
            pooled = safe_l2_normalize(pooled.unsqueeze(0)).squeeze(0)
            outputs.append(
                {
                    "token_embeddings": token_vecs.detach(),
                    "attention_mask": mask,
                    "sentence_embedding": pooled.detach(),
                    "tokens": tokens,
                }
            )
        return outputs

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        output_value: str | None = "sentence_embedding",
        convert_to_tensor: bool = True,
        convert_to_numpy: bool = False,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> Any:
        single_input = isinstance(sentences, str)
        texts = [sentences] if single_input else list(sentences)
        records = self.encode_tokens_internal(texts)

        if output_value is None:
            return records[0] if single_input else records

        if output_value == "token_embeddings":
            tok_list = [r["token_embeddings"] for r in records]
            return tok_list[0] if single_input else tok_list

        emb = torch.stack([r["sentence_embedding"] for r in records], dim=0)
        if single_input:
            emb = emb.squeeze(0)
        if convert_to_numpy and not convert_to_tensor:
            return emb.cpu().numpy()
        return emb


class HookOnlyMockModel(nn.Module):
    """Offline mock model that does not support output_value=None to test forward hook fallback."""

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tokenizer = OfflineMockTokenizer()
        self.token_projector = nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            self.token_projector.weight.copy_(torch.eye(embedding_dim))

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        projected = self.token_projector(token_features)
        return safe_l2_normalize(projected.mean(dim=1))

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        output_value: str | None = "sentence_embedding",
        convert_to_tensor: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        if output_value is None:
            raise TypeError("HookOnlyMockModel does not support output_value=None")
        texts = [sentences] if isinstance(sentences, str) else list(sentences)
        max_len = max(max(len(t.split()), 1) for t in texts)
        batch_feats = torch.zeros((len(texts), max_len, self.embedding_dim), dtype=torch.float32)
        for i, text in enumerate(texts):
            words = text.split() or ["<pad>"]
            for j, w in enumerate(words):
                seed = sum(ord(ch) for ch in w) + j * 17
                gen = torch.Generator().manual_seed(seed)
                batch_feats[i, j] = torch.randn(self.embedding_dim, generator=gen)
        return self.forward(batch_feats).detach()


@pytest.fixture
def isotropic_embeddings() -> torch.Tensor:
    """Returns deterministic (128, 64) unit-normalized isotropic embeddings."""
    gen = torch.Generator().manual_seed(101)
    raw = torch.randn((128, 64), generator=gen, dtype=torch.float32)
    return safe_l2_normalize(raw)


@pytest.fixture
def anisotropic_embeddings() -> torch.Tensor:
    """Returns deterministic (128, 64) narrow-cone embeddings with 8 dead dimensions."""
    gen = torch.Generator().manual_seed(202)
    dominant_axis = torch.randn((1, 64), generator=gen, dtype=torch.float32)
    dominant_axis = safe_l2_normalize(dominant_axis)
    noise = torch.randn((128, 64), generator=gen, dtype=torch.float32) * 0.03
    combined = dominant_axis * 0.99 + noise
    combined[:, -8:] = 0.0
    return safe_l2_normalize(combined)


@pytest.fixture
def hub_embeddings() -> torch.Tensor:
    """Returns deterministic (128, 64) embeddings containing artificial hubs."""
    gen = torch.Generator().manual_seed(303)
    base_direction = safe_l2_normalize(torch.randn((1, 64), generator=gen))
    peripheral = safe_l2_normalize(base_direction * 0.75 + torch.randn((128, 64), generator=gen) * 0.65)
    peripheral[0] = safe_l2_normalize(peripheral.mean(dim=0, keepdim=True)).squeeze(0)
    peripheral[1] = safe_l2_normalize(peripheral.mean(dim=0, keepdim=True) + 0.01).squeeze(0)
    return peripheral


@pytest.fixture
def mock_model() -> OfflineMockSentenceTransformer:
    """Returns a deterministic offline mock SentenceTransformer."""
    return OfflineMockSentenceTransformer(embedding_dim=64, boilerplate_boost=4.0)


@pytest.fixture
def hook_only_model() -> HookOnlyMockModel:
    """Returns a mock model that requires forward hooks for token attribution."""
    return HookOnlyMockModel(embedding_dim=64)
