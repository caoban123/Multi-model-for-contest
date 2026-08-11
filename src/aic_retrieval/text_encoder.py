from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


class ClipTextEncoder:
    def __init__(
        self,
        model_id: str = DEFAULT_CLIP_MODEL_ID,
        cache_dir: Path | str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError(
                "text query search requires torch and transformers. "
                "Install them or use --query-vector/--query-from-video-id."
            ) from exc

        self._torch = torch
        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"local_files_only": local_files_only}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)

        self.model = CLIPModel.from_pretrained(model_id, **kwargs).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("query texts must not be empty")
        if any(not text.strip() for text in texts):
            raise ValueError("query text must not be empty")

        with self._torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            pooled_output = text_outputs.pooler_output
            text_features = self.model.text_projection(pooled_output)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy().astype("float32")


def encode_clip_text(
    text: str,
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    cache_dir: Path | str | None = None,
    local_files_only: bool = False,
) -> np.ndarray:
    encoder = ClipTextEncoder(model_id=model_id, cache_dir=cache_dir, local_files_only=local_files_only)
    return encoder.encode_text(text)


def encode_clip_texts(
    texts: list[str],
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    cache_dir: Path | str | None = None,
    local_files_only: bool = False,
) -> np.ndarray:
    encoder = ClipTextEncoder(model_id=model_id, cache_dir=cache_dir, local_files_only=local_files_only)
    return encoder.encode_texts(texts)
