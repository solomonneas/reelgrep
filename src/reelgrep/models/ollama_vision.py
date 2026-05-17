"""Vision-LLM person/object-finding backend via Ollama HTTP API."""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

from reelgrep.frames import Frame
from reelgrep.models import BasePersonModel, Match, ModelError, register

__all__ = ["OllamaVisionModel"]

logger = logging.getLogger(__name__)


@register("ollama_vision")
class OllamaVisionModel(BasePersonModel):
    """Vision-LLM backend: per-frame chat-completion with positive + negative reference images."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        label: str = "subject",
    ) -> None:
        """Configure the Ollama endpoint, model id, request timeout, and subject label."""
        self._base_url = (
            base_url or os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self._model = model or os.environ.get("OLLAMA_VISION_MODEL", "qwen2-vl:7b")
        self._timeout = timeout
        self._label = label

    def config_dict(self) -> dict[str, Any]:
        """Return the effective backend configuration for manifest provenance."""
        return {
            "base_url": self._base_url,
            "model": self._model,
            "timeout": self._timeout,
            "label": self._label,
        }

    def _client(self):
        """Lazily import httpx and return a configured Client instance."""
        try:
            import httpx
        except ImportError as exc:
            raise ModelError(
                "ollama_vision backend requires the [vision] extra: "
                "pip install reelgrep[vision] (installs httpx)"
            ) from exc
        return httpx.Client(timeout=self._timeout)

    def _encode(self, path: Path) -> str:
        """Read a file and return its base64-encoded contents."""
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _build_prompt(self, has_negatives: bool) -> str:
        """Build the system prompt, optionally mentioning negative reference images."""
        neg_clause = (
            " You will also see KNOWN-NOT-MATCH reference images that look superficially similar"
            " but are NOT the subject - use them to avoid false positives."
            if has_negatives
            else ""
        )
        return (
            f"You are matching the SUBJECT '{self._label}' across video frames.\n"
            f"You will see KNOWN-MATCH reference images of the subject first.{neg_clause}\n"
            "Then you will see ONE candidate frame from a video.\n"
            "Decide whether the subject appears in the candidate frame.\n"
            "Respond with strict JSON only, no prose:\n"
            '{"match": true|false, "confidence": 0.0-1.0, "reasoning": "<one short sentence>"}'
        )

    def _score_frame(
        self,
        client,
        frame: Frame,
        positive_b64: list[str],
        negative_b64: list[str],
    ) -> dict[str, Any]:
        """Send one chat-completion request for a candidate frame and parse the JSON verdict."""
        prompt = self._build_prompt(has_negatives=bool(negative_b64))
        candidate_b64 = self._encode(Path(frame.path))
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "KNOWN-MATCH reference images:",
                "images": positive_b64,
            },
        ]
        if negative_b64:
            messages.append(
                {
                    "role": "user",
                    "content": "KNOWN-NOT-MATCH reference images:",
                    "images": negative_b64,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": "CANDIDATE frame:",
                "images": [candidate_b64],
            }
        )
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "format": "json",
        }
        url = f"{self._base_url}/api/chat"
        try:
            resp = client.post(url, json=payload)
        except Exception as exc:
            raise ModelError(f"ollama request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise ModelError(f"ollama HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            envelope = resp.json()
        except json.JSONDecodeError as exc:
            raise ModelError(f"ollama non-JSON response: {resp.text[:200]!r}") from exc
        content = envelope.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("ollama returned non-JSON content: %r", content[:200])
            return {"match": False, "confidence": 0.0, "reasoning": "model returned non-JSON"}
        return parsed

    def find(
        self,
        frames: list[Frame],
        positive_examples: list[Path],
        negative_examples: list[Path],
        *,
        threshold: float,
        top_k: int | None = None,
    ) -> list[Match]:
        """Score every frame against positive/negative examples and return filtered matches."""
        if not positive_examples:
            raise ModelError("ollama_vision requires at least one positive example")
        positive_b64 = [self._encode(p) for p in positive_examples]
        negative_b64 = [self._encode(p) for p in negative_examples] if negative_examples else []

        matches: list[Match] = []
        with self._client() as client:
            for frame in frames:
                if not Path(frame.path).exists():
                    continue
                parsed = self._score_frame(client, frame, positive_b64, negative_b64)
                if not parsed.get("match"):
                    continue
                conf = float(parsed.get("confidence", 0.0))
                if conf < threshold:
                    continue
                matches.append(
                    Match(
                        frame=frame,
                        confidence=conf,
                        bbox=None,
                        reasoning=str(parsed.get("reasoning", ""))[:300],
                    )
                )

        matches.sort(key=lambda m: m.confidence, reverse=True)
        if top_k is not None:
            matches = matches[:top_k]
        return matches
