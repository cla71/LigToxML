"""Ollama model management for local Qwen 3.5 inference."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"

# Model assignments: heavier reasoning uses 9B, lighter tasks use 4B
ORCHESTRATOR_MODEL = "qwen3:8b"
WORKER_MODEL = "qwen3:4b"


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ConversationContext:
    messages: list[Message] = field(default_factory=list)
    model: str = ORCHESTRATOR_MODEL

    def add(self, role: str, content: str, **kwargs: Any) -> None:
        self.messages.append(Message(role=role, content=content, **kwargs))

    def to_ollama_messages(self) -> list[dict]:
        out = []
        for m in self.messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            out.append(entry)
        return out


def check_ollama_health() -> bool:
    """Check if Ollama server is reachable."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def list_local_models() -> list[str]:
    """Return names of models already pulled in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def ensure_model(model_name: str) -> bool:
    """Pull a model if not already available. Returns True on success."""
    local = list_local_models()
    # Check if model is already available (exact match or prefix match)
    for m in local:
        if m == model_name or m.startswith(model_name.split(":")[0]):
            logger.info("Model %s already available locally", model_name)
            return True

    logger.info("Pulling model %s (this may take a while)...", model_name)
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/pull",
            json={"name": model_name, "stream": False},
            timeout=1800,
        )
        return resp.status_code == 200
    except Exception as exc:
        logger.error("Failed to pull model %s: %s", model_name, exc)
        return False


def chat_completion(
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> dict:
    """Send a chat completion request to Ollama with optional tool definitions.

    Returns the parsed JSON response from the Ollama API.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = tools

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Ollama connection failed, retrying in %ds...", wait)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    "Cannot connect to Ollama. Start it with: ollama serve"
                )
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise RuntimeError(f"Ollama chat failed: {exc}") from exc

    raise RuntimeError("Exhausted retries for Ollama chat completion")


def extract_response_text(response: dict) -> str:
    """Extract the assistant's text content from an Ollama response."""
    msg = response.get("message", {})
    return msg.get("content", "")


def extract_tool_calls(response: dict) -> list[dict]:
    """Extract tool calls from an Ollama response."""
    msg = response.get("message", {})
    return msg.get("tool_calls", [])
