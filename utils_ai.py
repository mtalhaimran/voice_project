"""AI-related helpers for transcription, LLM, and TTS."""
from __future__ import annotations

import hashlib
import io
import time
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI


PRICING = {
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.00060 / 1000},
    "gpt-4o": {"input": 0.00250 / 1000, "output": 0.01000 / 1000},
}

def transcribe_cached(
    client: OpenAI, audio_bytes: bytes, fmt: str, cache: Dict[str, str]
) -> str:
    """Transcribe audio bytes with caching.

    The bytes are hashed with SHA-256.  If the hash exists in ``cache`` the
    cached transcription is returned.  Otherwise the bytes are written to a
    temporary file (with the appropriate extension) and sent to Whisper.
    """

    key = hashlib.sha256(audio_bytes).hexdigest()
    if key in cache:
        return cache[key]

    import tempfile

    suffix = f".{fmt}" if fmt else ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            with open(tmp.name, "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
        text = resp.text.strip()
    except Exception as e:
        text = ""
    cache[key] = text
    return text


def ask_llm(client: OpenAI, model: str, system: str, user: str, max_tokens: int = 400,
            temperature: float = 0.3) -> Tuple[str, Dict[str, int], float]:
    """Call the LLM and return text, usage, and latency."""
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise RuntimeError(f"LLM error: {e}") from e

    latency = time.time() - start
    text = resp.choices[0].message["content"]
    return text, resp.usage, latency


def ask_llm_stream(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    placeholder: Any,
    max_tokens: int = 400,
    temperature: float = 0.3,
) -> Tuple[str, Dict[str, int], float]:
    """Stream LLM output, updating the given placeholder."""
    start = time.time()
    text = ""
    usage: Optional[Dict[str, int]] = None
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and (content := delta.get("content")):
                text += content
                placeholder.markdown(text)
            if getattr(chunk, "usage", None):
                usage = chunk.usage
    except Exception as e:
        placeholder.error(f"LLM error: {e}")
        return "", None, time.time() - start

    latency = time.time() - start
    placeholder.markdown(text)
    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": len(text) // 4, "estimated": True}
    return text, usage, latency


def estimate_cost(usage: Optional[Dict[str, int]], model: str) -> float:
    """Estimate dollar cost for a completion given token usage.

    ``usage`` may be ``None`` (e.g., when streaming). In that case the function
    assumes ~4 characters per token based on the ``completion_tokens`` provided
    by the caller (if any).
    """

    if usage is None:
        return 0.0

    rates = PRICING.get(model, {"input": 0.0, "output": 0.0})
    prompt = usage.get("prompt_tokens") or usage.get("prompt_tokens_est", 0)
    completion = usage.get("completion_tokens") or usage.get("completion_tokens_est", 0)
    return prompt * rates["input"] + completion * rates["output"]


def text_to_speech(client: OpenAI, text: str, voice: str = "alloy") -> io.BytesIO | None:
    """Convert text to speech and return an MP3 buffer.

    Uses streaming if available for lower latency, falling back to the standard
    API. Returns ``None`` on failure.
    """

    try:
        try:
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts", voice=voice, input=text
            ) as stream:
                buf = io.BytesIO()
                for chunk in stream.iter_bytes():
                    buf.write(chunk)
            buf.seek(0)
            return buf
        except AttributeError:
            speech = client.audio.speech.create(
                model="gpt-4o-mini-tts", voice=voice, input=text
            )
            return io.BytesIO(speech.content)
    except Exception:
        return None
