"""AI-related helpers for transcription, LLM, and TTS."""
from __future__ import annotations

import hashlib
import io
import time
from typing import Any, Dict, Tuple

from openai import OpenAI


PRICING = {
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.00060 / 1000},
    "gpt-4o": {"input": 0.00250 / 1000, "output": 0.01000 / 1000},
}


def transcribe_audio(client: OpenAI, file: io.BytesIO) -> str:
    """Transcribe audio using Whisper API (whisper-1)."""
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
        )
        return transcript.text
    except Exception:
        return ""


def transcribe_cached(client: OpenAI, audio_bytes: bytes, fmt: str, cache: Dict[str, str]) -> str:
    """Transcribe audio bytes with caching by hash."""
    key = hashlib.sha256(audio_bytes).hexdigest()
    if key not in cache:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"question.{fmt}"
        cache[key] = transcribe_audio(client, audio_file)
    return cache[key]


def ask_llm(client: OpenAI, model: str, system: str, user: str, max_tokens: int = 400,
            temperature: float = 0.3) -> Tuple[str, Dict[str, int], float]:
    """Call the LLM and return text, usage, and latency."""
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
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
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    text = ""
    usage: Dict[str, int] = {}
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and (content := delta.get("content")):
            text += content
            placeholder.markdown(text)
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    latency = time.time() - start
    placeholder.markdown(text)
    return text, usage, latency


def estimate_cost(usage: Dict[str, int], model: str) -> float:
    """Estimate dollar cost for a completion given token usage."""
    rates = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return usage.get("prompt_tokens", 0) * rates["input"] + usage.get("completion_tokens", 0) * rates["output"]


def text_to_speech(client: OpenAI, text: str, voice: str = "alloy") -> io.BytesIO | None:
    """Convert text to speech and return an MP3 buffer."""
    try:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
        )
        return io.BytesIO(speech.content)
    except Exception:
        return None
