"""I/O helper utilities for the Streamlit app."""
from __future__ import annotations

import base64
import io
import wave
from typing import Optional, Tuple, Union

import numpy as np
from pypdf import PdfReader


def get_api_key() -> str:
    """Return the OpenAI API key from env or Streamlit session."""
    import os
    import streamlit as st

    k_env = os.getenv("OPENAI_API_KEY", "").strip()
    k_ui = st.session_state.get("api_key_input", "").strip()
    return k_ui or k_env


def read_txt(file: io.BytesIO) -> str:
    """Read a UTF-8 text file into a string."""
    return file.read().decode("utf-8", errors="ignore")


def read_pdf(file: io.BytesIO) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = PdfReader(file)
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts)


def audio_bytes_from_input(recorded_audio: Union[dict, str]) -> Tuple[bytes, str, Optional[int]]:
    """Normalize audio input to raw bytes, format, and sample rate.

    ``streamlit_mic_recorder`` can return either a dictionary with metadata or a
    raw base64 string depending on the browser. This helper converts either form
    into a tuple of ``(audio_bytes, format, sample_rate)`` and normalizes common
    MIME-style format strings (e.g., ``"audio/wav"`` -> ``"wav"``).
    """

    def _norm_fmt(fmt: str) -> str:
        if not fmt:
            return "wav"
        fmt = fmt.lower()
        if fmt.startswith("audio/"):
            fmt = fmt.split("/", 1)[-1]
        fmt = fmt.split(";")[0]
        if fmt in ("x-wav", "wave"):
            fmt = "wav"
        if fmt in ("aac", "m4a", "x-m4a"):
            fmt = "m4a"
        return fmt or "wav"

    if isinstance(recorded_audio, dict):
        data = (
            recorded_audio.get("bytes")
            or recorded_audio.get("blob")
            or recorded_audio.get("audio")
        )
        if hasattr(data, "read"):
            data = data.read()
        if data is None and recorded_audio.get("file") is not None:
            file_obj = recorded_audio["file"]
            try:
                file_obj.seek(0)
                data = file_obj.read()
            except Exception:
                data = None
        fmt = _norm_fmt(
            recorded_audio.get("format")
            or recorded_audio.get("type")
            or "wav"
        )
        if isinstance(data, str):
            audio_bytes = base64.b64decode(data)
        else:
            audio_bytes = data or b""
        sample_rate = recorded_audio.get("sample_rate")
        return audio_bytes, fmt, sample_rate

    if isinstance(recorded_audio, str):
        header, b64data = (recorded_audio.split(",", 1) + [""])[0:2]
        fmt = _norm_fmt("audio/" + header.split("audio/")[-1].split(";")[0] if "audio/" in header else "wav")
        audio_bytes = base64.b64decode(b64data)
        return audio_bytes, fmt, None

    raise TypeError("Unsupported audio input type")


def compute_mic_level(audio_bytes: bytes, sample_width: int = 2) -> float:
    """Return a 0-1 mic level estimation using RMS amplitude.

    The function accepts either raw PCM bytes or an entire WAV file. When
    WAV-formatted data is provided, the PCM frames and sample width are
    extracted from the container before computing the RMS.
    """
    if audio_bytes[:4] == b"RIFF":
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
    else:
        frames = audio_bytes

    dtype = np.int16 if sample_width == 2 else np.int8
    itemsize = np.dtype(dtype).itemsize
    usable_len = len(frames) - (len(frames) % itemsize)
    frames = frames[:usable_len]
    data = np.frombuffer(frames, dtype=dtype)
    if data.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    if not np.isfinite(rms):
        return 0.0
    max_val = float(np.iinfo(dtype).max)
    return rms / max_val
