"""I/O helper utilities for the Streamlit app."""
from __future__ import annotations

import base64
import io
from typing import Tuple, Union

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


def audio_bytes_from_input(recorded_audio: Union[dict, str]) -> Tuple[bytes, str, int]:
    """Normalize recorder output to raw bytes, format, and sample width.

    Args:
        recorded_audio: The object returned by ``mic_recorder``. It may be a
            dictionary or a base64 string depending on the environment.

    Returns:
        A tuple ``(audio_bytes, fmt, sample_width)``.
    """
    if isinstance(recorded_audio, dict):
        return (
            recorded_audio["bytes"],
            recorded_audio.get("format", "wav"),
            recorded_audio.get("sample_width", 2),
        )
    audio_bytes = base64.b64decode(recorded_audio.split(",")[-1])
    return audio_bytes, "wav", 2


def compute_mic_level(audio_bytes: bytes, sample_width: int = 2) -> float:
    """Return a 0-1 mic level estimation using RMS amplitude."""
    dtype = np.int16 if sample_width == 2 else np.int8
    data = np.frombuffer(audio_bytes, dtype=dtype)
    if data.size == 0:
        return 0.0
    level = float(np.sqrt(np.mean(np.square(data))))
    max_val = float(np.iinfo(dtype).max)
    return level / max_val
