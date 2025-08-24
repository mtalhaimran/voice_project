import os
import io
import time

import streamlit as st

try:
    import numpy as np
    from pypdf import PdfReader
    from openai import OpenAI
    from streamlit_mic_recorder import mic_recorder
except ModuleNotFoundError as e:
    st.error(f"Missing dependency: {e.name}. Please install requirements with `pip install -r requirements.txt`.")
    st.stop()

# ---------- Config ----------
st.set_page_config(page_title="Edu Voice MVP (Text Demo)", page_icon="🎓", layout="centered")

# ---------- Helpers ----------
def get_api_key():
    # Prefer env var; allow UI override for quick tests
    k_env = os.getenv("OPENAI_API_KEY", "").strip()
    k_ui = st.session_state.get("api_key_input", "").strip()
    return k_ui or k_env

def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def read_pdf(file):
    reader = PdfReader(file)
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)

def chunk_text(text, max_tokens_est=400):
    # naive chunker by characters. ~4 chars ≈ 1 token. 400 tokens ≈ 1600 chars.
    chunk_size = 1600
    overlap = 200
    text = text.strip().replace("\r", "")
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def embed_texts(client, texts):
    # Use a small, cheap embedding model
    # text-embedding-3-small returns 1536-d vectors
    res = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in res.data], dtype=np.float32)

def cosine_sim(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(b_norm, a_norm)

def retrieve_context(client, kb_chunks, kb_embeds, query, top_k=3):
    if not kb_chunks:
        return ""
    q_vec = embed_texts(client, [query])[0]
    sims = cosine_sim(q_vec, kb_embeds)
    idx = np.argsort(-sims)[:top_k]
    top_chunks = [kb_chunks[i] for i in idx]
    return "\n\n".join(top_chunks)

def ask_llm(client, model, system, user, max_tokens=400, temperature=0.3):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    # Stream tokens to UI
    full = ""
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        full += delta
        yield delta
    # small delay to ensure UI flush
    time.sleep(0.05)

def transcribe_audio(client, file):
    try:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=file,
        )
        return transcript.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""

def text_to_speech(client, text, voice="alloy"):
    try:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
        )
        return io.BytesIO(speech.content)
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

# ---------- UI ----------
st.title("🎓 Educational AI – MVP (Text Q&A)")
st.caption("Paste/upload a small dataset, ask a question, get an answer. Model: gpt-4o-mini by default.")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("OpenAI API Key", type="password", key="api_key_input",
                  help="Prefer using environment variable OPENAI_API_KEY in production.")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    tone = st.selectbox("AI Tone / Role", ["Friendly Tutor", "Formal Academic", "Kid-Friendly", "Concise Assistant"], index=0)
    top_k = st.slider("Contexts to retrieve", 1, 5, 3)
    st.markdown("---")
    st.caption("Upload a small PDF or TXT (a few pages is fine for MVP).")

    uploaded = st.file_uploader("Upload knowledge (.pdf or .txt)", type=["pdf","txt"])
    pasted = st.text_area("Or paste notes here", height=180, placeholder="Paste a short chapter, bullet notes, or definitions...")

api_key = get_api_key()
if not api_key:
    st.warning("Add your OpenAI API key in the sidebar (or set OPENAI_API_KEY env var) to begin.")
    st.stop()

client = OpenAI(api_key=api_key)

# Build knowledge base (KB)
kb_texts = []
if pasted and pasted.strip():
    kb_texts.append(pasted.strip())

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".pdf"):
            kb_texts.append(read_pdf(uploaded))
        else:
            kb_texts.append(read_txt(uploaded))
    except Exception as e:
        st.error(f"Failed to read file: {e}")

kb_text = "\n\n".join([t for t in kb_texts if t])
kb_chunks, kb_embeds = [], None

if kb_text:
    kb_chunks = chunk_text(kb_text)
    try:
        kb_embeds = embed_texts(client, kb_chunks)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        kb_chunks = []
        kb_embeds = None

# Chat area
st.markdown("### Ask a question")

# Record audio directly in the browser
recorded_audio = mic_recorder(
    start_prompt="🎙️ Record Question",
    stop_prompt="Stop",
    use_container_width=True,
    key="recorder",
)

# Fallback to uploading an audio file
audio_question = st.file_uploader(
    "Or ask with voice (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"]
)

question = st.text_input(
    "Your question", placeholder="e.g., Explain photosynthesis in simple steps."
)

if recorded_audio and not question.strip():
    question = transcribe_audio(client, io.BytesIO(recorded_audio))
    if question:
        st.markdown(f"**Transcribed question:** {question}")
elif audio_question is not None and not question.strip():
    question = transcribe_audio(client, audio_question)
    if question:
        st.markdown(f"**Transcribed question:** {question}")

col1, col2 = st.columns([1,1])
with col1:
    go = st.button("Get Answer")
with col2:
    clear = st.button("Clear")

if clear:
    st.session_state.pop("last_answer", None)
    st.experimental_rerun()

answer_box = st.empty()
st.session_state.setdefault("last_answer", "")

# System prompts
tone_map = {
    "Friendly Tutor": "You are a friendly, supportive tutor. Explain clearly, step by step, with simple language and examples.",
    "Formal Academic": "You are a formal academic instructor. Provide rigorous, structured explanations with definitions and references if useful.",
    "Kid-Friendly": "You are a patient teacher speaking to a 10-year-old. Use simple words, short sentences, and analogies.",
    "Concise Assistant": "Answer concisely. Use bullet points when helpful. Avoid unnecessary detail."
}
system_prompt = tone_map[tone]

if go:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Build context
    context = ""
    if kb_chunks and kb_embeds is not None:
        context = retrieve_context(client, kb_chunks, kb_embeds, question, top_k=top_k)

    user_prompt = (
        f"Answer the question using the context if relevant.\n\n"
        f"---\nContext:\n{context}\n---\n\nQuestion: {question}"
        if context else f"Question: {question}"
    )

    collected = ""
    try:
        for token in ask_llm(client, model=model, system=system_prompt, user=user_prompt):
            collected += token
            answer_box.markdown(collected)
    except Exception as e:
        st.error(f"LLM error: {e}")
    if collected:
        st.session_state["last_answer"] = collected

if st.session_state.get("last_answer"):
    answer_box.markdown(st.session_state["last_answer"])
    if st.button("🔊 Play Answer Audio"):
        audio_out = text_to_speech(client, st.session_state["last_answer"])
        if audio_out:
            st.audio(audio_out, format="audio/mp3")

st.markdown("---")
st.caption("Demo: text or voice Q&A with optional context from your uploads/notes.")
