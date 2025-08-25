import streamlit as st

try:
    from openai import OpenAI
    from streamlit_mic_recorder import mic_recorder
except ModuleNotFoundError as e:
    st.error(f"Missing dependency: {e.name}. Please install requirements with `pip install -r requirements.txt`.")
    st.stop()

from utils_io import (
    get_api_key,
    read_pdf,
    read_txt,
    audio_bytes_from_input,
    compute_mic_level,
)
from utils_rag import chunk_text, embed_texts, retrieve_context
from utils_ai import transcribe_cached, ask_llm, text_to_speech, estimate_cost

# ---------- Config ----------
st.set_page_config(page_title="Edu Voice MVP (Text Demo)", page_icon="🎓", layout="centered")

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

question = st.text_input(
    "Your question", placeholder="e.g., Explain photosynthesis in simple steps."
)

if recorded_audio:
    audio_bytes, fmt, sample_width = audio_bytes_from_input(recorded_audio)
    level = compute_mic_level(audio_bytes, sample_width)
    st.progress(min(int(level * 100), 100))
    if st.button("Re-record"):
        for k in ["recorder_output", "_last_mic_recorder_audio_id"]:
            st.session_state.pop(k, None)
        st.experimental_rerun()
    if not question.strip():
        cache = st.session_state.setdefault("transcription_cache", {})
        question = transcribe_cached(client, audio_bytes, fmt, cache)
        if question:
            st.markdown(f"**Transcribed question:** {question}")

col1, col2 = st.columns([1,1])
with col1:
    go = st.button("Get Answer")
with col2:
    clear = st.button("Clear")

if clear:
    for k in ("last_answer", "last_meta"):
        st.session_state.pop(k, None)
    st.experimental_rerun()

answer_box = st.empty()
st.session_state.setdefault("last_answer", "")
st.session_state.setdefault("last_meta", {})

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

    context = ""
    if kb_chunks and kb_embeds is not None:
        context = retrieve_context(client, kb_chunks, kb_embeds, question, top_k=top_k)

    user_prompt = (
        f"Answer the question using the context if relevant.\n\n"
        f"---\nContext:\n{context}\n---\n\nQuestion: {question}"
        if context
        else f"Question: {question}"
    )

    try:
        answer, usage, latency = ask_llm(
            client, model=model, system=system_prompt, user=user_prompt
        )
        answer_box.markdown(answer)
        st.session_state["last_answer"] = answer
        st.session_state["last_meta"] = {
            "latency": latency,
            "cost": estimate_cost(usage, model),
        }
    except Exception as e:
        st.error(f"LLM error: {e}")

if st.session_state.get("last_answer"):
    answer_box.markdown(st.session_state["last_answer"])
    meta = st.session_state.get("last_meta", {})
    if meta:
        st.caption(
            f"Latency: {meta.get('latency', 0):.2f}s • Estimated cost: ${meta.get('cost', 0):.6f}"
        )
    if st.button("🔊 Play Answer Audio"):
        audio_out = text_to_speech(client, st.session_state["last_answer"])
        if audio_out:
            st.audio(audio_out, format="audio/mp3")

st.markdown("---")
st.caption("Demo: text or voice Q&A with optional context from your uploads/notes.")
