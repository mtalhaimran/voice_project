import streamlit as st

try:
    from openai import OpenAI
except ModuleNotFoundError as e:
    st.error(
        f"Missing dependency: {e.name}. Please install requirements with `pip install -r requirements.txt`."
    )
    st.stop()

try:
    from streamlit_mic_recorder import mic_recorder
except ModuleNotFoundError:
    mic_recorder = None

from utils_io import (
    get_api_key,
    read_pdf,
    read_txt,
    audio_bytes_from_input,
)
from utils_rag import chunk_text, embed_texts, retrieve_context
from utils_ai import transcribe_cached, ask_llm, ask_llm_stream, text_to_speech, estimate_cost

def process_audio_question(
    client,
    question,
    model,
    system_prompt,
    kb_chunks,
    kb_embeds,
    top_k,
    placeholder=None,
):
    """Handle LLM call and TTS.

    Returns the question, answer text, synthesized audio,
    and metadata about the call (latency/cost).
    """

    if not question.strip():
        return question, "", None, {}

    context = ""
    if kb_chunks and kb_embeds is not None:
        context = retrieve_context(client, kb_chunks, kb_embeds, question, top_k=top_k)

    user_prompt = (
        f"Answer the question using the context if relevant.\n\n"
        f"---\nContext:\n{context}\n---\n\nQuestion: {question}"
        if context
        else f"Question: {question}"
    )

    if placeholder is not None:
        with st.spinner("Thinking..."):
            answer, usage, latency = ask_llm_stream(
                client,
                model=model,
                system=system_prompt,
                user=user_prompt,
                placeholder=placeholder,
            )
    else:
        with st.spinner("Thinking..."):
            answer, usage, latency = ask_llm(
                client, model=model, system=system_prompt, user=user_prompt
            )

    with st.spinner("Generating audio..."):
        audio_out = text_to_speech(client, answer)
    meta = {"latency": latency, "cost": estimate_cost(usage, model)}

    return question, answer, audio_out, meta

# ---------- Config ----------
st.set_page_config(page_title="Edu Voice MVP (Text Demo)", page_icon="🎓", layout="centered")

# ---------- UI ----------
st.title("🎓 Educational AI – MVP (Text Q&A)")
st.caption(
    "Paste/upload a small dataset, ask a question, get an answer. Model: gpt-4o-mini by default."
)

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
    st.warning(
        "Add your OpenAI API key in the sidebar (or set OPENAI_API_KEY env var) to begin."
    )
client = OpenAI(api_key=api_key) if api_key else None

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
    if client:
        try:
            kb_embeds = embed_texts(client, kb_chunks)
        except Exception as e:
            st.error(f"Embedding error: {e}")
            kb_chunks = []
            kb_embeds = None

# Chat area
st.markdown("### Ask a question")

question_box = st.empty()
transcript_box = st.empty()

# Determine whether the browser supports MediaRecorder with WebM
recorder_support = st.session_state.get("recorder_support")
if recorder_support is None:
    recorder_support = st.components.v1.html(
        """
        <script>
        const supported = !!window.MediaRecorder && MediaRecorder.isTypeSupported('audio/webm');
        Streamlit.setComponentValue(supported);
        </script>
        """,
        height=0,
    )
    if recorder_support is not None:
        st.session_state["recorder_support"] = recorder_support
recorder_supported = st.session_state.get("recorder_support", True)

# Record audio directly in the browser if supported
if mic_recorder and recorder_supported:
    recorded_audio = mic_recorder(
        start_prompt="🎙️ Record Question",
        stop_prompt="Stop",
        use_container_width=True,
        key="recorder",
    )
else:
    if not recorder_supported:
        st.info("Using basic recorder due to limited browser support.")
    elif not mic_recorder:
        st.info("streamlit-mic-recorder not installed. Using basic recorder.")
    rec_data = st.components.v1.html(
        """
        <div>
          <button id='start-rec'>🎙️ Start</button>
          <button id='stop-rec'>Stop</button>
          <p id='rec-msg'></p>
        </div>
        <script>
        var rec, chunks=[], mime='';
        const start=document.getElementById('start-rec');
        const stop=document.getElementById('stop-rec');
        const msg=document.getElementById('rec-msg');
        if(!window.MediaRecorder){
            msg.textContent='MediaRecorder not supported in this browser.';
            start.disabled=true;stop.disabled=true;
        }else{
            if(MediaRecorder.isTypeSupported('audio/webm')) mime='audio/webm';
            else if(MediaRecorder.isTypeSupported('audio/mp4')) mime='audio/mp4';
            else if(MediaRecorder.isTypeSupported('audio/ogg')) mime='audio/ogg';
            if(!mime){
                msg.textContent='No supported audio recording format.';
                start.disabled=true;stop.disabled=true;
            }
        }
        start.onclick=async ()=>{
            if(!mime) return;
            const stream=await navigator.mediaDevices.getUserMedia({audio:true});
            chunks=[];
            try{rec=new MediaRecorder(stream,{mimeType:mime});}
            catch(e){rec=new MediaRecorder(stream);}
            rec.ondataavailable=e=>chunks.push(e.data);
            rec.onstop=()=>{
                const blob=new Blob(chunks,{type:mime});
                const reader=new FileReader();
                reader.onload=()=>{Streamlit.setComponentValue({bytes:reader.result.split(',')[1],format:mime});};
                reader.readAsDataURL(blob);
            };
            rec.start();
        };
        stop.onclick=()=>{if(rec && rec.state!=='inactive') rec.stop();};
        </script>
        """,
        height=120,
    )
    uploaded_audio = st.file_uploader(
        "Upload audio", type=["wav", "mp3", "m4a", "aac"], accept_multiple_files=False
    )
    if rec_data:
        recorded_audio = rec_data
    elif uploaded_audio is not None:
        recorded_audio = {"file": uploaded_audio, "format": uploaded_audio.type}
    else:
        recorded_audio = None

if recorded_audio:
    # Only process and clear the question when a new recording is captured
    if recorded_audio != st.session_state.get("last_recorded_audio"):
        audio_bytes, fmt, _ = audio_bytes_from_input(recorded_audio)
        st.session_state["last_mic_audio_bytes"] = audio_bytes
        st.session_state["last_mic_audio_fmt"] = fmt
        st.session_state["last_recorded_audio"] = recorded_audio
        st.session_state.pop("question_text", None)

if st.session_state.get("last_mic_audio_bytes"):
    bytes_ = st.session_state["last_mic_audio_bytes"]
    fmt = st.session_state.get("last_mic_audio_fmt", "wav")
    st.audio(bytes_, format=f"audio/{fmt}")
    st.caption(f"Recorded audio: {len(bytes_)} bytes ({fmt})")
    if client and st.button("📝 Transcribe voice"):
        recorded_audio = st.session_state.get("last_recorded_audio")
        if recorded_audio is not None:
            cache = st.session_state.setdefault("transcription_cache", {})
            with st.spinner("Transcribing..."):
                try:
                    audio_bytes, fmt, _ = audio_bytes_from_input(recorded_audio)
                    text = transcribe_cached(client, audio_bytes, fmt, cache)
                except Exception as e:
                    st.error(str(e))
                else:
                    st.session_state["question_text"] = text
                    transcript_box.markdown(f"**Transcription:** {text}")
    elif not client:
        st.error("API key required for transcription.")
    if st.session_state.get("question_text"):
        transcript_box.markdown(
            f"**Transcription:** {st.session_state.get('question_text', '')}"
        )
else:
    st.session_state.pop("last_recorded_audio", None)

question = st.text_input(
    "Your question",
    placeholder="e.g., Explain photosynthesis in simple steps.",
    key="question_text",
)

if st.session_state.get("last_mic_audio_bytes") and st.button("Re-record"):
    for k in [
        "recorder_output",
        "question_text",
        "last_mic_audio_bytes",
        "last_mic_audio_fmt",
        "last_recorded_audio",
    ]:
        st.session_state.pop(k, None)
    st.session_state.get("transcription_cache", {}).clear()
    st.rerun()

col1, col2 = st.columns([1,1])
with col1:
    go = st.button("Get Answer")
with col2:
    clear = st.button("Clear")

if clear:
    for k in ("last_answer", "last_meta", "last_audio", "last_question"):
        st.session_state.pop(k, None)
    st.rerun()

answer_box = st.empty()
st.session_state.setdefault("last_answer", "")
st.session_state.setdefault("last_meta", {})
st.session_state.setdefault("last_audio", None)
st.session_state.setdefault("last_question", "")

# System prompts
tone_map = {
    "Friendly Tutor": "You are a friendly, supportive tutor. Explain clearly, step by step, with simple language and examples.",
    "Formal Academic": "You are a formal academic instructor. Provide rigorous, structured explanations with definitions and references if useful.",
    "Kid-Friendly": "You are a patient teacher speaking to a 10-year-old. Use simple words, short sentences, and analogies.",
    "Concise Assistant": "Answer concisely. Use bullet points when helpful. Avoid unnecessary detail."
}
system_prompt = tone_map[tone]

if go:
    if not client:
        st.error("API key required for LLM call.")
    else:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                q, answer, audio_out, meta = process_audio_question(
                    client,
                    question,
                    model,
                    system_prompt,
                    kb_chunks,
                    kb_embeds,
                    top_k,
                    answer_box,
                )
                st.session_state["last_question"] = q
                st.session_state["last_answer"] = answer
                st.session_state["last_meta"] = meta
                st.session_state["last_audio"] = audio_out
                question_box.markdown(f"**Question:** {q}")
                if not audio_out:
                    st.error("TTS failed.")
            except Exception as e:
                st.error(f"Processing error: {e}")

if st.session_state.get("last_answer"):
    question_box.markdown(f"**Question:** {st.session_state.get('last_question', '')}")
    answer_box.markdown(st.session_state["last_answer"])
    meta = st.session_state.get("last_meta", {})
    if meta:
        st.caption(
            f"Latency: {meta.get('latency', 0):.2f}s • Estimated cost: ${meta.get('cost', 0):.6f}"
        )
    if st.session_state.get("last_audio") and st.button("🔊 Play Answer Audio"):
        st.audio(st.session_state["last_audio"], format="audio/mpeg")

st.markdown("---")
st.caption("Demo: text or voice Q&A with optional context from your uploads/notes.")
