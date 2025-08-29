# voice_project

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

Provide your OpenAI API key in the sidebar when prompted.

### Voice features

You can ask questions by recording your voice directly in the browser. The app transcribes your voice with OpenAI's speech-to-text and replies with both text and an audio answer.

After an answer appears, use the **🔊 Play Answer Audio** button to listen to the response.

### Safari Compatibility

The in-browser voice recording relies on the MediaRecorder API, which is supported in Safari 14+
on macOS Big Sur and iOS 14 or later. Older Safari versions may not expose this API, so the
record button will be disabled or silently fail. Even on supported versions, Safari does not
allow recording from background tabs and may take a moment to initialize the microphone.

### Troubleshooting

- **Enable microphone access**: When recording fails, ensure Safari has permission to use the
  microphone. The first time you visit the app, Safari will prompt for access. You can change
  this later in *System Preferences → Privacy → Microphone* (macOS) or *Settings → Safari →
  Microphone* (iOS).
- **Select the correct input**: If no audio is captured, click the camera/microphone icon in the
  address bar to choose the right input device and refresh the page.
- **Fallback upload**: When browser recording is unsupported or permission is denied, record
  audio with another application and use the file upload option to submit your question.

