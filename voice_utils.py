import streamlit as st
import requests
import openai
from gtts import gTTS
from io import BytesIO
import base64

def transcribe_audio_bytes(audio_bytes, api_key):
    """
    Transcribe the audio (provided as bytes) using OpenAI's Whisper API.
    """
    if len(audio_bytes) < 1000:
        st.error("Recording too short. Bitte sprechen Sie etwas Längeres.")
        return None

    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.wav"  # Whisper requires a filename

    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (audio_file.name, audio_file, "audio/wav")}
    data = {"model": "whisper-1"}

    response = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers=headers,
        files=files,
        data=data
    )
    if response.ok:
        return response.json().get("text")
    else:
        st.error("Transkription fehlgeschlagen.")
        st.write("Status code:", response.status_code)
        st.write("Error details:", response.text)
        return None

def record_and_transcribe(api_key):
    """
    Record a voice message using Streamlit's built-in audio input and transcribe it.
    """
    audio_value = st.audio_input("Frage per Sprache aufnehmen:")
    if audio_value is not None:
        audio_bytes = audio_value.read()
        transcript = transcribe_audio_bytes(audio_bytes, api_key)
        return transcript
    else:
        st.info("Bitte sprechen Sie Ihre Frage.")
        return None

def text_to_speech(text):
    """
    Convert text to speech using gTTS in German.
    """
    try:
        tts = gTTS(text, lang="de")  # Use German
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        st.error("Text-to-Speech-Konvertierung fehlgeschlagen: " + str(e))
        return None

def autoplay_audio(audio_data):
    """
    Autoplay the provided audio bytes in a hidden HTML audio element.
    """
    if not audio_data:
        return
    b64_audio = base64.b64encode(audio_data).decode()
    audio_html = f"""
    <audio autoplay style="display:none">
        <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def speak_text(text):
    """
    Convert a text string to German speech and play it silently.
    """
    tts_bytes = text_to_speech(text)
    autoplay_audio(tts_bytes)
