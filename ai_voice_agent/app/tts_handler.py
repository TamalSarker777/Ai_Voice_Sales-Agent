import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.helpers import LocalAudioPlayer
import io
import soundfile as sf
import numpy as np

# Load .env
load_dotenv()

# Initialize API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def text_to_audio(text, voice="nova", tone="cheerful and confident"):
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions=f"Speak in a {tone} tone.",
        response_format="pcm"
    )

    # Get raw PCM audio bytes
    pcm_audio = response.content

    # Convert to 16-bit PCM numpy array
    audio_array = np.frombuffer(pcm_audio, dtype=np.int16)

    # Convert to WAV in memory
    buffer = io.BytesIO()
    sample_rate = 24000  # gpt-4o TTS outputs PCM at 24kHz
    sf.write(buffer, audio_array, samplerate=sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer