import speech_recognition as sr
import openai
import os
from openai import OpenAI
import io

openai.api_key = os.getenv("OPENAI_API_KEY")
    

# Transcription function with correct filename handling
def Speech_to_voice(audio_bytes, filename):
    client = OpenAI()

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename  # Preserve correct extension

    translation = client.audio.translations.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )

    return translation


