# 🗣️ AI Voice Sales Agent

An AI-powered voice sales assistant that simulates real-time conversations with potential customers — capable of handling objections, personalizing pitches, and driving conversions for online courses.

🎥 **Demo Video**: [Watch Now](https://youtu.be/ANlN1o1OMk4)

---

## Features

1. Real-time voice-to-text (STT) and text-to-speech (TTS)  
2. Conversational intelligence using LLM with a sales persona  
3. Objection handling and lead qualification  
4. RAG integration (PDF upload for dynamic course info)  
5. Interactive Streamlit UI with audio recording and playback  
6. RESTful FastAPI backend with Swagger UI  

---

## Tech Stack

| Component             | Technology Used                                  |
|----------------------|--------------------------------------------------|
| **STT**              | `whisper-1` (OpenAI)                             |
| **TTS**              | `gpt-4o-mini-tts` (OpenAI)                        |
| **LLM**              | `gpt-4.1-nano-2025-04-14` via **LangChain**      |
| **RAG**              | FAISS + SentenceTransformers + PDFLoader         |
| **Backend**          | FastAPI                                          |
| **Frontend**         | Streamlit                                        |

---


## API Endpoints

| Endpoint                      | Purpose                                    |
|------------------------------|--------------------------------------------|
| `POST /start-call`           | Initiate a new call session                |
| `POST /respond/{call_id}`    | Send user reply and receive AI response    |
| `GET /conversation/{call_id}`| Get full conversation history              |
| `POST /upload-pdf/`          | Upload course PDFs for RAG context         |
| `POST /voice-response/`      | Generate voice response (TTS)              |
| `POST /transcribe/`          | Transcribe uploaded audio (STT)            |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)


### Install Requirements
pip install -r requirements.txt

### Set Up Environment Variables
Inside the AI_VOICE_AGENT/ directory, create a .env file and add: <br>

OPENAI_API_KEY="Your OpenAI API Key" <br>
Replace "Your OpenAI API Key" with your actual OpenAI key. <br>

### Run the Backend
uvicorn main:app --reload <br>
Visit: http://localhost:8000/docs

### Launch the Frontend (Streamlit UI)
cd streamlit_app <br>
streamlit run app.py <br>
This will open a local web interface for testing conversations.
