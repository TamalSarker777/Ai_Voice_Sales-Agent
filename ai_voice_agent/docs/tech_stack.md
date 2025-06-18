1. Tech Stack Overview
   This document outlines the tools, frameworks, models, and architectural decisions used to implement the AI Voice Sales Agent.

2. Voice Processing
Speech-to-Text (STT):
    Model: whisper-1 (via OpenAI)

Reason: Highly accurate, simple API, no need to manage local audio processing.


Text-to-Speech (TTS)
    Model: gpt-4o-mini-tts (via OpenAI)

    Format: Returned audio in 24kHz PCM, converted to WAV using soundfile.

Note: Due to browser restrictions, auto-play is not possible — requires user click to play.

3. Conversational Intelligence
LLM for Dialogue
    Model: gpt-4.1-nano-2025-04-14

    Framework: LangChain

Capabilities:
    Dynamic prompting with a predefined sales agent persona ("Sarah")
    Handles user queries, objections, and pushes for conversion

4. RAG (Retrieval-Augmented Generation)
    Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (via HuggingFace)

    Vector Store: FAISS

    PDF Parsing: PyPDFLoader from langchain_community.document_loaders

    Use Case: Upload course-related PDFs to enhance conversation relevance

5. Backend API
Framework:
    FastAPI — clean RESTful backend with interactive Swagger docs

6. Endpoints Implemented
     Endpoint	                               Purpose
POST /start-call	           Starts a new call session with a unique call_id
POST /respond/{call_id}	       Processes user reply and returns AI response
GET /conversation/{call_id}	   Returns full conversation history
POST /upload-pdf/	           Enables PDF upload for dynamic context
POST /voice-response/	       Returns TTS audio stream from a given message
POST /transcribe/	           Returns STT transcription from audio file

7. Frontend
Streamlit
    Simulates a chat interface with:
        Audio recording input
        Voice playback
        State/session handling
        Optional RAG via PDF upload

8. Local Setup
Requirements Installation:
    pip install -r requirements.txt
Run FastAPI Backend
    uvicorn main:app --reload

Visit: http://localhost:8000/docs for interactive API testing.

Run Streamlit UI
    cd streamlit_folder_name
    streamlit run app.py


9. Environment Variables
Create a .env file with:
    OPENAI_API_KEY=your_openai_key


10. Not Included (By Design)
    Multi-agent flow (LangGraph): Not required for this use case, but can be added if needed.

    Telephony integration: Call simulation only- no actual phone call functionality as per task instructions.

11. Final Note
This project was developed alongside my regular machine learning tasks. I focused on creating a working, modular solution that can be easily expanded in the future, all within a tight deadline. The design choices were made to ensure the system is clear, reliable, and easy to improve later on.