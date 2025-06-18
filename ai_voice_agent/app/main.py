from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Dict
from uuid import uuid4
from llm_handler import call_model
from tts_handler import text_to_audio
from stt_handler import Speech_to_voice
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from rag import load_and_prepare_vectorstore, build_qa_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from fastapi.responses import StreamingResponse


import os
import tempfile

app = FastAPI()

# Memory store for calls
sessions: Dict[str, BaseChatMessageHistory] = {}
chat_models: Dict[str, RunnableWithMessageHistory] = {}
vectorstore = None


# ----------------------------- Prompt Chain -----------------------------
def get_chain(model):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a friendly and persuasive AI sales agent named Sarah. "
                "You represent a company offering an 'AI Mastery Bootcamp'. "
                "Greet users warmly, ask a few questions to understand their interests, "
                "Only response the company or course related question, avoid other questions. "
                "and then pitch the course confidently. Respond naturally, handle any objections "
                "politely (such as price, time, or usefulness), and always try to move the conversation "
                "toward getting their interest or a commitment. Keep your tone polite, helpful, and enthusiastic.",),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | model | StrOutputParser()


# ----------------------------- API Models -----------------------------
class StartCallInput(BaseModel):
    phone_number: str
    customer_name: str

class RespondInput(BaseModel):
    message: str


# ----------------------------- API Endpoints -----------------------------
@app.post("/start-call")
def start_call(payload: StartCallInput):
    call_id = str(uuid4())

    # Setup chat history
    sessions[call_id] = ChatMessageHistory()

    # Setup model and history chain
    model = call_model()
    chain = get_chain(model)
    chain_with_history = RunnableWithMessageHistory(chain, lambda _: sessions[call_id])
    chat_models[call_id] = chain_with_history

    first_msg = f"Hi {payload.customer_name}, I’m Sarah from AI Mastery Bootcamp! Ready to boost your AI skills?"

    sessions[call_id].add_user_message(first_msg)
    response = chain_with_history.invoke([HumanMessage(content=first_msg)], config={"configurable": {"session_id": call_id}})
    sessions[call_id].add_ai_message(response)

    return {
        "call_id": call_id,
        "message": f"Call started with {payload.customer_name}",
        "first_message": response
    }


@app.post("/respond/{call_id}")
def respond(call_id: str, payload: RespondInput):
    if call_id not in sessions:
        return {"error": "Invalid call_id"}

    sessions[call_id].add_user_message(payload.message)
    chain = chat_models[call_id]
    response = chain.invoke([HumanMessage(content=payload.message)], config={"configurable": {"session_id": call_id}})
    sessions[call_id].add_ai_message(response)

    should_end = "thank you" in payload.message.lower() or "bye" in payload.message.lower()

    return {"reply": response, "should_end_call": should_end}


@app.get("/conversation/{call_id}")
def conversation(call_id: str):
    if call_id not in sessions:
        return {"error": "Invalid call_id"}
    history = sessions[call_id].messages
    return {"call_id": call_id, "history": [{"role": msg.type, "content": msg.content} for msg in history]}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    vectorstore = load_and_prepare_vectorstore(tmp_path)
    os.remove(tmp_path)
    return {"status": "PDF uploaded and processed for RAG."}


@app.post("/voice-response/")
async def get_voice(message: RespondInput):
    audio_buffer = text_to_audio(message.message)
    return StreamingResponse(audio_buffer, media_type="audio/wav")


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    text = Speech_to_voice(audio_bytes, file.filename)
    return {"transcription": text}


