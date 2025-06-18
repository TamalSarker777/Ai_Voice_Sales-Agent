import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
import os
import openai
from llm_handler import call_model
from audio_recorder_streamlit import audio_recorder
from stt_handler import Speech_to_voice
from tts_handler import text_to_audio


text = None

# # Setting Environment
openai.api_key = os.getenv("OPENAI_API_KEY")


if 'model' not in st.session_state:
    st.session_state.model = None
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
    

#--------------------------------------------------
#Generate 4 digit random character
import random
import string
def generate_random_characters(length=4):
    """Generates a random string of the specified length using characters, digits, and punctuation."""
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

#----------------------------------------------------------    

# Getting previous history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # Check if history exists for this session ID
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]

random_characters = generate_random_characters()


config = {"configurable": {"session_id": f"{random_characters}"}}
if "config" not in st.session_state:
    st.session_state.config = config
    
    
def get_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly and persuasive AI sales agent named Sarah. "
                "You represent a company offering an 'AI Mastery Bootcamp'. "
                "Greet users warmly, ask a few questions to understand their interests, "
                "Only response the company or course related question, avoid other questions. "
                "and then pitch the course confidently. Respond naturally, handle any objections "
                "politely (such as price, time, or usefulness), and always try to move the conversation "
                "toward getting their interest or a commitment. Keep your tone polite, helpful, and enthusiastic."

                "Course to Pitch – AI Mastery Bootcamp:"
                "1. Duration: 12 weeks"
                "2. Price: $499 (special offer: $299)"

                "Key Benefits:"
                "1. Learn LLMs, Computer Vision, and MLOps"
                "2. Hands-on projects"
                "3. Job placement assistance"
                "4. Certificate upon completion",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    output_parser = StrOutputParser()
    chain = prompt | st.session_state.model | output_parser 
    return chain

    

def display_previous_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    

    

def input_from_user(with_message_history, text):
    display_previous_chat()

    if text:
        # Display user message in chat message container
        st.chat_message("user").markdown(text)
        st.session_state.messages.append({"role": "user", "content": text})

        # If RAG is available, use it
        if "vectorstore" in st.session_state:
            qa_chain = build_qa_chain(st.session_state.vectorstore)
            result = qa_chain.invoke({"query": text})  # use raw text

            assistant_reply = result['result']
        else:
            # fallback to regular LLM chat
            assistant_reply = with_message_history.invoke(
                [HumanMessage(content=text)],
                config=st.session_state.config
            )

        # Display assistant message
        st.chat_message("assistant").markdown(assistant_reply)
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    
        
        

st.title("Chat With Your :blue[Voice Sales Agent]:sunglasses: ", anchor=False)

with st.sidebar:
    st.subheader("🎤 Speak Now")
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        # Pass audio to Whisper model
        with st.spinner("Sending..."):
            try:
                # 'audio.wav' is just a placeholder name 
                text = Speech_to_voice(audio_bytes, "audio.wav")
            except Exception as e:
                st.warning(f"Cannot hear properly. Speak again !")
try:    
    # calling openai model
    llm = call_model()  
    st.session_state.model = llm
    with_message_history = RunnableWithMessageHistory(get_chain(), get_session_history) 
    input_from_user(with_message_history, text)
    
    with st.sidebar:
        # Speak it back
        with st.spinner("Converting to voice..."):
            audio_path = text_to_audio(st.session_state.messages[-1]["content"])
            st.subheader("🔊 Voice Output")
            st.audio(audio_path, format="audio/wav")
       
except:
    st.warning("Please say something.!")    
    
    
#----------------------------------------RAG -----------------------------------------

#Upload section
from rag import load_and_prepare_vectorstore, build_qa_chain

uploaded_file = st.sidebar.file_uploader("📄 Upload a PDF file for RAG", type="pdf", key="pdf_uploader_sidebar")
if uploaded_file is not None:
    if st.sidebar.button(":blue[Upload PDF]"):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Processing PDF for RAG..."):
            vectorstore = load_and_prepare_vectorstore("temp.pdf")
            os.remove("temp.pdf")
            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_uploaded = True  # Track that RAG is now ready
            st.success("✅ PDF uploaded! Future responses will use it.")
#----------------------------------------RAG-based Response -----------------------------------------

if "vectorstore" in st.session_state and "messages" in st.session_state and len(st.session_state.messages) > 0:
    try:
        last_user_msg = st.session_state.messages[-2]["content"] if st.session_state.messages[-1]["role"] == "assistant" else st.session_state.messages[-1]["content"]
        qa_chain = build_qa_chain(st.session_state.vectorstore)
        result = qa_chain.invoke({"query": last_user_msg})

        # Replace last assistant message with RAG-based one
        if st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages[-1]["content"] = result['result']
        else:
            st.session_state.messages.append({"role": "assistant", "content": result['result']})

        with st.chat_message("assistant"):
            st.markdown(result['result'])

        # Updating voice
        with st.sidebar:
            with st.spinner("Updating voice response..."):
                audio_path = text_to_audio(result['result'])
                st.subheader("AG Voice Output")
                st.audio(audio_path, format="audio/wav")
    except Exception as e:
        st.warning("RAG response failed. Falling back to normal chat.")


