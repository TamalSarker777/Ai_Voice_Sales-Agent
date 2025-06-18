from dotenv import load_dotenv
import os
import getpass
from langchain_openai import ChatOpenAI

load_dotenv()  

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    
 # LLM initialize   
def call_model():
    llm = ChatOpenAI(
    model="gpt-4.1-nano-2025-04-14",
    temperature=0,
    )
    return llm
