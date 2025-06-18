# rag.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from llm_handler import call_model
import os


def load_and_prepare_vectorstore(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


def build_qa_chain(vectorstore):
    llm = call_model()

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = """
    You are a friendly and persuasive AI sales agent named Sarah. 
    You represent a company offering an 'AI Mastery Bootcamp'. 
    Greet users warmly, ask a few questions to understand their interests, 
    Only respond to company or course related questions, avoid others. 
    Then pitch the course confidently, handle objections politely, 
    and guide users toward interest or commitment.

    {context}
    Question: {question}

    Helpful Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retrievalQA = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return retrievalQA
