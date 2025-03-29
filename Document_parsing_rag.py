import streamlit as st
import pinecone
import fitz  # PyMuPDF for PDFs
import docx
import pandas as pd
import openai
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAI

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536)  # Adjust dimensions as needed

index = pinecone.Index(PINECONE_INDEX_NAME)
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone(index, embeddings)

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_csv(index=False)

def process_file(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return extract_text_from_excel(file)
    else:
        return None

def chunk_and_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    vectorstore.add_texts(chunks)

st.title("RAG Agent with LangChain, OpenAI & Pinecone")

uploaded_file = st.file_uploader("Upload a file (PDF, Word, Excel)", type=["pdf", "docx", "xlsx"])
if uploaded_file:
    with st.spinner("Processing file..."):
        extracted_text = process_file(uploaded_file)
        if extracted_text:
            chunk_and_store(extracted_text)
            st.success("File processed and stored in vector database!")
        else:
            st.error("Unsupported file type")

query = st.text_input("Enter your query")
if query:
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA(llm=OpenAI(), retriever=retriever)
    response = qa_chain.run(query)
    st.write("**Response:**", response)
