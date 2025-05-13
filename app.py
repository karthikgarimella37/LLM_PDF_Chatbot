import streamlit as st
import PyPDF2
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from gpt4all import GPT4All
import tempfile
import requests
from pathlib import Path

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load the LLM model
@st.cache_resource
def load_llm_model():
    model_path = "models"  # Folder path for the LLM model
    model_name = "q4_0-orca-mini-3b.gguf"
    return GPT4All(model_name, model_path=model_path, allow_download=False)

def extract_text_from_pdf(pdf_file):
    text_chunks = []
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name
    # Extract text from PDF
    with open(tmp_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            # Split text into chunks [500]
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            for chunk in chunks:
                text_chunks.append({
                    'text': chunk,
                    'page': page_num + 1
                })
    
    os.unlink(tmp_file_path)
    return text_chunks

# Create embeddings
def create_embeddings(text_chunks):
    model = load_embedding_model()
    texts = [chunk['text'] for chunk in text_chunks]
    embeddings = model.encode(texts)
    return embeddings

# Create FAISS index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

# Find relevant chunks
def find_relevant_chunks(query, k=3):
    model = load_embedding_model()
    query_embedding = model.encode([query])
    distances, indices = st.session_state.index.search(query_embedding.astype('float32'), k)
    return [st.session_state.pdf_text[i] for i in indices[0]]

# Generate answer with context and page numbers
def generate_answer(query, relevant_chunks):
    context = "\n".join([f"Page {chunk['page']}: {chunk['text']}" for chunk in relevant_chunks])
    prompt = f"""Context: {context}

Question: {query}

Please provide a detailed answer based on the context above.

Answer:"""
    
    if st.session_state.model is None:
        st.session_state.model = load_llm_model()
    
    response = st.session_state.model.generate(prompt, max_tokens=500)
    return response

# Streamlit UI 
st.title("PDF Question Answering System")
st.write("Upload a PDF file and ask questions about its content.")

# PDF File Uploader
pdf_file = st.file_uploader("Upload PDF", type=['pdf'])

if pdf_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            st.session_state.pdf_text = extract_text_from_pdf(pdf_file)
            
            # Create embeddings
            st.session_state.embeddings = create_embeddings(st.session_state.pdf_text)
            
            # Create FAISS index
            st.session_state.index = create_faiss_index(st.session_state.embeddings)
            
            st.success("PDF processed successfully!")

# Question textbox for user input
question = st.text_input("Ask a question about the PDF:")

# Answer generation button
if question and st.session_state.index is not None:
    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(question)
            
            # Generate answer
            answer = generate_answer(question, relevant_chunks)
            
            # Display answer
            st.write("Answer:")
            st.write(answer)
            
            # Display source pages
            st.write("\nSource Pages:")
            for chunk in relevant_chunks:
                st.write(f"Page {chunk['page']}") 