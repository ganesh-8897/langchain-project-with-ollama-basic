import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

st.title("🤖 Chat with your PDF using Ollama")

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    reader = PdfReader(pdf)
    raw_text = ''
    for page in reader.pages:
        raw_text += page.extract_text()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    embeddings = OllamaEmbeddings(model="mistral")  # or llama3, codellama, etc.
    db = FAISS.from_texts(texts, embeddings)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        docs = db.similarity_search(query)
        llm = Ollama(model="mistral")
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=query)
        st.write(answer)
