# app.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io
import torch
from translate import HindiTranslator  # Import the HindiTranslator class

translator = HindiTranslator()

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_file_like = io.BytesIO(pdf.read())
        pdf_reader = PdfReader(pdf_file_like)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    provided context just say, "The answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply:", response["output_text"])
    language = st.selectbox("Choose a regional language", ["Hindi", "Telugu", "Tamil"])

    if language == "Hindi":
        st.write("Translation in Hindi:", translator.to_hindi(response["output_text"]))
    elif language == "Telugu":
        st.write("Translation in Telugu:", translator.to_telugu(response["output_text"]))
    elif language == "Tamil":
        st.write("Translation in Tamil:", translator.to_tamil(response["output_text"]))
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(90deg, rgba(255,0,150,0.1) 0%, rgba(0,255,255,0.1) 100%);
            padding: 20px;
            border-radius: 10px;
        }
        .stApp {
            background: linear-gradient(45deg, #f3ec78, #af4261);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.header("Chat with Multiple PDFs!")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process", accept_multiple_files=True, type="pdf")
        if st.button("Submit & Process"):
            with st.spinner("Processing...."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
