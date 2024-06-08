import os
import base64
import gc
import tempfile
import uuid

from IPython.display import Markdown, display
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

#if you want it to be fully open source
# from langchain_cohere import CohereEmbeddings

import streamlit as st

# Load environment variables from .env file
load_dotenv()

## load the GROQ And OpenAI API KEY 
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    st.session_state.context = None

session_id = st.session_state.id

def reset_chat():
    # """Reset chat history and context."""
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    # """Display PDF preview."""
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

def process_file(uploaded_file):
    # """Process uploaded PDF file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if os.path.exists(file_path):
            st.session_state.loader = PyPDFLoader(file_path)
        else:
            st.error('Could not find the file you uploaded, please check again...')
            st.stop()

        documents = st.session_state.loader.load()
        return documents

def setup_chain(documents):
    # """Setup retrieval chain."""
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    embeddings = OpenAIEmbeddings()
    #embeddings = CohereEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    final_documents = text_splitter.split_documents(documents)
    vectors = FAISS.from_documents(final_documents, embeddings) 

    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context}\n"
        "---------------------\n"
        "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
        "Query: {input}\n"
        "Answer: "
    )
    qa_prompt_tmpl = ChatPromptTemplate.from_template(qa_prompt_tmpl_str)
    document_chain = create_stuff_documents_chain(llm, qa_prompt_tmpl)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

with st.sidebar:
    st.header("Upload the PDF file")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        file_key = f"{session_id}-{uploaded_file.name}"
        if file_key not in st.session_state.file_cache:
            st.write("Please wait while the document is being indexed")
            try:
                documents = process_file(uploaded_file)
                retrieval_chain = setup_chain(documents)
                st.session_state.file_cache[file_key] = retrieval_chain
                st.success("Proceed to Chat!")
                display_pdf(uploaded_file)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()
        else:
            retrieval_chain = st.session_state.file_cache[file_key]
            st.success("Proceed to Chat!")
            display_pdf(uploaded_file)

col1, col2 = st.columns([6, 1])

with col1:
    st.header("Llama-3 PDF Chatbot")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        streaming_response = retrieval_chain.invoke({'input': prompt})
        
        for chunk in streaming_response['answer']:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(streaming_response["context"]):
                st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
