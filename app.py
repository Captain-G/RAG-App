import streamlit as st
import chromadb
from Invoke_OpenAI import get_open_ai_response
from prompt.RAG_prompt import prompt, legal_prompt
from pdf_reader import read_pdf, extract_text_from_pdf
from chunking_strategy import invoke_text_spliter
from chromadb_function import create_collection, add_to_collection

st.set_page_config(
    page_title="RAG Chat (PDF)",
    page_icon="📄🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 RAG PDF Chatbot")
st.markdown("Upload a PDF and chat with it using Retrieval-Augmented Generation (RAG).")

# Sidebar configurations
st.sidebar.title("Configuration")
upload_pdf = st.sidebar.file_uploader("Upload PDF", type="pdf")
open_ai_key = st.sidebar.text_input("OpenAI API Key", type="password")
collection_name = st.sidebar.text_input("Collection Name").replace(" ", "_")

# Set up Chroma DB
client = chromadb.PersistentClient("./mycollection")

# Initialize chat session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "collection" not in st.session_state:
    st.session_state.collection = None

# Handle PDF upload and indexing
if upload_pdf and open_ai_key and collection_name and not st.session_state.collection:
    try:
        st.info("Processing PDF and building vector store...")
        collection = create_collection(collection_name, client)
        reader = read_pdf(upload_pdf)
        pdf_content = extract_text_from_pdf(reader)
        text_chunks = invoke_text_spliter(
            separators=["\n\n", "\n", ". ", "? ", "! "],
            chunk_size=2000,
            chunk_overlap=250,
            content=pdf_content
        )
        add_to_collection(text_chunks=text_chunks, collection=collection)
        st.session_state.collection = collection
        st.success("PDF indexed and ready for chat! 🎉")
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Chat input
user_input = st.chat_input("Ask your question here...")

if user_input and st.session_state.collection and open_ai_key:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant context from ChromaDB
    try:
        results = st.session_state.collection.query(
            query_texts=[user_input],
            n_results=2,
            include=["documents", "metadatas"]
        )
        search_context = "".join(results['documents'][0])
        full_prompt = legal_prompt.format(user_question=user_input, search_text=search_context)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = get_open_ai_response(OpenAI_Key=open_ai_key, prompt=full_prompt)
            placeholder.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error while getting response: {e}")

elif user_input and not upload_pdf:
    st.error("Please upload a PDF file.")
elif user_input and not open_ai_key:
    st.error("Please enter your OpenAI API Key.")
