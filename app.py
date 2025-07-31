import streamlit as st
import chromadb
from Invoke_OpenAI import get_open_ai_response
from prompt.RAG_prompt import prompt, legal_prompt
from pdf_reader import read_pdf, extract_text_from_pdf
from chunking_strategy import invoke_text_spliter
from chromadb_function import create_collection, add_to_collection

st.set_page_config(
    page_title="RAG Application(PDF)",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded"
)

client = chromadb.PersistentClient("./mycollection")

st.title("RAG Application(PDF)🎉")
st.markdown("This app uses Chroma to perform similarity searches on a collection of documents.")

# Sidebar configurations
st.sidebar.title("Configurations")
st.sidebar.markdown("Adjust the settings for your query.")
upload_pdf = st.file_uploader("Upload File", type="pdf")
user_question = st.text_input("Ask a question", key='user_question')
n_results = 2
open_ai_key = st.sidebar.text_input("OpenAI Key",type="password")
collection_name = st.sidebar.text_input("Collection Name")


# Main functionality
if not open_ai_key:
    st.error("Please provide the OpenAI Key.")
else:
    if st.button("Get Answers"):
        if not upload_pdf: 
            st.error("Please provide the pdf file.")
        else:
            collection = create_collection(collection_name,client)
            reader = read_pdf(upload_pdf)
            pdf_content = extract_text_from_pdf(reader)
            text_chunks = invoke_text_spliter(separators=["\n\n", "\n", ". ", "? ", "! "], chunk_size=2000, chunk_overlap=250, content=pdf_content)
            add_to_collection(text_chunks=text_chunks, collection=collection)

            # st.write(f"User Question: {user_question}")
            # st.write(f"Number of results: {n_results}")

            results = collection.query(query_texts=[user_question], n_results=n_results, include=["documents", "metadatas"])

            result_text = "".join(results['documents'][0])
            response = get_open_ai_response(OpenAI_Key=open_ai_key, prompt=legal_prompt.format(user_question=user_question, search_text=result_text))
            st.write(f"Search results: {response}")
            st.success(f"Got the response successfully 🎯")
