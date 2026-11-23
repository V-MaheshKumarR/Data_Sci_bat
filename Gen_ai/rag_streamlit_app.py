"""
Streamlit RAG Chat App (ChatGPT-like interface)
- Upload a PDF
- App chunks the PDF, creates embeddings, stores them in Chroma
- Chat with the document
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb

# ---------------- Setup ----------------
load_dotenv()
API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF and ask questions — powered by RAG + Chroma + OpenRouter")


# ---------------- Helpers ----------------
@st.cache_resource(show_spinner=False)
def get_openai_client():
    if not API_KEY:
        st.error("Missing OPEN_ROUTER_API_KEY in environment / .env file.")
        st.stop()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_store_streamlit")


def pdf_chunker(pdf_file_path: str):
    complete_chunks = []
    pdf_loader = PyPDFLoader(file_path=pdf_file_path)
    pdf_pages = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    for page in pdf_pages:
        complete_chunks.extend(text_splitter.split_text(page.page_content))
    return complete_chunks


def get_embedding(client: OpenAI, chunk_text: str):
    embedding = client.embeddings.create(
        model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        dimensions=1024,
        input=[{"content": [{"type": "text", "text": chunk_text}]}],
        encoding_format="float",
    )
    return embedding.data[0].embedding


def embed_chunks(client: OpenAI, chunks):
    embeddings = []
    progress = st.progress(0, text="Embedding chunks...")
    for i, ch in enumerate(chunks):
        embeddings.append(get_embedding(client, ch))
        progress.progress((i + 1) / len(chunks), text=f"Embedding {i + 1}/{len(chunks)}")
    progress.empty()
    return embeddings


def build_collection(chroma_client, collection_name, chunks, embeddings):
    # Reset collection for a fresh upload
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
    )
    ids = [f"doc_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "upload", "doc_id": i} for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return collection


def retrieve_context(client: OpenAI, collection, query: str, n_results: int = 3):
    query_embedding = get_embedding(client, query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    return " ".join(docs)


def chat_with_model(client: OpenAI, query: str, context: str, history):
    prompt = f"""Use the following pieces of context to answer the question at the end.
    If you don't know the answer based on the context, say you don't know.

    Context:
    {context}

    Question: {query}
    """
    messages = [{"role": "system", "content": "You are a helpful assistant that answers questions based on the provided PDF context."}]
    # include short history (last few turns) for continuity
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=messages,
    )
    return response.choices[0].message.content


# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ---------------- Sidebar: PDF Upload ----------------
with st.sidebar:
    st.header("📤 Upload PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
    process_btn = st.button("Process PDF", type="primary", disabled=uploaded_pdf is None)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.pdf_name:
        st.success(f"Active PDF: {st.session_state.pdf_name}")


# ---------------- Process Uploaded PDF ----------------
if process_btn and uploaded_pdf is not None:
    client = get_openai_client()
    chroma_client = get_chroma_client()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name

    try:
        with st.spinner("📑 Chunking PDF..."):
            chunks = pdf_chunker(tmp_path)
        st.info(f"Created {len(chunks)} chunks.")

        with st.spinner("🧠 Generating embeddings..."):
            embeddings = embed_chunks(client, chunks)

        with st.spinner("💾 Storing in Chroma..."):
            collection = build_collection(
                chroma_client,
                collection_name="uploaded_pdf_collection",
                chunks=chunks,
                embeddings=embeddings,
            )

        st.session_state.collection = collection
        st.session_state.pdf_name = uploaded_pdf.name
        st.session_state.messages = []
        st.success(f"✅ PDF processed! {collection.count()} chunks indexed. Ask your questions below.")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------- Chat Interface ----------------
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_query = st.chat_input(
    "Ask a question about your PDF..."
    if st.session_state.collection
    else "Please upload and process a PDF first..."
)

if user_query:
    if st.session_state.collection is None:
        st.warning("⚠️ Please upload and process a PDF before asking questions.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        client = get_openai_client()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    context = retrieve_context(client, st.session_state.collection, user_query)
                    answer = chat_with_model(
                        client, user_query, context, st.session_state.messages[:-1]
                    )
                except Exception as e:
                    answer = f"❌ Error: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
