import streamlit as st
import time
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

openai_apikey = os.getenv('open_router_api_key')
mongodb_uri = os.getenv('mongo_db_uri')

st.title("RAG Chat App 🤖")
st.write("Ask questions about the war document — answers are grounded in your PDF via MongoDB Vector Search.")

# --- Setup clients (cached so they persist across reruns) ---
@st.cache_resource
def get_openai_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openai_apikey
    )

@st.cache_resource
def get_mongo_collection():
    mongo_client = MongoClient( mongodb_uri )
    return mongo_client["rag_db"]["war_documents"]

client = get_openai_client()
collection = get_mongo_collection()

# --- Embedding & search functions ---
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
LLM_MODEL = "openai/gpt-oss-120b:free"

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[{"content": [{"type": "text", "text": text}]}],
        encoding_format="float",
    )
    return response.data[0].embedding

def get_query_results(query):
    query_embedding = get_embedding(query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 2048,
                "limit": 5,
            }
        },
        {"$project": {"_id": 0, "text": 1}},
    ]
    results = collection.aggregate(pipeline)
    return [doc for doc in results]

def get_rag_response(query):
    context_docs = get_query_results(query)
    context_string = " ".join([doc["text"] for doc in context_docs])
    prompt = f"""Use the following pieces of context to answer the question at the end.
    {context_string}
    Question: {query}
"""
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content

# --- Chat UI ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about the war document. 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Searching documents... ⏳")

        try:
            assistant_response = get_rag_response(prompt)
        except Exception as e:
            assistant_response = f"Error: {e}"
        # # Simulate stream of response with milliseconds delay
        # full_response = ""
        # for chunk in assistant_response.split():
        #     full_response += chunk + " "
        #     time.sleep(0.05)
        #     # Add a blinking cursor to simulate typing
        #     message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(assistant_response)
        # st.markdown(assistant_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
