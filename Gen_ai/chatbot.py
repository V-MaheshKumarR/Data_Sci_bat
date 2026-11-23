import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

st.title("💬 Chatbot")
st.caption("Powered by OpenRouter • Markdown responses rendered live")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Let's start chatting! 👇"}
    ]

# Sidebar: clear chat
with st.sidebar:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Let's start chatting! 👇"}
        ]
        st.rerun()

# Display chat history (st.markdown renders markdown automatically)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response (preserves markdown — newlines, code blocks, lists, etc.)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model="openai/gpt-oss-120b:free",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    full_response += token
                    # Live markdown render with a typing cursor
                    message_placeholder.markdown(full_response + "▌")

            # Final render without the cursor — full markdown formatting preserved
            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"❌ Error: {e}"
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# pip install streamlit openai python-dotenv
# streamlit run "D:\Data Science_bat\Gen_ai\chatbot.py"