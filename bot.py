import os
import streamlit as st
from dotenv import load_dotenv
from utils import store_docs, get_response, get_chroma_client

# ✅ Load API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY is missing! Please check your .env file.")
    st.stop()

# ✅ Streamlit UI Setup
st.set_page_config(page_title="Chat with Webpages", layout="wide")
st.title("💬 Chat with a Webpage (ChromaDB)")

# ✅ Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Sidebar - URL Input
with st.sidebar:
    st.header("🔗 Webpage Input")
    url = st.text_input("Enter Website URL")
    if st.button("Process URL") and url:
        with st.spinner("Processing webpage..."):
            store_docs(url)
            st.success("Webpage processed successfully! You can now ask questions.")

# ✅ Initialize ChromaDB Client
vector_store = get_chroma_client()

# ✅ Chat Interface
st.subheader("💡 Ask a Question")
user_query = st.text_area("Enter your question:")

if st.button("Get Answer") and user_query:
    with st.spinner("Fetching answer..."):
        response = get_response(user_query)
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Bot", response))

# ✅ Display Chat History
st.subheader("📜 Chat History")
chat_container = st.container()
with chat_container:
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"<div class='user-message'><b>You:</b> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'><b>Bot:</b> {message}</div>", unsafe_allow_html=True)

# ✅ Display stored documents (optional debugging)
if st.checkbox("Show Stored Documents"):
    docs = vector_store.get(include=['documents'])
    st.write(f"**Number of stored documents:** {len(docs['documents'])}")
    if docs['documents']:
        st.text_area("First Document:", docs['documents'][0], height=200)
