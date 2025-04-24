import os
import streamlit as st
import toml
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["apikey"]

# App config
st.set_page_config(page_title="AI Q&A System")
st.title("🤖 AI-Powered Q&A with Memory")

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai_api_key
)

# Set up conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# UI - Question Input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question:", placeholder="Ask me anything...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    response = conversation.predict(input=user_input)
    st.write("🧠 **AI Response:**", response)

# Show full conversation history
with st.expander("📜 Chat History"):
    for msg in st.session_state.memory.chat_memory.messages:
        if msg.type == "human":
            st.markdown(f"🧍 **You**: {msg.content}")
        else:
            st.markdown(f"🤖 **AI**: {msg.content}")

# Optional: Reset button
if st.button("🔄 Reset Chat"):
    st.session_state.memory.clear()
    st.experimental_rerun()

