
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
import toml

# Load OpenAI API Key
secrets = toml.load("secrets.toml")
openai_api_key = secrets["openai"]["api_key"]

# Streamlit UI
st.set_page_config(page_title="AI Q&A System")
st.title("🤖 AI-Powered Q&A System with Memory")

# Set up session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize LLM with memory
llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
conversation = ConversationChain(llm=llm, memory=st.session_state.memory, verbose=False)

# Input from user
user_input = st.text_input("Ask a question:")

# Handle input and output
if user_input:
    response = conversation.predict(input=user_input)
    st.write("🧠 AI:", response)

    # Show conversation history
    with st.expander("💬 Conversation History"):
        for msg in st.session_state.memory.chat_memory.messages:
            role = "You" if isinstance(msg, HumanMessage) else "AI"
            st.markdown(f"**{role}:** {msg.content}")
