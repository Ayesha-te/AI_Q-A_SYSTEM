import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["apikey"]


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

llm = OpenAI(temperature=0.7)
conversation = ConversationChain(
    llm=llm, 
    memory=st.session_state.memory,
    verbose=False
)

# Streamlit UI setup
st.title("💬 AI-Powered Q&A Chatbot")
st.write("Ask me anything <3")

# Initialize chat history (if not already initialized)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box for user to ask questions
user_input = st.text_input("You:", key="input")

if user_input:
    # Get response from the conversation chain
    response = conversation.run(user_input)
    
    # Update chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", response))

# Display the conversation history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑‍💻 {sender}:** {message}")
    else:
        st.markdown(f"**🤖 {sender}:** {message}")
