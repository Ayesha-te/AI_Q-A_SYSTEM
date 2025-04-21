import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load OpenAI API key securely
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["apikey"]

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Initialize conversation chain
llm = OpenAI(temperature=0.7)
conversation = ConversationChain(
    llm=llm, 
    memory=st.session_state.memory,
    verbose=False
)

# Streamlit UI
st.title("💬 AI-Powered Q&A Chatbot")
st.write("Talk to me like you would in a conversation!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("You:", key="input")

if user_input:
    # Get response from LLM
    response = conversation.run(user_input)
    
    # Update chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", response))

# Display conversation history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑‍💻 {sender}:** {message}")
    else:
        st.markdown(f"**🤖 {sender}:** {message}")
