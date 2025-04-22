import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load OpenAI API key securely from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["apikey"]

# Initialize memory if not already done (for conversation context)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Initialize the OpenAI LLM with a temperature setting
llm = OpenAI(temperature=0.7)

# Set up the conversation chain (using the LLM and memory)
conversation = ConversationChain(
    llm=llm, 
    memory=st.session_state.memory,
    verbose=False
)

# Streamlit UI setup
st.title("💬 AI-Powered Q&A Chatbot")
st.write("Ask me anything <3")

# Initialize chat history if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box for user query
user_input = st.text_input("You:", key="input")

if user_input:
    # Get response from the conversation chain (AI response)
    response = conversation.run(user_input)
    
    # Update chat history for display
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", response))

# Display the conversation history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑‍💻 {sender}:** {message}")
    else:
        st.markdown(f"**🤖 {sender}:** {message}")
