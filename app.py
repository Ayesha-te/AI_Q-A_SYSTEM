import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load OpenAI API key securely from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["apikey"]

# Set up LLM with memory
llm = OpenAI(temperature=0.7)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Streamlit UI
st.title("🧠 AI-Powered Q&A System")
st.write("Ask me anything, and I'll try to help!")

# Input field
user_input = st.text_input("Your question")

# When user enters a question
if user_input:
    response = conversation.run(user_input)
    st.write("🤖", response)

    with st.expander("🧾 Conversation History"):
        st.info(memory.buffer)
