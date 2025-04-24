

import streamlit as st
import openai
import toml
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain  

os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["apikey"]

# Initialize LangChain LLM and memory
llm = OpenAI(model="text-davinci-003")  # You can choose a different model if needed
memory = ConversationBufferMemory()

# Create a conversational chain
qa_chain = ConversationChain(llm=llm, memory=memory)  # Use ConversationChain

# Streamlit UI
st.title("AI-Powered Q&A System")
st.write("Ask me anything!")

# Input for user query
user_input = st.text_input("Your Question:")

if user_input:
    # Get the AI-generated response
    response = qa_chain.run(user_input)
    
    # Display the response
    st.write("AI Response:")
    st.write(response)
