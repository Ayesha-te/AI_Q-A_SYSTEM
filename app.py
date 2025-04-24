import os
import streamlit as st
import toml
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["apikey"]



# Streamlit setup
st.set_page_config(page_title="AI Q&A")
st.title("🤖 AI-Powered Q&A System with Memory")

# Initialize memory safely
if "memory" not in st.session_state:
    memory = ConversationBufferMemory(return_messages=True)
    st.session_state.memory = memory
else:
    memory = st.session_state.memory

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    
)

# Set up the conversation chain
qa_chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Input form
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question:", placeholder="Type here...")
    submitted = st.form_submit_button("Send")

# If question submitted
if submitted and user_input:
    response = qa_chain.predict(input=user_input)
    st.write("🧠 **AI:**", response)

# Show memory / chat history
with st.expander("💬 Conversation History"):
    for msg in memory.chat_memory.messages:
        role = "You" if msg.type == "human" else "AI"
        st.markdown(f"**{role}:** {msg.content}")

# Optional reset button
if st.button("🔁 Reset Conversation"):
    memory.clear()
    st.experimental_rerun()
