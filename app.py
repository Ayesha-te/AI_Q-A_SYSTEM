import streamlit as st
import os
import time
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import traceback
def get_llm():
    from langchain_community.llms import Together

    try:
        return Together(
            model="togethercomputer/gpt-neo-2.7B",  # Use the correct model here
            temperature=0.7,
            max_tokens=256,
            together_api_key=st.secrets["TOGETHER_API_KEY"]
        )
    except Exception as e:
        st.error(f"⚠️ Failed to initialize Together API: {str(e)}")
        raise e

if "conversation" not in st.session_state:
    try:
        llm = get_llm()

        memory = ConversationBufferMemory()
        prompt = PromptTemplate(
            input_variables=["input"],
            template="You are a helpful assistant.\n\nUser: {input}\nAI:"
        )

        st.session_state.conversation = LLMChain(
            llm=llm,
            memory=memory,
            prompt=prompt
        )
    except Exception as e:
        st.error("⚠️ Could not initialize conversation chain.")
        st.text(traceback.format_exc())

# --- Streamlit UI ---
st.title("🧠 AI-Powered Q&A System")
user_input = st.text_input("Ask your question:")

if user_input:
    try:
        # Predict using the conversation chain
        response = st.session_state.conversation.predict(input=user_input)
        st.markdown(f"**AI:** {response}")
    except Exception as e:
        # Log the error for debugging
        st.error(f"❌ Error: {str(e)}")
        # Capture full traceback for debugging
        st.text(traceback.format_exc())
        # Retry logic (if needed)
        retries = 3
        for i in range(retries):
            try:
                response = st.session_state.conversation.predict(input=user_input)
                st.markdown(f"**AI (retry {i+1}):** {response}")
                break
            except Exception as retry_error:
                st.error(f"❌ Retry {i+1} failed: {str(retry_error)}")
                if i == retries - 1:
                    st.error("⚠️ All retries failed.")
