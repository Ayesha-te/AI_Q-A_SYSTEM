def get_llm():
    from langchain_community.llms import Together

    try:
        # Use a valid model from the list of available models
        return Together(
            model="togethercomputer/gpt-3.5-turbo",  # Replace with a valid, available model
            temperature=0.7,
            max_tokens=256,
            together_api_key=st.secrets["TOGETHER_API_KEY"]
        )
    except Exception as e:
        st.error(f"⚠️ Failed to initialize Together API: {str(e)}")
        raise e
