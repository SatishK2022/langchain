import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Page Config
st.set_page_config(
    page_title="Q&A Chatbot",
    page_icon="👽",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title
st.title("👽 Q&A Chatbot")
st.text("This is a simple Q&A chatbot using Langchain and Groq.")

# Sidebar
with st.sidebar:
    st.header("Settings")

    # API Key
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get your Groq API key from https://groq.io/",
    )

    # Models Selection
    model_name = st.selectbox("Model", ["gemma2-9b-it", "llama-3.1-8b-instant"], index=0)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LLM chain
@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None

    # Initialize Groq LLM
    llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0.7, streaming=True)

    # Define chat prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant powered by Groq. Answer questions clearly and concisely."),
        ("user", "{question}")
    ])

    # Create full chain
    chain = prompt | llm | StrOutputParser()
    return chain

# Get chain
chain = get_chain(api_key, model_name)

if not chain:
    st.warning("Please provide your Groq API key to use the chatbot.")
    st.markdown("Get your Groq API key from [https://groq.io/](https://groq.io/).")

else:
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask me anything"):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Stream the response using correct input format
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Examples
st.markdown("---")
st.markdown("### 📚 Try These Examples:")
col1, col2 = st.columns(2)

with col1:
    st.markdown("- What is Langchain?")
    st.markdown("- Explain the difference between Langchain and OpenAI?")

with col2:
    st.markdown("- How do I get started with Langchain?")
    st.markdown("- Write a haiku about AI")

# Footer
st.markdown("---")
st.markdown("Powered by [Langchain](https://langchain.com/) & [Groq](https://groq.io/)")
