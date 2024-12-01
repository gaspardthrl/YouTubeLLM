import os
import streamlit as st


from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.document_loaders import YoutubeLoader

st.markdown(
    """
    <style>
    .stButton button[kind="primary"] {
        width: 100%;
        border: none;
        border-radius: 0;
        border-bottom: 2px solid gray;
        background-color: transparent;
        color: inherit;
        padding: 0;
        margin: 0;
        box-shadow: none;
    }
    .stButton button[kind="primary"]:focus {
        border-bottom: 2px solid rgb(255, 153, 154);
        border-radius: 0;
        background-color: transparent;
        color: rgb(255, 153, 154);
    }
    .element-container:has(#full-width-button) + div button {
        width: 100%;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set OpenAI model
OPENAI_MODEL = "gpt-4o-mini"

# Initialize chat messages for session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            "You are a helpful assistant designed to summarize and answer questions about YouTube videos."
        )
    ]

if "query" not in st.session_state:
    st.session_state.query = False

def setup_api_key():
    openai_api_key = st.text_input("Enter Your OpenAI API key.", value="", placeholder="sk_...",type="password")
    st.markdown('<span id="full-width-button"></span>', unsafe_allow_html=True)
    if st.button("Validate"):
        if openai_api_key == "":
            st.error("API key cannot be empty")
        else:
            os.environ['OPENAI_API_KEY'] = openai_api_key
            try:
                # Setup LLM with API key
                st.session_state.llm = ChatOpenAI(model=OPENAI_MODEL)
                st.success("API key validated successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}.")

def select_video():
    ytb_url = st.text_input("Enter the URL of a YouTube video.", value="")

    st.markdown('<span id="full-width-button"></span>', unsafe_allow_html=True)
    if st.button("Summarize & Ask Questions"):
        if ytb_url == "":
            st.error("Please enter a valid YouTube URL.")
            return

        try:
            # Load the YouTube transcript
            loader = YoutubeLoader.from_youtube_url(ytb_url, add_video_info=False)
            docs = loader.load()
            transcript = docs[0].page_content  # Assume single document

            st.session_state.messages.append(
                HumanMessagePromptTemplate.from_template(
                        f"Given the following video transcript:\n{transcript}\nSummarize the video."
                )
            )
            
            # Generate summary
            summary_prompt = ChatPromptTemplate.from_messages(
                st.session_state.messages
            ).format()
            
            summary_response = st.session_state.llm.invoke(summary_prompt)

            # Store the summary response
            st.session_state.messages.append(
                AIMessage(content=summary_response.content)
            )

            st.session_state.summary = summary_response

            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}.")

def qa():
    with st.container(height=500):
        for message in st.session_state.messages[::-1]:
            with st.container(border=True):
                if type(message) == type(HumanMessage("")):
                    st.markdown("##### :blue[User:]")
                    st.markdown(message.content)
                elif type(message) == type(AIMessage("")):
                    st.markdown("##### :red[Assistant:]")
                    st.markdown(message.content)
                else:
                    continue

    def clear_input():
        st.session_state.temp_input_saved = st.session_state.temp_input
        st.session_state.temp_input = ""
    
    if "temp_input_saved" not in st.session_state:
        st.session_state.temp_input_saved = ""
        
    st.text_input("You:", value="", key="temp_input", placeholder="", label_visibility="collapsed")

    st.markdown('<span id="full-width-button"></span>', unsafe_allow_html=True)
    if st.button("Send", on_click=clear_input):
        if st.session_state.temp_input_saved == "":
            st.error("You cannot ask an empty question.")
        else:
            # Add user's message to the conversation
            st.session_state.messages.append(
                HumanMessage(
                    st.session_state.temp_input_saved
                )
            )

            st.session_state.query = True
            st.rerun()
        
    if st.session_state.query:

        prompt = ChatPromptTemplate.from_messages(
                st.session_state.messages
        ).format()

        response = st.session_state.llm.invoke(prompt)

        st.session_state.messages.append(
                AIMessage(content=response.content)
        )
        st.session_state.query = False
        st.rerun()
    
# Set up the app interface
st.title("YouTubeLLM")

if "llm" not in st.session_state:
    setup_api_key()
elif "summary" not in st.session_state:
    select_video()
else:
    qa()
