__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()
st.set_page_config(page_title="Chatbot", page_icon="🤖")


@st.cache_resource
def load_data():

    # Text loader
    loader = PyPDFDirectoryLoader("./pdf")
    data = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    # Text Splitter
    split_documents = text_splitter.split_documents(data)

    texts = [doc.page_content for doc in split_documents]

    # Embeddings
    embeddings_model = OpenAIEmbeddings()

    # Vector Store
    db = Chroma.from_texts(
        texts,
        embeddings_model,
    )

    return db


# Load data and create vector store
db = load_data()

st.title("Ask to Sooyong")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "bot",
            "content": "안녕하세요 저는 신수용 입니다. 제게 궁금한 점들을 물어보세요",
        }
    ]

chat_messages = st.session_state.chat_messages


def display_chat_messages(chat_messages):
    chat_html = """
    <div class="chat-container">
    """
    for msg in chat_messages:
        if msg["role"] == "user":
            chat_html += f"<div class='user-message'>{msg['content']}</div>"
        elif msg["role"] == "bot":
            chat_html += f"<div class='bot-message'>{msg['content']}</div>"
    chat_html += "</div>"
    return chat_html


with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

chat_container = st.empty()
chat_container.markdown(display_chat_messages(chat_messages), unsafe_allow_html=True)

# Create columns for text input and button
user_input = st.text_input("")
if st.button("전송"):
    if user_input:
        chat_messages.append({"role": "user", "content": user_input})
        st.session_state.chat_messages = chat_messages

        chat_container.markdown(
            display_chat_messages(chat_messages), unsafe_allow_html=True
        )

        # Show spinner
        spinner_placeholder = st.empty()
        with spinner_placeholder:
            with st.spinner("답변을 작성중입니다..."):
                # Retrieve response from the model
                llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
                result = qa_chain(
                    {
                        "query": """
                        Imagine you are Sooyong Shin. The user will send you a message. Please respond as Sooyong Shin would, in a formal tone and conversational manner. 
                        If the user asks about your studies or achievements, provide detailed, personal insights.
                        Make sure the length does not exceed 200 characters unless essential information needs to be included.
                        If someone ask question that you didn't know, please respond that the question is too personal.
                        Do not include a formal opening or closing. User:
                        """
                        + user_input
                    }
                )

        # Get the response from the model
        answer = result["result"]

        chat_messages.append({"role": "bot", "content": answer})
        st.session_state.chat_messages = chat_messages

        chat_container.markdown(
            display_chat_messages(chat_messages), unsafe_allow_html=True
        )

        st.session_state.user_input = ""
        spinner_placeholder.empty()  # Clear the spinner
    else:
        st.write("질문을 입력해주세요")
