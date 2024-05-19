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
    db = Chroma.from_texts(texts, embeddings_model)

    return db


db = load_data()

st.title("Ask to Sooyong")

user_input = st.text_input("신수용에 대해 궁금한 점을 물어보세요")

# Button
if st.button("전송"):
    if user_input:
        # Retrieval
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        result = qa_chain({"query": user_input})

        # Output
        st.write(result["result"])
    else:
        st.write("질문을 입력해주세요")
