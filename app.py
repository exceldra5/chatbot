from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

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

# Retrieval
question = "대회 수상 실적은 뭐가 있어?"
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
result = qa_chain({"query": question})
print(result)
