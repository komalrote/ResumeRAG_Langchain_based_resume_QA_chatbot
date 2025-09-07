import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from pygpt4all import GPT4All_J

# 1️⃣ Load your resume PDF
loader = PyPDFLoader(r"D:\ResumeRAG – LangChain-based Resume Q&A Chatbot\Komal_rote_cv.pdf")
docs = loader.load()

# 2️⃣ Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(docs)

# 3️⃣ Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4️⃣ Vector store
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 5️⃣ Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 6️⃣ Load small local GPT4All model (CPU friendly)
# Download: https://gpt4all.io/models/gpt4all-lora-quantized.bin (~1 GB)
llm = GPT4All_J(r"D:\Models\gpt4all-lora-quantized.bin")

# 7️⃣ QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# 8️⃣ Ask questions
query = "What machine learning models has Komal worked on?"
print("Q:", query)
print("A:", qa.run(query))
