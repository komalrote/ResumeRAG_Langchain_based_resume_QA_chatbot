import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1️⃣ HuggingFace model & token
model_name = "mistralai/Mistral-7B-Instruct-v0.1"


# 2️⃣ Load model and tokenizer (once)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    device_map="auto",
    dtype="auto"
)


# 3️⃣ Create a text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=16
)

# 4️⃣ Wrap pipeline in LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# 5️⃣ Load environment variables
load_dotenv()

# 6️⃣ Load resume PDF
loader = PyPDFLoader(r"D:\ResumeRAG – LangChain-based Resume Q&A Chatbot\Komal_rote_cv.pdf")
docs = loader.load()

# 7️⃣ Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(docs)

# 8️⃣ HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 9️⃣ Vector database
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 🔟 Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 1️⃣1️⃣ QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# 1️⃣2️⃣ Ask a question
query = "What machine learning projects Komal has worked on?"
print("Q:", query)
print("A:", qa.run(query))
