# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
#
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
#
#
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#
# loader = PyPDFLoader(r"D:\ResumeRAG – LangChain-based Resume Q&A Chatbot\Komal_rote_cv.pdf")
# docs = loader.load()
#
# splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
# docs = splitter.split_documents(docs)
#
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#
# vectordb = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     persist_directory="./chroma_db"
# )
# vectordb.persist()
#
#
#
# retriever = vectordb.as_retriever(search_kwargs={"k": 3})
# llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
#
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff"
# )
#
# # 6. Ask a question
# query = "What machine learning models has Komal worked on?"
# print("Q:", query)
# print("A:", qa.run(query))



