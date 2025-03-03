import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GooglePalmEmbeddings,OllamaEmbeddings
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

os.environ['GOOGLE_API_KEY']  = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


loaders = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")

docs = loaders.load()
# separators: A list of separators (e.g., ["\n\n", "\n", " ", ""]) used to split the text.

# Recursive Splitting: If the resulting chunks are still too large, it recursively splits them further using the next separator in the list.

doc_chunker  = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents = doc_chunker.split_documents(docs[:2])


embedding = GooglePalmEmbeddings()
# embedding = OllamaEmbeddings()

print(final_documents)
print("before vectors")

vectors = FAISS.from_documents(final_documents,embedding)

print(vectors)

print("enterd here hello hi")

# print("exited here")

llm = ChatGroq(grok_api_key=os.getenv("GROQ_API_KEY"),model_name="gemma-7b-it")


prompt = ChatPromptTemplate.from_template(
    
""" 
Answer the question based on the context.
Please provide most accurate response.
<context>
{context}
<context>
Question:{input}
                                          
""")

retriever = vectors.as_retriever()
document_chain = create_stuff_documents_chain(llm,prompt)

retrival_chain = create_retrieval_chain(retriever,document_chain)

print(retrival_chain)
# 
# prompt = st.text_input("Input your prompt here")

retrival_chain.invoke({"input":"what is langchain"})









