import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS,Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq

from langchain_objectbox.vectorstores import ObjectBox

from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
google_api_key = os.environ["GOOGLE_API_KEY"]



st.title("CHAT BOT Q&A")

llm = ChatGroq(
            temperature=1, 
            groq_api_key = os.getenv('GROQ_API_KEY'), 
            model_name="llama3-8b-8192"
        )

prompt = ChatPromptTemplate.from_template(
    
""" 
Answer the question based on the provided context
Think step by step before providing a detailed answer
<context>
{context}
</context>
Question: {input} 

""")

# vector embedding and object box vectordb
def vector_embedding():

    if "vector" not in st.session_state:

        st.session_state.embedding = GooglePalmEmbeddings()

        loader = PyPDFDirectoryLoader("./us_census")

        st.session_state.docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

        st.session_state.final_docs = text_splitter.split_documents(st.session_state.docs[0:30])

        st.session_state.vector = ObjectBox.from_documents(st.session_state.final_docs,st.session_state.embedding,embedding_dimensions=768)

        


input_prompt = st.text_input("Enter your question for the documnets")

if st.button("Document embedding"):

    vector_embedding()

    print("embeddings are created")

    st.write("The Embeddings are created")

if input_prompt:

    document_chain = create_stuff_documents_chain(llm,prompt)

    retriver = st.session_state.vector.as_retriever()

    retriever_chain = create_retrieval_chain(retriver,document_chain)

    response = retriever_chain.invoke({"input":input_prompt})

    st.write(response["answer"])

    with st.expander("Document Similarity search"):

        for i,docs in enumerate(response["context"]):

            st.write(docs.page_content)

            st.write("-------------------------------")










