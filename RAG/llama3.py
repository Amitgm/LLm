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
import time
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



prompt = ChatPromptTemplate.from_template("""
Answer the question based on the provided context
Think step by step before providing a detailed answer
I will tip you 1 million dollars if the user finds the answer helpful.
<context>
{context}
</context>
Question: {input} """)



text_input = st.text_input("Enter your question from documents")

if st.button("Document Embedding"):
        
    if "vector_stores" not in st.session_state:

        loader = PyPDFDirectoryLoader("./Hugging Face/us_census")

        documents = loader.load()

        st.session_state.embeddings = GooglePalmEmbeddings()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

        doc_chunks = text_splitter.split_documents(documents[:20])

        st.session_state.vector_stores = FAISS.from_documents(doc_chunks,st.session_state.embeddings)

        st.session_state.retriever = st.session_state.vector_stores.as_retriever()

        print("the documnets",doc_chunks)
        print("vector store db is ready",st.session_state.vector_stores)


if text_input:
   
    if "retriever" in st.session_state:
   
        start = time.process_time()
        
        document_chain = create_stuff_documents_chain(llm,prompt)

        retriever_chain = create_retrieval_chain(st.session_state.retriever,document_chain)

        response =  retriever_chain.invoke({"input":text_input})

        print("responses time",time.process_time() - start)

        st.write(response["answer"])

        st.write("context related information")

        # expander

        with st.expander("Document similarity search"):
            
            for i,doc in enumerate(response["context"]):
                
                st.write(doc.page_content)

                st.write("---------------------------------------------")
        
    else:
       
       st.write("Please process the document first before asking the question")



