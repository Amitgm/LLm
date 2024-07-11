import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GooglePalmEmbeddings,OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_google_genai import GoogleGenerativeAI
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from groq import Groq

from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
print("groq api key",groq_api_key)


os.environ['GOOGLE_API_KEY']  = 'AIzaSyAb_DlJVnyH43uaxNI7LzbkqFuGQnXLO44'

if "vectors" not in st.session_state:

    st.session_state.embedding = GooglePalmEmbeddings()

    st.session_state.loaders = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")

    st.session_state.docs = st.session_state.loaders.load()

    st.session_state.doc_chunker  = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    st.session_state.final_documents = st.session_state.doc_chunker.split_documents(st.session_state.docs[:2])

    # embedding = OllamaEmbeddings()
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embedding)

# print("exited here")

llm = ChatGroq(
            temperature=1, 
            groq_api_key = os.getenv('GROQ_API_KEY'), 
            model_name="mixtral-8x7b-32768"
        )

# llm = ChatGroq(grok_api_key=groq_api_key,model_name="gemma-7b-it")

# llm = Ollama(model="llama2")

st.title("chat model")

# llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key="AIzaSyAb_DlJVnyH43uaxNI7LzbkqFuGQnXLO44", temperature=0.1)


print("the llm",llm)


prompt = ChatPromptTemplate.from_template(
    
""" 
Answer the question based on the context.
Please provide most accurate response.
<context>
{context}
<context>
Question:{input}
                                          
""")


retriever = st.session_state.vectors.as_retriever()
document_chain = create_stuff_documents_chain(llm,prompt)

retrival_chain = create_retrieval_chain(retriever,document_chain)
# 
prompt = st.text_input("Input your prompt here")

if prompt:

    start = time.process_time()
    response = retrival_chain.invoke({"input":prompt})
    print("the process time",time.process_time()-start)

    st.write(response["answer"])

    # with a streamlit expander

    st.write("Below are the answers taken from the context")

    for i,doc in enumerate(response["context"]):

        st.write(doc.page_content)

        st.write("----------------------------------")

# print("the final response",response["answer"])









