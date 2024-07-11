from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st 

from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# LANGSMITH TRACKING
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# PROMPT TEMPLATE

prompt = ChatPromptTemplate.from_messages(

    [
        ("system","You are a helpful chatbot"),
        ("user","Question:{question}")

    ]

)
# streamlit framework

st.title("LANGchain demo with openapi")

input_text = st.text_input("Search the topic you want")

# open ai llm

llm = Ollama(model="gemma:2b")

output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_text:

    st.write(chain.invoke({'question':input_text}))