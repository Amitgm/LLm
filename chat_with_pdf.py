import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#import google.generativeai as palm
## IMPORTING THE EMBEDDINGS
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm 
## FAISS-CPU IS USED AS A VECTOR DATABASE
from langchain.vectorstores import FAISS 
## THIS KEEPS MEMORY OF THE CONVERSATIONS history, has a memeory componenet
from langchain.chains import ConversationalRetrievalChain

from langchain.memory import ConversationBufferMemory

import os 

global start_point


os.environ['GOOGLE_API_KEY']  = 'AIzaSyAb_DlJVnyH43uaxNI7LzbkqFuGQnXLO44'


def pdf_reader(pdf_docs):

    text = ''

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:

            text+=page.extract_text()
    
    return text

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    #print("this is the text splitter",text_splitter)

    chunks = text_splitter.split_text(text)

    #print("chunks are",chunks)

    return chunks

def get_vector_store(text_chunks):

    embeddings = GooglePalmEmbeddings()

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)



    return vector_store 

def get_conversational_chain(vector_store):

    llm= GooglePalm()

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

    conversational_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(),memory=memory)

    return conversational_chain

## GETTING THE RELEVANT FACTS OF THE DOCUMENT
def primary_questions():
    
    

    questions = ['Give a brief introduction of the document?',
                 'what is the purpose of the document,']
    
   
    answers = []
    
    for ques in questions:

        response =  st.session_state.conversation({"question":ques})

        st.session_state.chatHistory = response['chat_history']

        ## to get the current response of the chat hitory

        answers.append(st.session_state.chatHistory[-1].content)


 
    st.session_state.start_point = len(st.session_state.chatHistory)

    st.session_state.answers = answers
    
    return answers

                



## GETTING THE USER INPUT OF THE DOCUMENT
def user_input(user_question):

    

    response =  st.session_state.conversation({"question":user_question})
    
    st.session_state.chatHistory = response['chat_history']

    start_point = st.session_state.start_point

    answers = st.session_state.answers

    for ans in answers:

        st.write(ans)


    for i,message in enumerate(st.session_state.chatHistory):

        #st.write(i, message)

        if i >= start_point:

            if i%2 == 0:

                st.write("Human: ",message.content)
            
            else:

                st.write("Bot: ",message.content)


def main():

    global start_point

    answers = []

    st.set_page_config("Chat with pdf's")
    st.header('chat with multiple pdfs')

    user_question = st.text_input("Ask a question from the PDF files")

    if 'conversation' not in st.session_state:

        st.session_state.conversation = None

    if 'chatHistory' not in st.session_state:

        st.session_state.chatHistory = None
    
    if 'answers' not in st.session_state:

        st.session_state.answers = None

    
    if user_question:

        user_input(user_question)



    with st.sidebar:

        st.title('upload your documents')
    

        pdf_docs = st.file_uploader('upload your pdf files AND click on porcess button',accept_multiple_files=True)

        if st.button("Process"):

            with st.spinner('Processing'):

                raw_text = pdf_reader(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vector_store = get_vector_store(text_chunks)

                st.session_state.conversation = get_conversational_chain(vector_store)

                st.success("Done")

                answers  = primary_questions()

        
        else:

            answers = None

    if answers!=None:   

        st.write("Basic Document Information")

        for ans in answers:  

            st.write(ans)
    


        


if __name__ == "__main__":

    main()
