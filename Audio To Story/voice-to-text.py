
from transformers import pipeline
import soundfile as sf
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools import WikipediaQueryRun
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain import hub
from IPython.display import Audio,display
from dotenv import load_dotenv
import os


load_dotenv()

audio_path = r"C:\Users\Amit\Documents\voice-to-text-RAG\MemoirsVictorHugo.mp3"


# CONVERTING VOICE TO TEXT
# def convert_voice_to_text(audio_path):

#     model_name = "openai/whisper-medium"

#     whisper = pipeline("automatic-speech-recognition", model=model_name)

#     data, samplerate = sf.read(audio_path)

#     text = whisper(data)

#     return text

# # DISPLAYING THE AUDIO
# def display_audio(audio_path):
#     # Display the audio player
#     display(Audio(audio_path,autoplay=True))


# # RUNNING THE CODE TO CONVERT THE AUDIO TO TEXT AND DISPLAY THE AUDIO
# text = convert_voice_to_text(audio_path)
# display_audio(audio_path)

# with open("/content/drive/MyDrive/datasets1/audio-to-text.txt","w") as f:

#   f.write(text["text"])

# f.close()


# AUDIO TO TEXT CONVERSION
with open(r"C:\Users\Amit\Documents\voice-to-text-RAG\audio-to-text.txt","r") as f:

    text = f.read()

f.close()

os.environ["OPENAI_API_KEY"] =  os.getenv("OPENAI_API_KEY")

# openai_api_key = os.getenv("OPENAI_API_KEY")

# os.environ["OPENAI_API_KEY"] = str(openai_api_key)

# DEFINING THE TOOLS FOR MORE ELABORATE ANSWERS BASED OF ON THE AUDIO SIGNAL
class Tool:
     
    api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=100)

    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = hub.pull("hwchase17/openai-functions-agent")



tools = Tool()

# SPLIT THE TEXTS
def split_texts(dataset):

    splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=30)

    texts = splitter.split_text(dataset)

    return texts

speech_text = split_texts(text)

# CREATING VECTOR STORES

def create_vector_stores(speech_text):

       vector_stores =  FAISS.from_texts(speech_text,embedding=OpenAIEmbeddings())

       retriever = vector_stores.as_retriever()

       return retriever


retriever = create_vector_stores(speech_text)

# CREATING CHAINS
def create_chain(retriever):

    prompt = ChatPromptTemplate.from_template("""
        Answer the questions as best as you can, given the audio sample extracted, take sources from outside as well to enrich your answer (most of the transcript is philosphical)
        <context>
        {context}
        </context>     
        question : {input}
    """)

    document_chain = create_stuff_documents_chain(tools.llm,prompt)

    retriever_chain = create_retrieval_chain(retriever,document_chain)

    return retriever_chain

# CREATING AN ANSWER
def create_doc_answer(text):
     
    retriever_chain = create_chain(retriever)

    print(retriever_chain.invoke({"input":text})["answer"])

# CREATING AN CREATIVE ANSWER
def create_creative_answer(text):
         

    retriever_tool = create_retriever_tool(retriever,"Audio","Audio related to a small audio Transcript")

    all_tools = [tools.wiki_tool,retriever_tool]

    agent = create_openai_tools_agent(tools.llm,all_tools,tools.prompt)
    
    agent_executor = AgentExecutor(agent=agent,tools=all_tools,verbose=True)

    print(agent_executor.invoke({"input":text})["output"])


# CREATING ANSWERS BASES ON THE DOCUMENT INFORMATION
create_doc_answer(text="blood chains in relation to the document, give me a more philophical answer")

# CREATING CREATIVE ANSWERS USING AGENT TOOLS
create_creative_answer(text="blood chains in relation to the document, give me a more philophical answer")







