

from dotenv import load_dotenv, find_dotenv

import os
import keras_core as keras

# Use a pipeline as a high-level helper
from transformers import pipeline

from langchain import PromptTemplate,LLMChain

from langchain.llms import GooglePalm

import requests

from IPython.display import Audio





load_dotenv(find_dotenv())


HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# os.environ['GOOGLE_API_KEY'] = 'AIzaSyBHx7H45i7ds6Y8GO2FU6ootsFKc8HFcXc'

llm = GooglePalm()




def img_to_text(img_url):

    # image_to_Text  = pipeline("image-to-text", model="microsoft/trocr-base-handwritten")
    # image_to_Text  = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    # pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b")

    image_to_Text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


    output = image_to_Text(img_url)[0]['generated_text']

    

    return output

def generate_story(scenario):

    template = """ You are a romantic story teller , you generate a story based on a single narrative
    
    The story should be no more than 100 words

    CONTEXT: {scenario}

    STORY : 

    """

    

    prompt = PromptTemplate(template = template, input_variables=["scenario"])

    story_llm = LLMChain(llm=llm,prompt=prompt)

    story = story_llm.predict(scenario=scenario)

    return story


def text_to_speech(stories):

    # API_URL = "https://api-inference.huggingface.co/models/myshell-ai/MeloTTS-English-v2"
    # headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    # payloads ={

    #     "inputs" : stories
    # }

    # response = requests.post(API_URL, headers=headers, json=payloads)

    # print("this is the hugging face API token",HUGGINGFACEHUB_API_TOKEN)

    # with open('audio4.flac','wb') as file:

    #     file.write(response.content)

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"

    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    audio_bytes = query({
        "inputs": stories
    })

    # You can access the audio with IPython.display for example

    print("the audio bytes",audio_bytes)
    
    with open ("audio.flac",'wb') as f:

        f.write(audio_bytes)

     



# scenario = img_to_text(r'C:\Users\Amit\Documents\llm\images\model.jpg')

# stories = generate_story(scenario)

# text_to_speech(stories)

import streamlit as st

def main():

    st.header("turn image into voice")

    uploaded_file = st.file_uploader("choose an image",type="jpg")

    if uploaded_file is not None:

        bytes_data = uploaded_file.getvalue()

        with open(uploaded_file.name,'wb') as f:

            f.write(bytes_data)

        st.image(uploaded_file,caption="Image Uploaded",use_column_width =True)

        scenario = img_to_text(uploaded_file.name)

        stories = generate_story(scenario)

        text_to_speech(stories)

        with st.expander("scenario"):

            st.write(scenario)
        with st.expander("story"):

            st.write(stories)

        st.audio("audio.flac")

if __name__ == "__main__":

    main()











