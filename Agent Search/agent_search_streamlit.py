from langchain.tools import WikipediaQueryRun,DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain import hub
from io import StringIO
import streamlit as st


import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["ELEVEN_API_KEY"] = os.getenv("ELEVEN_API_KEY")


# WIKIPEDIA

try:
    api_wrapper = WikipediaAPIWrapper(top_k_results=5,doc_content_chars_max=500)
    wikitool = WikipediaQueryRun(api_wrapper=api_wrapper)

except Exception as e:

    wikitool = None

    print(f"An error occurred: {e}")
    # You can include any fallback logic here if needed
    # For now, we'll just pass to move forward
    
try:

    duckduckgotool = DuckDuckGoSearchRun()

except Exception as e:

    duckduckgotool = None

    print(f"An error occurred: {e}")

if wikitool == None and duckduckgotool == None:

    tools = None

elif duckduckgotool == None:

    tools = [wikitool]

elif wikitool == None:

    tools = [duckduckgotool]

else:

    tools = [wikitool,duckduckgotool]

# llm = ChatOpenAI(choose_temperature=0.5)




def main():

    
    st.header("Chat Through Agent Tools")

    prompt = hub.pull("hwchase17/openai-functions-agent")

    with st.sidebar:

        
        choose_llm = st.selectbox('Choose LLM',["llama3-8b-8192","gpt-3.5-turbo","gpt-4o","mixtral-8x7b-32768","gemma-7b-it"])

        choose_temperature = st.slider('Select a Temperature value between 0.0 and 1.0', 0.0, 1.0,0.5, step=0.1)


        if choose_llm == "mixtral-8x7b-32768" or choose_llm == "llama3-8b-8192" or choose_llm ==  "gemma-7b-it":

            llm = ChatGroq(
                    temperature=choose_temperature, 
                    groq_api_key = os.getenv('GROQ_API_KEY'), 
                    model_name=choose_llm
                )


        elif choose_llm == "gpt-3.5-turbo" or choose_llm == "gpt-4o":

            st.write("llm chosen")

            llm = ChatOpenAI(model=choose_llm,temperature=choose_temperature)


        input_prompt_text = st.file_uploader('Upload your text document file for your prompt')

        if st.button("Update prompt"):

            if input_prompt_text is not None:

                text = ""
                # Read the file and decode to a string
                content = input_prompt_text.read().decode("utf-8")
                # Use StringIO to work with the string content as a file
                stringio = StringIO(content)

                # Further processing of stringio
                # For example, reading line by line
                for line in stringio:
                   
                   text = line + text

                st.session_state.prompt = text 
                

    if "prompt" in  st.session_state:

        prompt.messages[0].prompt.template = st.session_state.prompt 


    input_prompt = st.text_input("Enter your defined prompt")
    user_query = st.text_input("Enter your user query")

    if input_prompt:

        prompt.messages[0].prompt.template = input_prompt

    # elif input_prompt_text:

        

    with st.expander("The input prompts given"):

        st.write("prompt given : {}".format(prompt.messages[0].prompt.template))

        # for tool in tools:
        
        #     st.write("Tool used : {}".f)
        st.write("Tools Used:")

        st.write(tools)

    
    
    agent = create_openai_functions_agent(llm,tools,prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools,varbose=True)

    if st.button("submit"):

        if user_query:

            # st.write(agent_executor.invoke({"input":user_query}))

            answers = agent_executor.invoke({"input":user_query})["output"]

            st.write(answers)


if __name__ == "__main__":

    main()




    





