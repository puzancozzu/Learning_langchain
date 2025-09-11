import os
from apikey import apikey
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


# set Gemini API key
os.environ["GOOGLE_API_KEY"] = apikey

# app framework
st.title("🦜🛠️ Youtube GEMINI Creator")
prompt = st.text_input("Enter your prompt here")

# Prompt Templates
title_template = PromptTemplate(
    input_variables=["topic"],
    template="Write me a youtube video title about {topic}. " \
    "Just a one line topic just one line only one line topic " \
    "that you think sutiable for youtube video title",
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template="Write me a youtube video script based on the title TITLE : {title},"
             " while leveraging this wikipedia research: {wikipedia_research} "
             "Make it detailed and long with a hook and intro and outro. "
             "Use a friendly tone like a youtube video script, and keep it concise but detailed enough."
)

# Memory 
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")

# LLM (Gemini) - instance of the model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7
)

#llm chains
title_chain = LLMChain(
    llm=llm, 
    prompt=title_template, 
    verbose=True, 
    output_key="title",
    memory=title_memory
    )

script_chain = LLMChain(
    llm=llm, 
    prompt=script_template, 
    verbose=True, 
    output_key="script",
    memory=script_memory
    )

# instance of wikipedia api wrapper - tool from langchain
wiki = WikipediaAPIWrapper()

# show the response if there's a prompt
if prompt:
    title = title_chain({"topic": prompt})
    wiki_research = wiki.run(prompt)
    script = script_chain({
        "title": title['title'],
        "wikipedia_research": wiki_research
        })
    
    st.subheader("🎬 Video Title")
    st.write(title['title'])
    st.subheader("📜 Script")
    st.write(script['script'])

    with st.expander("Conversation Title History"):
        st.info(title_memory.buffer)

    with st.expander("Conversation Script History"):
        st.info(script_memory.buffer)   

    with st.expander("Wikipedia Research History"):
        st.info(wiki_research)
