import os
from apikey import apikey

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory

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
    input_variables=["title"],
    template="Write me a youtube video script based on the tile TITLE : {title}," \
    "Make it detailed and long with a hook and intro and outro. " \
    "Use a friendly tone like a youtube video script, and try to in consise and good " \
    "not so long not so  short just good enough script",
)

# Memory 
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")

# LLM (Gemini)
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

# instance of sequential chain - combine two chains
sequential_chain = SequentialChain(
    chains=[title_chain, script_chain],
    input_variables=["topic"],
    output_variables=["title", "script"],
    verbose=True,
)

# show the response if there's a prompt
if prompt:
    response = sequential_chain({"topic": prompt})
    st.subheader("🎬 Video Title")
    st.write(response["title"])
    st.subheader("📜 Script")
    st.write(response["script"])

    with st.expander("Conversation Title History"):
        st.info(title_memory.buffer)

    with st.expander("Conversation Script History"):
        st.info(script_memory.buffer)   
