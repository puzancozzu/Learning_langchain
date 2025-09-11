# LangChain RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built using **LangChain** and **OpenAI LLMs**.  
This project demonstrates how to integrate a knowledge base, vector database, and API to create a context-aware AI chatbot.

## Features
- Context-aware responses using RAG (retrieves relevant documents before answering)
- FastAPI backend for API endpoints
- Uses a local vector database for knowledge storage
- Easy to extend with additional data sources

## Setup
1. Clone the repository:
   ```bash
   git clone git@github.com-personal:puzancozzu/Learning_langchain.git

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. Set your OpenAI API key in apikey.py.
   ```bash
   streamlit run app.py
   
