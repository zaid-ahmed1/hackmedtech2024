# from langchain_ollama import OllamaEmbeddings
import os
import dotenv
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

def get_embedding_function():
    embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
    )
    return embeddings
