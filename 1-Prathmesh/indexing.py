import os 
from langchain_mistralai.chat_models import ChatMistralAI

def setup_llm(): 
    llm = ChatMistralAI(api_key="")
    return llm


model=setup_llm()
print(model.invoke("Hello! How can I assist you today?"))

