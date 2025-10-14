import os
from langchain_community.vectorstores import Chroma
from langchain_mistralai.embeddings import MistralAIEmbeddings
import warnings
import numpy as np
warnings.filterwarnings("ignore", message="Could not download mistral tokenizer")

if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64 

    
def setup_embeddings_and_vectorstore(chunks):
    os.environ["MISTRAL_API_KEY"] = "lHcwga2vJ6yyjV470WdMIFn5hRgtMbcc"
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_store
