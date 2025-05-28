#%%
"""Set up the RAG chain for question answering using a pre-trained LLM."""

from pydantic import BaseModel, Field

from file_loader import Loader #loads and splits files into text chunks
from vectorstore import VectorDB #stores embeddings and performs similarity search
from offline_rag import Offline_RAG #builds RAG chain that queries the LLM using retrieved chunks

class InputQA (BaseModel): #Define the input schema using Pydantic
    question: str = Field (..., title="Question to ask the model") #title: used for UI display

class OutputQA (BaseModel) :
    answer: str = Field(..., title="Answer from the model")

def build_rag_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2) #load and split files into text chunks, load files in parallel using 2 workers
    retriever = VectorDB(documents = doc_loaded).get_retriever() #Store documents in a vector store and return a retriever
    rag_chain = Offline_RAG(llm).get_chain(retriever) #Build the RAG chain that queries the LLM using retrieved chunks
    return rag_chain
# %%
