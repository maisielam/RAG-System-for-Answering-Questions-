# %% FastAPI app initialization and imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warning

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # CORS middleware for frontend/backend communication
from langserve import add_routes #turn LangServe into web APIs

# Import internal modules 
from llm_model import get_hf_llm
from main import build_rag_chain, InputQA, OutputQA #Functions and data models for RAG and API input/output

# %% App setup
genai_chain = None  #Placeholder for the RAG chain

#Initialize FastAPI app with metadata
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using Langchain's Runnable interfaces",
)

#Allow API to be accessed from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Allow any website to call API
    allow_credentials=True,         # Allow credentials (cookies, authorization headers, etc.) to be sent
    allow_methods=["*"],            # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],           
    expose_headers=["*"]            
)

# %% Startup event to load model and build the chain
@app.on_event("startup")
async def load_chain(): #function can run in the background without stopping other things
    """
    Called when FastAPI server starts.
    Loads the LLM, reads documents, and build RAG chain, expose the chain as a REST API endpoint via LangServe.
    """
    global genai_chain
    llm = get_hf_llm(temperature=0.7)  # Load LLM model 
    genai_docs = "./data_source/generative_ai" 
    genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")  # Build RAG pipeline

    # Add interactive LangServe playground to turn into onlin API (A simple web form to ask questions)
    add_routes(
        app,
        genai_chain,
        playground_type="default",
        path="/generative_ai"
    )

# %% test if the app is working
@app.get("/check")
def check():
    return {"status": "ok"}

# %% RAG question-answering 
@app.post("/generative_ai", response_model=OutputQA)
def generative_ai(inputs: InputQA):
    """
    - Accepts a POST request with a question (InputQA)
    - Passes the question through the RAG pipeline (LLM + Retriever)
    - Returns the generated answer in a structured format (OutputQA)
    """
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}
