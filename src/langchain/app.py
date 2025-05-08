# %% FastAPI app initialization and imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warning

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# Import internal modules from the src.langchain directory
from src.llm_model import get_hf_llm
from src.main import build_rag_chain, InputQA, OutputQA

# %% App setup
genai_chain = None  # Placeholder for the RAG chain to be initialized on startup

# Initialize the FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using Langchain's Runnable interfaces",
)

# Add CORS middleware for frontend/backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],            # Allow all HTTP methods
    allow_headers=["*"],
    expose_headers=["*"]
)

# %% Startup event to load model and build the chain
@app.on_event("startup")
async def load_chain():
    """
    Called when FastAPI server starts.
    Loads the LLM, reads documents, and initializes the RAG chain.
    """
    global genai_chain
    llm = get_hf_llm(temperature=0.7)  # Load LLM model (phi-1.5)
    genai_docs = "./data_source/generative_ai"  # Path to PDF research papers
    genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")  # Build RAG pipeline

    # Add interactive LangServe playground
    add_routes(
        app,
        genai_chain,
        playground_type="default",
        path="/generative_ai"
    )

# %% Health check endpoint
@app.get("/check")
def check():
    """
    Simple endpoint to verify server is running.
    """
    return {"status": "ok"}

# %% RAG question-answering endpoint
@app.post("/generative_ai", response_model=OutputQA)
def generative_ai(inputs: InputQA):
    """
    Accepts a question and returns an answer generated from retrieved research paper chunks.
    """
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}
