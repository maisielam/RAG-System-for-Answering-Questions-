# RAG System for Answering Questions from Research Papers

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system to answer user questions using information extracted from research papers. The system loads PDFs, splits their content into chunks, indexes them using a vector database, and uses a HuggingFace-based language model to generate answers.

The primary goal is to build a question-answering system that can retrieve relevant information from domain-specific research articles and respond accurately to user queries.

## Project Structure
```
rag/
├── data_source/                  # Folder containing input PDFs 
│   └── generative_ai/
│       └── download.py          # Downloads sample academic papers in PDF format
├── src/
│   └── app.py               # FastAPI app for serving the RAG system
│   └── file_loader.py       # Loads and chunks PDF text using PyPDFLoader
│   └── llm_model.py         # Loads the HuggingFace model (phi-1.5)
│   └── main.py              # Builds the RAG chain using retriever and LLM
│   └── offline_rag.py       # Core logic for Prompt + Retrieval + Parsing
│   └── utils.py             # Utility functions for post-processing
│   └── vectorstore.py       # Initializes vector database (Chroma/FAISS)
├── env/                         # Virtual environment directory
├── requirements.txt             # Python dependencies
```

## Features
- Loads and chunks PDF files using `PyPDFLoader`
- Retrieves relevant chunks using Chroma/FAISS + HuggingFace embeddings
- Generates answers with LLM (microsoft/phi-1.5)
- Serves with FastAPI and LangServe Playground for interactive testing

## Setup Instructions
1. **Clone the repo & activate your environment**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   conda activate yourenvname
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   uvicorn src.langchain.app:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Visit**:
   - Swagger API: http://localhost:8000/docs
   - LangServe playground: http://localhost:8000/generative_ai/playground/

## Model
- Model used: `microsoft/phi-1.5` via HuggingFace pipeline
- Vector DB: `Chroma` or `FAISS` (pluggable)
- Embeddings: `HuggingFaceEmbeddings`

## Evaluation _(In Progress)_
Evaluation metrics include:
- Precision@k
- Mean Reciprocal Rank (MRR)

Evaluate by comparing:
- Model answers to gold answers
- Whether the retrieved context truly supports the response

## Baselines _(In Progress)_
- Keyword Matching
- TF-IDF + Cosine Similarity retrieval
These provide simple reference points to compare neural performance.



