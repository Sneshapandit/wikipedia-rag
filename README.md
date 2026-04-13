# Wikipedia RAG Chat Assistant

A lightweight Retrieval-Augmented Generation (RAG) chatbot built with Python, FAISS, Hugging Face models, Wikipedia, and Streamlit.

The app retrieves relevant Wikipedia content, builds a grounded answer from those sources, and shows clickable source links so users can verify the response directly on Wikipedia.

## Features

- Wikipedia-powered question answering
- FAISS vector search over embedded Wikipedia text chunks
- Sentence-transformer embeddings using `all-MiniLM-L6-v2`
- Streamlit chat interface for non-technical users
- Clickable Wikipedia source links
- Source snippets for quick verification
- Match-quality indicator (`High match`, `Possible match`, `Low match`, `No match`)
- Graceful handling for misspellings
- Graceful handling for weak topic matches
- Graceful handling for ambiguous topics
- Graceful handling for no-answer cases

## Tech Stack

- Python
- Streamlit
- FAISS
- sentence-transformers
- Hugging Face Transformers
- Wikipedia API wrapper

## Project Structure

```text
.
|-- app.py
|-- rag_model.py
|-- requirements.txt
|-- README.md
```

- `app.py`: Streamlit user interface
- `rag_model.py`: retrieval, topic matching, answer generation, and Wikipedia integration
- `requirements.txt`: Python dependencies

## How It Works

1. The app loads a small default Wikipedia knowledge base.
2. Pages are cleaned, chunked, and embedded using MiniLM.
3. Chunks are indexed in FAISS for semantic retrieval.
4. When a user asks a question, the system:
   - searches Wikipedia for likely matching page titles
   - prefers relevant topic matches
   - retrieves grounded supporting chunks
   - builds a short answer from the retrieved source text
   - shows source links and source snippets for verification
5. If no reliable match is found, the app refuses to guess and instead shows a fallback message or likely Wikipedia suggestions.

## Setup

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd Wikipedia
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the App

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## First Run Notes

- The first run may take a little time because Hugging Face models must be downloaded.
- The app also needs internet access to fetch Wikipedia content.
- After the first successful model download, later runs are faster because the models are cached locally.

## Example Questions

- What is artificial intelligence?
- How is machine learning different from deep learning?
- Who is Alan Turing?
- Tell me about the Indian Constitution
- What is natural language processing?

## Reliability Notes

This project is designed to reduce hallucinations by grounding answers in retrieved Wikipedia content and refusing weak matches when necessary.

Even so, it is still a lightweight RAG system, not a guaranteed source of truth. Users should review the linked Wikipedia sources for important or sensitive questions.

## Current Improvements Over a Basic RAG App

- Better chunk cleaning and overlap
- More conservative topic matching
- Stronger no-answer behavior
- Misspelling and near-match suggestions
- Cleaner answer formatting
- User-friendly Streamlit interface

## Limitations

- Accuracy depends on Wikipedia search quality and available pages
- Some niche or highly ambiguous topics may still return no reliable answer
- The system currently uses a lightweight local approach rather than a full reranking pipeline

## Future Improvements

- Add Wikipedia result selection before answering for ambiguous topics
- Add persistent FAISS index caching on disk
- Add evaluation tests for retrieval quality
- Add reranking for better source selection
- Add richer answer cards and follow-up question suggestions

## Sharing on GitHub

Before pushing to GitHub:

- keep `venv/` out of version control
- keep `__pycache__/` out of version control
- do not commit machine-specific secrets or cache files

Suggested commands:

```powershell
git init
git add .
git commit -m "Add Wikipedia RAG chat assistant"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Author Notes

This project is intended as a practical RAG demo focused on:

- grounded answers
- transparent source attribution
- non-technical usability
- safer handling of uncertain results
