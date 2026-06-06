# Chaty_PDFs

Chaty_PDFs is an advanced AI-powered chatbot that revolutionizes document interaction. Instead of manually searching through hundreds of pages, users can upload PDF documents and engage in natural conversations to extract insights, summaries, and answers.

Built on FastAPI, LangChain, LLaMA, HuggingFace embeddings, and ChromaDB, the system intelligently extracts knowledge from PDFs, stores it in a semantic vector database, and delivers context-aware responses through natural language processing.

---

## Why Chaty_PDFs

Traditional PDF tools rely on basic keyword search, which fails when users need:
- Context spanning multiple sections
- Summaries of complex topics
- Instant answers from large documents

Chaty_PDFs addresses these limitations through:

**Context Understanding** – Grasps semantic meaning, not just keyword matches  
**Precision Retrieval** – Identifies exact sections, provides summaries, and explains concepts  
**Scalability** – Handles thousands of pages through embeddings and vector databases  
**Natural Interaction** – Delivers real-time, conversational responses  
**Extensibility** – Designed to support future features like images, graphs, and code analysis

This is not just a chatbot. It's a personal AI knowledge assistant.

---

## Architecture

### Folder Structure

```
Chaty_PDFs/
│
├── backend/
│   ├── main.py               # FastAPI backend (core API for chat & PDF upload)
│   ├── extractor.py          # PDF → Text → Embeddings → ChromaDB
│   ├── requirements.txt      # Python dependencies
│   ├── README.md             # Backend-specific documentation
│   │
│   ├── Pdfs/                 # PDF document directory
│   │
│   ├── static/               # Frontend served by FastAPI
│   │   └── index.html        # Chat interface (Tailwind CSS + JS)
│   │
│   ├── pdf_files/            # Temporary storage for uploaded PDFs
│   ├── chroma_db/            # Vector database (stores embeddings)
│   └── logs/                 # Debugging & query logs
│
├── .gitignore                # Ignore cache, venv, logs, DB files
└── README.md                 # Project-level documentation
```

### Component Overview

**backend** – Core engine containing the FastAPI backend, which runs APIs for document chat, search queries, and PDF upload/processing.

**static** – Clean, responsive web interface built with Tailwind CSS and JavaScript for real-time PDF upload and chatbot interaction.

**pdf_files** – Temporary storage for uploaded PDFs. Once processed and converted into embeddings, files can be referenced or cleared.

**chroma_db** – Vector database storing all embeddings, metadata, and semantic representations of documents for efficient semantic search and retrieval.

**logs** – Storage for debugging details, query traces, and error logs. Essential for monitoring system performance and troubleshooting.

---

## Technical Architecture

The system operates in four distinct stages:

### 1. Document Ingestion

When a user uploads a PDF through the frontend, PyMuPDF extracts raw text page by page, preserving document structure and readability. Since feeding an entire PDF into the model would be inefficient and imprecise, the extracted text is split into semantic chunks of approximately 650 characters with a small overlap. This chunking ensures that:
- Embeddings remain contextually meaningful
- Important sentences are not cut off
- Queries can precisely target specific sections for accurate retrieval

### 2. Embedding Generation

Each text chunk is transformed into a 384-dimensional vector using the HuggingFace model `sentence-transformers/all-MiniLM-L6-v2`. These embeddings capture semantic meaning rather than just keyword matches, enabling context-aware retrieval. Embeddings are normalized to ensure stable and accurate similarity scoring. 

Semantically similar sentences like "AI is a branch of ML" and "Machine learning includes AI" are positioned closely in vector space, allowing the system to recognize them as related concepts even with different wording.

### 3. Knowledge Storage in ChromaDB

Each embedding is saved in ChromaDB along with associated metadata such as book name, page number, and chunk ID.

Example entry:
```json
{
  "id": "uuid1234",
  "book_name": "DeepLearningIntro",
  "page_number": 5,
  "content": "Deep learning is a subset of ML using neural networks..."
}
```

This structured storage ensures every vector can be traced back to its exact location in the document. Since the database is persisted locally in `/chroma_db/`, previously processed documents do not need re-embedding, improving efficiency and scalability.

### 4. Semantic Search + Retrieval-Augmented Generation (RAG)

When a user submits a query:
1. The query is embedded into a vector using the same HuggingFace model
2. ChromaDB compares the query vector against stored embeddings using cosine similarity
3. The most semantically relevant chunks are retrieved
4. Retrieved context is combined with the original query and fed into the LLaMA model (running locally as a GGUF quantized version)
5. LLaMA generates a natural, accurate, and context-aware response

This approach delivers answers that go beyond simple keyword search by understanding user intent and document context.

---

## Detailed Embedding Process

### Text Extraction and Chunking

After PDF upload, raw text is extracted using PyMuPDF, which handles complex PDFs including images and formatting. The text is broken into manageable semantic chunks using LangChain's RecursiveCharacterTextSplitter. Default chunk size is approximately 650 characters with a 50-character overlap to preserve context across boundaries. This ensures each chunk is meaningful, self-contained, and ready for embedding.

### Embedding Step

Each text chunk is transformed into a dense vector representation using a HuggingFace embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`). Unlike keyword search, embeddings capture semantic meaning—so chunks about "neural networks" and "deep learning" appear close together in vector space, even without shared exact words. This allows the system to truly understand content instead of just matching strings.

### Storage in ChromaDB

Generated embeddings are stored in ChromaDB, a specialized vector database. Along with embedding vectors, metadata such as document name, page number, and unique chunk IDs is saved. This enables efficient retrieval. One PDF page often becomes multiple chunks, each stored as a separate searchable entry in ChromaDB, ensuring even very large documents remain searchable at fine granularity.

### Query Processing

When a user submits a query, the same embedding model encodes the question into a vector. ChromaDB performs semantic similarity search by comparing the query vector with all stored vectors using cosine similarity. The database returns top-matching chunks along with metadata (document name, page number, confidence score). These chunks are passed to the LLaMA model, which uses them as context to generate a natural, context-aware answer.

### Pipeline Summary

```
PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Semantic Search → LLaMA Answer
```

---

## Workflow Diagram

```
[ PDF Upload ]
       ↓
[ Text Extraction → Chunking ]
       ↓
[ HuggingFace Embeddings ]
       ↓
[ ChromaDB Vector Store ]
       ↓
User Question → [ Embedding ] → [ Semantic Search ] → [ LLaMA Model ] → Answer
```

---

## Technology Stack

**Backend**  
- FastAPI
- LangChain
- HuggingFace Embeddings
- LLaMA.cpp
- ChromaDB

**Frontend**  
- Tailwind CSS
- Vanilla JavaScript (dark mode, responsive UI)

**PDF Processing**  
- PyMuPDF (fitz)

**Infrastructure**  
- Local ChromaDB
- Caching
- Modular design

**Additional Tools**  
- tqdm (progress bars)
- Logging (debug & query monitoring)

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/Chaty_PDFs.git
cd Chaty_PDFs/backend
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Build the Knowledge Base

Run the extractor to embed your PDFs into ChromaDB:

```bash
python extractor.py
```

### Run the Application

```bash
uvicorn main:app --reload
```

### Access the Interface

Open your browser and navigate to:

```
http://127.0.0.1:8000
```

---

## Use Cases

**Education** – Chat with textbooks, research papers, and lecture slides

**Enterprise** – Query company policies, contracts, and compliance documents

**Legal** – Extract insights from case laws, agreements, and regulations

**Healthcare** – Summarize clinical research, guidelines, and medical papers

**Research & Media** – Digest long reports and archives in minutes

---

## Roadmap

**Multi-PDF Queries** – Support for querying across multiple documents simultaneously

**Voice-Enabled Queries** – Voice input for hands-free interaction

**Cloud Deployment** – One-click deployment to AWS, Railway, or Heroku

**Multi-Modal Input** – Support for text, images, and graphs

**User Management** – Role-based authentication and saved conversation history

---

## License

MIT License – Free to use, modify, and distribute.

---

## Acknowledgments

- [LangChain](https://www.langchain.com/)
- [LLaMA.cpp](https://github.com/ggerganov/llama.cpp)
- [ChromaDB](https://www.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)

---

Chaty_PDFs represents a step toward the future of human-AI collaboration with knowledge.
