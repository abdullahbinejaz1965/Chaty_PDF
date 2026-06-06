import os
import re
import logging
import asyncio
from typing import List, AsyncGenerator
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
import fitz 
import torch
from llama_cpp import Llama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document

# === CONFIGURATION: update these paths to your environment ===
PDF_DIR = os.path.abspath(r"path to pdf files") #pdate this path#
CHROMA_DIR = os.path.abspath(r"path to chroma db you will get this after indexing the pdfs in pdf extractor") #update this path#
GGUF_MODEL_PATH = os.path.abspath(r"path to gguf model") #update this path#
EMBEDDING_MODEL_NAME = r"path to embedding model you will have to use am embedding model to embed the pdf text" #update this path#

SYSTEM_PROMPT = """
You are an intelligent assistant trained to answer questions using retrieved document knowledge or your general knowledge.

Your role is to provide a detailed, essay-style answer that is:
- Comprehensive, very thorough, detailed, and informative.
- Written in 2 to 3 well-structured paragraphs.
- Organized into multiple paragraphs with clear bold headings and subheadings where applicable.
- Inclusive of explanations, examples, analysis, and relevant context. Add bullet points where needed.
- Written in a formal, academic style.
"""

logging.basicConfig(level=logging.INFO)

# === Prepare directories ===
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Models & vector DB 
llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_ctx=32768,
    n_threads=os.cpu_count(),
    n_gpu_layers=35 if torch.cuda.is_available() else 0,
    verbose=False
)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# === FastAPI app ===
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# mount static (frontend) directory - adjust if you serve HTML elsewhere
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.mount("/pdf_files", StaticFiles(directory=PDF_DIR), name="pdf_files")

chat_memory = {}

# === Utilities ===
def split_chunks(text: str, size: int = 650):
    paras = text.split("\n\n")
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) < size:
            cur += p + "\n\n"
        else:
            out.append(cur.strip())
            cur = p + "\n\n"
    if cur:
        out.append(cur.strip())
    return out

def clean_ans(x: str) -> str:
    m = re.search(r"(?i)answer:\s*(.+)", x, flags=re.DOTALL)
    return m.group(1).strip() if m else x.strip()

def format_output(raw_text: str, retrieved_docs: List[Document]) -> str:
    formatted = raw_text.strip()

    # If references already present, avoid duplicating
    if "References:" in formatted:
        return formatted
    if not retrieved_docs:
        return formatted
    references = "\n\nReferences:\n"
    seen = set()
    for doc in retrieved_docs:
        book = doc.metadata.get("book_name", "Unknown Book")
        page = doc.metadata.get("page_number", "Unknown Page")
        ref = f"{book}, Page {page}"
        if ref not in seen:
            references += f"- {book}, Page {page}\n"
            seen.add(ref)
    return formatted + references

# === Request model ===
class ChatRequest(BaseModel):
    user_id: str
    message: str
    mode: str
    memory_size: int = 1
    context_window_size: int = 10000
    pdfs: List[str] = []
    max_tokens: int = 4096

# === Endpoints ===
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Streams token-by-token output from the LLM back to client.
    Frontend should read the body stream and append chunks to the visible bot message.
    """
    user_id = req.user_id
    user_message = req.message.strip()
    mode = req.mode.lower()
    memory_size = req.memory_size
    selected_pdfs = req.pdfs

    memory = chat_memory.setdefault(user_id, [])[-memory_size:]

    docs, context_chunks = [], []
    if mode == "rag":
        # retrieve
        docs = retriever.get_relevant_documents(user_message)
        if selected_pdfs:
            docs = [d for d in docs if d.metadata.get("book_name") in selected_pdfs]
        for d in docs:
            book = d.metadata.get("book_name", "Unknown Book")
            page = d.metadata.get("page_number", "Unknown Page")
            context_chunks.append(f"[Source: {book} | Page: {page}]\n{d.page_content.strip()}")
        context = "\n\n".join(context_chunks)
    else:
        context = ""

    history = ""
    if memory:
        last_turn = memory[-1]
        history = f"User: {last_turn[0]}\nAssistant: {last_turn[1]}"

    disaster_keywords = ["role of ndma"]
    disaster_instruction = ""
    if any(kw in user_message.lower() for kw in disaster_keywords):
        disaster_instruction = (
            "\n\nIMPORTANT: Since this question is related to disaster or emergency, "
            "you must include a section titled 'Roles of Disaster Management Entities' "
            "that describes the specific roles of:\n"
            "- NDMA\n- PDMAs\n- DDMAs\n- SDMAs\n- NGOs/INGOs\n- Local Communities\n- Private Sector\n"
        )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Conversation History:\n{history}\n\n"
        f"User Question:\n{user_message}\n\n"
        f"Please provide a thorough and detailed answer."
        f"{disaster_instruction}\nAnswer:"
    )

    async def generator() -> AsyncGenerator[bytes, None]:
        """
        Use llama_cpp streaming interface if available. This generator yields
        small text chunks as bytes so StreamingResponse can pipe them to client.
        """
        collected_text = ""

        try:
            # Prefer create_completion streaming API if available
            # llama_cpp returns a generator yielding dict chunks when stream=True
            stream = None
            if hasattr(llm, "create_completion"):
                stream = llm.create_completion(
                    prompt=prompt,
                    stream=True,
                    max_tokens=req.max_tokens,
                    temperature=0.7,
                    stop=["</s>", "User:"]
                )
            else:
                # Fallback: call llm() with stream=True
                stream = llm(prompt=prompt, stream=True, max_tokens=req.max_tokens, temperature=0.7)

            # iterate stream and yield tokens as they arrive
            for chunk in stream:
                # chunk structure varies by llama_cpp version; handle common shapes
                token = ""
                if isinstance(chunk, dict):
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        # typical: {"choices": [{"text": "..."}, ...]}
                        token = chunk["choices"][0].get("text", "")
                    else:
                        # sometimes chunk may be like {"delta": {"content": "..."}}
                        if "delta" in chunk:
                            token = chunk["delta"].get("content", "") or ""
                        else:
                            # unknown chunk shape; stringify fallback
                            token = str(chunk)
                else:
                    token = str(chunk)

                if token:
                    collected_text += token
                    try:
                        yield token.encode("utf-8")
                        # a tiny sleep allows other tasks to run & provides smoother streaming
                        await asyncio.sleep(0)
                    except Exception:
                        # client disconnected maybe; stop
                        break

        except Exception as e:
            logging.exception("LLM stream error")
            yield f"\n\n[Stream Error] {e}".encode("utf-8")

        # after streaming completes, clean and store final answer in memory
        cleaned_output = clean_ans(collected_text)
        final_output = format_output(cleaned_output, docs)
        memory.append((user_message, final_output))
        chat_memory[user_id] = memory[-memory_size:]

    # streaming plain text; frontend will read as stream chunks and append
    return StreamingResponse(generator(), media_type="text/plain")


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        dst = os.path.join(PDF_DIR, file.filename)
        if os.path.exists(dst):
            return JSONResponse({"status": "already", "chunks": 0})
        with open(dst, "wb") as f:
            f.write(await file.read())

        doc = fitz.open(dst)
        docs = []
        for i, page in enumerate(doc, start=1):
            text = ""
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span.get("text", "") + " "
                    text += "\n\n"
            for chunk in split_chunks(text):
                docs.append(Document(page_content=chunk, metadata={"book_name": file.filename, "page_number": i}))
        if docs:
            vectordb.add_documents(docs)
            vectordb.persist()
        return JSONResponse({"status": "embedded", "chunks": len(docs)})
    except Exception as e:
        logging.exception("upload failed")
        return JSONResponse({"status": "error", "detail": str(e)})


@app.get("/pdfs")
async def list_pdfs():
    try:
        return {"files": os.listdir(PDF_DIR)}
    except Exception as e:
        return {"files": [], "error": str(e)}


@app.post("/reset")
async def reset_chat(user_id: str = Form(...)):
    chat_memory.pop(user_id, None)
    return {"status": "reset"}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    If you serve the provided index.html from the same directory, this will return it.
    Otherwise, point your browser to the static frontend file location.
    """
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>PDF Chat Assistant</h1><p>Frontend not found. Put index.html into ./static/</p>", status_code=200)