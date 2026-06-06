import os
import uuid
from pathlib import Path
from typing import List

from tqdm import tqdm
import fitz  # PyMuPDF to validate PDFs
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==== CONFIGURATION ====
PDF_DIR = r"Path to your pdfs files"  # ⬅️ Update this path
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ⬅️ Updated model
CHUNK_SIZE = 650
CHUNK_OVERLAP = 50

# ==== INIT EMBEDDINGS ====
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ==== INIT TEXT SPLITTER ====
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# ==== INIT VECTOR DB ====
vectordb = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embedding_model
)

# ==== PDF VALIDATION ====
def is_valid_pdf(filepath: str) -> bool:
    try:
        with fitz.open(filepath) as _:
            return True
    except Exception:
        return False

# ==== MAIN PDF PROCESSOR ====
def process_all_pdfs(pdf_folder: str):
    pdf_folder_path = Path(pdf_folder).resolve()
    if not pdf_folder_path.exists():
        raise FileNotFoundError(f"[!] Folder not found: {pdf_folder_path}")

    pdf_files = list(pdf_folder_path.glob("*.pdf"))
    if not pdf_files:
        print("[!] No PDF files found.")
        return

    print(f"[+] Found {len(pdf_files)} PDF(s) in {pdf_folder_path}")

    total_chunks = 0

    for pdf_path in pdf_files:
        book_name = pdf_path.stem

        # Validate file
        if not is_valid_pdf(str(pdf_path)):
            print(f"[!] Skipping invalid or corrupted PDF: {pdf_path.name}")
            continue

        try:
            # Check for duplicates
            existing_docs = vectordb.get(where={"book_name": book_name})
            if existing_docs and existing_docs["ids"]:
                print(f"⏩ Skipping '{book_name}' (already in vector store)")
                continue

            print(f"\n📖 Processing: {book_name}")
            loader = PyPDFLoader(str(pdf_path))

            try:
                pages = loader.load_and_split()
            except Exception as e:
                print(f"[!] Failed to load '{book_name}': {e}")
                continue

            chunks: List[Document] = []

            for i, page in enumerate(tqdm(pages, desc=f"🔍 Splitting pages from {book_name}")):
                page_chunks = splitter.split_text(page.page_content)
                for chunk in page_chunks:
                    if isinstance(chunk, str) and chunk.strip():  # Only non-empty valid strings
                        chunks.append(Document(
                            page_content=chunk.strip(),
                            metadata={
                                "id": str(uuid.uuid4()),
                                "book_name": book_name,
                                "page_number": i + 1
                            }
                        ))

            vectordb.add_documents(chunks)
            print(f"✅ Stored {len(chunks)} chunks from '{book_name}'")
            total_chunks += len(chunks)

        except Exception as e:
            print(f"[!] Unexpected error with '{book_name}': {e}")

    print(f"\n✅ All done! Total chunks stored: {total_chunks}")

# ==== QUERY FUNCTION ====
def search_query(query: str, top_k: int = 5):
    results = vectordb.similarity_search_with_score(query, k=top_k)

    for doc, score in results:
        meta = doc.metadata
        print(f"\n[📚 Match - Score: {score:.4f}]")
        print(f"Book: {meta.get('book_name')}, Page: {meta.get('page_number')}, ID: {meta.get('id')}")
        print(f"Content:\n{doc.page_content[:300]}...\n")

# ==== MAIN ENTRY ====
if __name__ == "__main__":
    process_all_pdfs(PDF_DIR)
    # Optional query test
    # search_query("What is deep learning?")
