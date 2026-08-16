import os
import shutil
import stat
from git import Repo
from dotenv import load_dotenv
import time

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from chunker import extract_chunks_from_code
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

CLONE_DIR = "temp_repo"
PERSIST_DIR = "chroma_db"


def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def build_vectorstore(documents, embeddings, persist_dir, batch_size=5, delay_seconds=3, max_retries=5):
    """
    Embeds documents in small batches, with exponential backoff retry
    if Google API rate limits (ResourceExhausted) are encountered.
    """
    vectorstore = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch, embedding=embeddings, persist_directory=persist_dir
                    )
                else:
                    vectorstore.add_documents(batch)
                break  # Success
            except ResourceExhausted:
                wait_time = delay_seconds * (2 ** attempt)
                print(f"Rate limit hit, retrying batch in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
        else:
            raise RuntimeError("Failed to embed a batch after multiple retries. Try again later.")

        time.sleep(delay_seconds)

    return vectorstore


def ingest_repo(repo_url: str):
    # --- STEP 1: FAST SHALLOW CLONE (depth=1) ---
    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR, onerror=remove_readonly)
    
    # depth=1 downloads ONLY the latest commit snapshot (5x faster)
    Repo.clone_from(repo_url, CLONE_DIR, depth=1)

    # --- STEP 2: HYBRID AST CHUNKING ---
    all_documents = []
    indexed_files = set()

    for root, _, files in os.walk(CLONE_DIR):
        if ".git" in root:
            continue
        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, CLONE_DIR)

            # README ingestion
            if filename.lower() in ("readme.md", "readme.txt", "readme"):
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    readme_text = f.read()
                doc = Document(
                    page_content=f"File: {rel_path}\n\n{readme_text}",
                    metadata={"source": rel_path, "name": "README"}
                )
                all_documents.append(doc)
                indexed_files.add(rel_path)
                continue

            if not filename.endswith(".py"):
                continue

            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()

            chunks = extract_chunks_from_code(code, rel_path)

            for chunk in chunks:
                doc = Document(
                    page_content=chunk["text"],
                    metadata={"source": chunk["source"], "name": chunk["name"]}
                )
                all_documents.append(doc)
                indexed_files.add(rel_path)

    if not all_documents:
        raise ValueError("No valid code or documents found in this repository.")

    # --- STEP 3: EMBED & VECTORSTORE ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, onerror=remove_readonly)

    build_vectorstore(all_documents, embeddings, PERSIST_DIR)

    return len(all_documents), sorted(list(indexed_files))