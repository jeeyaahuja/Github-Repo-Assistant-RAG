import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional

from ingest import ingest_repo
from rag_chain import get_rag_chain

app = FastAPI(title="GitHub Assistant API")

# Enable CORS for Chrome Extension requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session state
rag_chain_instance = None
current_repo_name = None
current_file_list = []

LIST_FILES_TRIGGERS = ["all the files", "list all files", "list the files", "what files", "which files"]

class IngestRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[Tuple[str, str]]] = []

@app.get("/")
def health_check():
    return {
        "status": "online",
        "loaded_repo": current_repo_name,
        "indexed_files_count": len(current_file_list),
        "chain_ready": rag_chain_instance is not None
    }

@app.post("/api/ingest")
def ingest_repository(payload: IngestRequest):
    global rag_chain_instance, current_repo_name, current_file_list
    repo_url = payload.repo_url.strip()
    if not repo_url:
        raise HTTPException(status_code=400, detail="Repository URL is required.")
    
    try:
        rag_chain_instance = None
        current_file_list = []

        count, file_list = ingest_repo(repo_url)
        rag_chain_instance = get_rag_chain()
        current_repo_name = repo_url.rstrip("/").split("/")[-1]
        current_file_list = file_list
        
        return {
            "status": "success",
            "repo_name": current_repo_name,
            "chunk_count": count,
            "file_count": len(file_list),
            "files": file_list,
            "message": f"Successfully indexed {count} chunks across {len(file_list)} files from {current_repo_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def query_repository(payload: QueryRequest):
    global rag_chain_instance, current_file_list
    question = payload.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Check for direct file list queries
    if any(trigger in question.lower() for trigger in LIST_FILES_TRIGGERS):
        if not current_file_list:
            answer = "No repository files are currently indexed."
        else:
            file_list_md = "\n".join(f"- `{f}`" for f in current_file_list)
            answer = f"This repository has {len(current_file_list)} indexed files:\n\n{file_list_md}"
        
        return {
            "answer": answer,
            "standalone_question": question,
            "sources": []
        }

    if rag_chain_instance is None:
        if os.path.exists("chroma_db"):
            try:
                rag_chain_instance = get_rag_chain()
            except Exception:
                pass
        
        if rag_chain_instance is None:
            raise HTTPException(status_code=400, detail="No repository has been ingested yet.")

    try:
        result = rag_chain_instance({
            "question": question,
            "history": payload.history or []
        })

        sources = []
        seen = set()
        for doc in result.get("sources", []):
            source_file = doc.metadata.get("source", "Unknown")
            source_name = doc.metadata.get("name", "Unknown")
            key = f"{source_file}::{source_name}"
            if key not in seen:
                sources.append({
                    "file": source_file,
                    "name": source_name
                })
                seen.add(key)

        return {
            "answer": result["answer"],
            "standalone_question": result.get("standalone_question", question),
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
