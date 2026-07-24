from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import re

# Add scripts directory to import path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import memory_manager
from scope_processor import ScopeProcessor
from cognition_layer import CognitionLayer

app = FastAPI(title="Inquiry Engine API", version="1.0.0")

# Enable CORS for local-first UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since it is a local-first desktop app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas
class ScopeConfigSchema(BaseModel):
    namespaces: Optional[List[str]] = []
    domains: Optional[List[str]] = []
    include_tags: Optional[List[str]] = []
    exclude_tags: Optional[List[str]] = []
    include_terms: Optional[List[str]] = []
    exclude_terms: Optional[List[str]] = []

class ScopeValidateRequest(BaseModel):
    query: str
    scope: ScopeConfigSchema
    limit: Optional[int] = 10

class NoteSaveRequest(BaseModel):
    path: str
    content: str

# Endpoints
@app.get("/api/memory/notes")
def get_notes():
    """Retrieve recursive list of refined notes with metadata"""
    try:
        notes = memory_manager.get_refined_notes_list()
        return {"status": "ok", "notes": notes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/notes/content")
def get_note_content(path: str = Query(..., description="Relative path under refined_notes/")):
    """Read a specific note file"""
    try:
        content = memory_manager.read_note_content(path)
        return {"status": "ok", "path": path, "content": content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Note not found: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/notes/save")
def save_note(req: NoteSaveRequest):
    """Save a refined note and trigger auto-indexing"""
    try:
        memory_manager.save_note_content(req.path, req.content)
        return {"status": "ok", "message": f"Saved and indexed {req.path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/sessions")
def get_sessions():
    """Load inquiry sessions history"""
    try:
        sessions = memory_manager.load_session_memory()
        return {"status": "ok", "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scopes/config")
def get_scopes_config():
    """
    Get all available domains, namespaces, and tags currently indexed in the system.
    This helps the UI display filters dynamically.
    """
    try:
        # Default namespaces
        namespaces = ["chatgpt_export", "refined_notes"]
        
        # Collect domains and tags from refined notes
        notes = memory_manager.get_refined_notes_list()
        note_domains = set()
        tags = set()
        
        for n in notes:
            note_domains.add(n["domain"])
            for t in n["tags"]:
                tags.add(t)
                
        # Merge with typical chatgpt classification domains
        default_domains = [
            "career_positioning", "life_pacing", "identity", 
            "meaning", "social_moves", "money_moves", 
            "environment_design", "reputation_strategy", "general"
        ]
        all_domains = list(note_domains.union(set(default_domains)))
        
        # Sort values
        all_domains.sort()
        sorted_tags = sorted(list(tags))
        
        return {
            "status": "ok",
            "namespaces": namespaces,
            "domains": all_domains,
            "tags": sorted_tags
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scopes/validate")
def validate_scope(req: ScopeValidateRequest):
    """
    Executes a semantic search against ChromaDB, applies Scope Layer filter bounds,
    and returns matching sources with reasons for inclusion/exclusion if desired.
    """
    try:
        collection = memory_manager.get_collection()
        model = memory_manager.get_embedding_model()
        
        # 1. Encode query
        query_emb = model.encode([req.query]).tolist()[0]
        
        # 2. Extract Chroma-specific metadata filter (domain/namespace)
        where_filter = ScopeProcessor.build_chroma_where(req.scope.dict())
        
        # 3. Retrieve raw candidate list (fetch slightly more to allow for filtering headroom)
        raw_limit = req.limit * 4
        
        if where_filter:
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=raw_limit,
                where=where_filter
            )
        else:
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=raw_limit
            )
            
        # 4. Format into a structured list for the scope filter
        raw_list = []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results.get("distances", [[]])[0]
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            raw_list.append({
                "document": doc,
                "metadata": meta,
                "distance": dist
            })
            
        # 5. Apply the Scope Layer processor to filter out items
        filtered_results = ScopeProcessor.filter_results(raw_list, req.scope.dict())
        
        # 6. Apply ultimate limit
        final_results = filtered_results[:req.limit]
        
        # 7. Collect diagnostics on what was excluded for debugging
        excluded_details = []
        for item in raw_list:
            is_ok, reason = ScopeProcessor.matches_scope(item["document"], item["metadata"], req.scope.dict())
            if not is_ok:
                excluded_details.append({
                    "title": item["metadata"].get("title"),
                    "source": item["metadata"].get("source"),
                    "reason": reason
                })
        
        return {
            "status": "ok",
            "query": req.query,
            "scope": req.scope,
            "total_raw_candidates": len(raw_list),
            "total_passed": len(filtered_results),
            "results": final_results,
            "excluded_diagnostics": excluded_details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CognitionReasonRequest(BaseModel):
    query: str
    mode: str
    scope: ScopeConfigSchema
    model: Optional[str] = "gemma3:12b"

@app.post("/api/cognition/reason")
def reason(req: CognitionReasonRequest):
    """
    Executes the five-layer cognition pipeline: Recall, Synthesis, Framework, Raw, Grounding, Mapping
    """
    try:
        result = CognitionLayer.execute_cognition_pipeline(
            query=req.query,
            mode=req.mode,
            scope=req.scope.dict(),
            model_name=req.model
        )
        return {
            "status": "ok",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Start on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
