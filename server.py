import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Initialize the pipeline
from src.agent import KnowledgeBaseAgent
from src.embeddings.base import get_embedder_by_name
from src.retrieval.store import EmbeddingStore

app = FastAPI(title="TTHC RAG API")

# Initialize agent globally
try:
    print("Initializing embedding model...")
    embedder = get_embedder_by_name()
    print("Initializing vector store...")
    store = EmbeddingStore(embedder=embedder)
    print("Initializing RAG agent...")
    AGENT = KnowledgeBaseAgent(store=store)
    print("Agent initialized successfully.")
except Exception as e:
    print(f"Warning: Could not initialize KnowledgeBaseAgent: {e}")
    AGENT = None

class QueryRequest(BaseModel):
    query: str

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "agent_loaded": AGENT is not None
    }

@app.post("/api/query")
def run_query(request: QueryRequest):
    if not AGENT:
        raise HTTPException(status_code=500, detail="Agent is not initialized. Please configure API keys and embedding server.")
    
    try:
        response = AGENT.answer_structured(request.query)
        return response.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/procedure/{ma_thu_tuc}")
def get_procedure(ma_thu_tuc: str):
    """Fallback: Just find the file and parse it. 
    In a real app, this would query the DB. We can scan local files for the matching ID."""
    from src.parsing.tthc_parser import TTHCParser
    ids_dir = Path("data/thutuchanhchinh/TTHC_IDs")
    
    parser = TTHCParser(ids_dir=ids_dir if ids_dir.exists() else None)
    
    # We might not know which folder it is in, so we just do rglob
    data_dir = Path("data/thutuchanhchinh/markdown_json")
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Data dir not found")
        
    for md_file in data_dir.rglob("*.md"):
        if md_file.stem == ma_thu_tuc or ma_thu_tuc in md_file.name:
            try:
                doc = parser.parse_file(md_file)
                # Convert dataclass to dict
                return {
                    "doc_id": doc.doc_id,
                    "ma_thu_tuc": doc.ma_thu_tuc,
                    "ten_thu_tuc": doc.ten_thu_tuc,
                    "source_url": doc.source_url,
                    "agency_folder": doc.agency_folder,
                    "flat_metadata": doc.flat_metadata,
                    "sections": [
                        {
                            "section_type": s.section_type,
                            "heading": s.heading,
                            "content": s.content
                        } for s in doc.sections
                    ]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse file: {e}")
                
    raise HTTPException(status_code=404, detail=f"Procedure {ma_thu_tuc} not found")

@app.get("/api/procedures")
def list_procedures(page: int = 1, limit: int = 20):
    from src.parsing.tthc_parser import TTHCParser
    ids_dir = Path("data/thutuchanhchinh/TTHC_IDs")
    parser = TTHCParser(ids_dir=ids_dir if ids_dir.exists() else None)
    
    data_dir = Path("data/thutuchanhchinh/markdown_json")
    if not data_dir.exists():
        return {"items": [], "total": 0}
        
    all_files = list(data_dir.rglob("*.md"))
    start = (page - 1) * limit
    end = start + limit
    
    paginated_files = all_files[start:end]
    results = []
    for f in paginated_files:
        try:
            doc = parser.parse_file(f)
            results.append({
                "ma_thu_tuc": doc.ma_thu_tuc,
                "ten_thu_tuc": doc.ten_thu_tuc,
                "agency_folder": doc.agency_folder
            })
        except:
            pass
            
    return {"items": results, "total": len(all_files), "page": page, "limit": limit}

@app.post("/api/inspect")
def inspect_retrieval(request: QueryRequest):
    """Admin endpoint to see retrieval steps"""
    if not AGENT:
        raise HTTPException(status_code=500, detail="Agent not loaded")
    
    # Run retrieval manually to expose steps
    store = AGENT.store
    query_text = request.query
    
    parsed = AGENT.query_parser.parse(query_text)
    meta_filter = parsed.metadata_filter
    
    search_results = store.search_with_filter(
        query_text, 
        top_k=AGENT.top_k, 
        metadata_filter=meta_filter
    )
    
    return {
        "parsed_query": parsed.query_variants,
        "filters_extracted": meta_filter,
        "parent_resolved": [
             {
                 "content": r["content"],
                 "score": r["score"],
                 "metadata": r.get("metadata", {})
             } for r in search_results
        ]
    }

@app.get("/api/benchmark")
def get_benchmark():
    """Admin endpoint to view benchmark dataset"""
    bench_path = Path("tests/benchmark_queries.json")
    if not bench_path.exists():
        return {"queries": [], "total": 0}
        
    import json
    queries = json.loads(bench_path.read_text(encoding="utf-8"))
    return {"queries": queries, "total": len(queries)}

# Mount static files
web_dir = Path("web")
if not web_dir.exists():
    web_dir.mkdir(parents=True)
    
app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
