import re
import json
from pathlib import Path
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# Paths and collection names
ROOT_DIR = Path(__file__).resolve().parents[1]
REFINED_DIR = ROOT_DIR / "refined_notes"
MEMORY_DIR = ROOT_DIR / "inquiry_memory"
MEMORY_FILE = MEMORY_DIR / "life_inquiry_memory.json"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "chat_notes"

# Initialize Chroma and Embedding model
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_collection():
    return _collection

def get_embedding_model():
    return _embed_model

def load_session_memory():
    """Load session history from life_inquiry_memory.json"""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("[]", encoding="utf-8")
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_session_memory(memory_data):
    """Save session history back to file"""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(memory_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def append_session_entry(entry):
    """Append a single inquiry entry to history"""
    history = load_session_memory()
    history.append(entry)
    save_session_memory(history)

def parse_markdown_metadata(content):
    """
    Parse title and key metadata items from a refined note.
    Expects standard format:
    # Title
    - Saved: YYYY-MM-DD HH:MM:SS
    - Mode: mode_name
    - Query: query_text
    - Tags: #tag1 #tag2
    """
    lines = content.split("\n")
    title = "Untitled"
    metadata = {
        "saved": "",
        "mode": "",
        "query": "",
        "tags": []
    }
    
    # Try to extract title from first line
    if lines and lines[0].startswith("#"):
        title = lines[0].lstrip("#").strip()
        
    for line in lines:
        line = line.strip()
        if line.startswith("- Saved:"):
            metadata["saved"] = line.replace("- Saved:", "").strip()
        elif line.startswith("- Mode:"):
            metadata["mode"] = line.replace("- Mode:", "").strip()
        elif line.startswith("- Query:"):
            metadata["query"] = line.replace("- Query:", "").strip()
        elif line.startswith("- Tags:"):
            tags_str = line.replace("- Tags:", "").strip()
            # extract all #tags
            metadata["tags"] = [t.strip() for t in re.findall(r"#\w+", tags_str)]
            
    return title, metadata

def chunk_text(text, max_chars=1200):
    """Chunk note text into pieces up to max_chars, keeping paragraphs intact if possible"""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)
        if current_chunk and current_len + para_len > max_chars:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len + 2
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

def index_refined_note(file_path: Path):
    """
    Index a single refined note file into ChromaDB.
    Deletes any existing chunks for this note's relative path first to avoid duplicates.
    """
    if not file_path.exists() or file_path.suffix != ".md" or file_path.name == "index.md":
        return
        
    rel_path = file_path.relative_to(REFINED_DIR).as_posix()
    domain = file_path.parent.relative_to(REFINED_DIR).as_posix()
    if domain == ".":
        domain = "general"
        
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return
        
    title, parsed_meta = parse_markdown_metadata(content)
    
    # 1. Clean old entries
    try:
        _collection.delete(where={"path": rel_path})
    except Exception as e:
        print(f"Chroma delete error for {rel_path} (can ignore if not exists): {e}")
        
    # 2. Chunk content
    chunks = chunk_text(content, max_chars=1000)
    
    # 3. Embed and Add
    documents = []
    metadatas = []
    ids = []
    
    tags_str = ",".join(parsed_meta["tags"])
    last_mod = parsed_meta["saved"] or datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"note_{rel_path.replace('/', '_').replace('.', '_')}_{i}"
        documents.append(chunk)
        metadatas.append({
            "title": title,
            "path": rel_path,
            "source": "refined_notes",
            "chunk_index": i,
            "domain": domain,
            "tags": tags_str,
            "last_modified": last_mod
        })
        ids.append(chunk_id)
        
    if documents:
        embeddings = _embed_model.encode(documents).tolist()
        _collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Indexed refined note: {rel_path} ({len(documents)} chunks)")

def index_all_refined_notes():
    """Recursively scan refined_notes/ and index all valid markdown notes"""
    if not REFINED_DIR.exists():
        REFINED_DIR.mkdir(parents=True, exist_ok=True)
        return
        
    print("Indexing all refined notes...")
    for file_path in REFINED_DIR.rglob("*.md"):
        if file_path.name == "index.md":
            continue
        index_refined_note(file_path)
    print("Refined notes indexing completed.")

def get_refined_notes_list():
    """Retrieve list of all refined notes with metadata"""
    if not REFINED_DIR.exists():
        REFINED_DIR.mkdir(parents=True, exist_ok=True)
        return []
        
    notes = []
    for file_path in REFINED_DIR.rglob("*.md"):
        if file_path.name == "index.md":
            continue
        try:
            rel_path = file_path.relative_to(REFINED_DIR).as_posix()
            domain = file_path.parent.relative_to(REFINED_DIR).as_posix()
            if domain == ".":
                domain = "general"
                
            content = file_path.read_text(encoding="utf-8")
            title, parsed_meta = parse_markdown_metadata(content)
            
            notes.append({
                "title": title,
                "path": rel_path,
                "domain": domain,
                "tags": parsed_meta["tags"],
                "saved": parsed_meta["saved"],
                "mode": parsed_meta["mode"],
                "query": parsed_meta["query"],
                "size_bytes": file_path.stat().st_size
            })
        except Exception as e:
            print(f"Error parsing note info for {file_path}: {e}")
            
    return notes

def read_note_content(rel_path):
    """Read full note content given its relative path"""
    file_path = REFINED_DIR / rel_path
    if not file_path.exists():
        raise FileNotFoundError(f"Note not found: {rel_path}")
    return file_path.read_text(encoding="utf-8")

def save_note_content(rel_path, content):
    """Save note content and trigger auto-indexing"""
    file_path = REFINED_DIR / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    index_refined_note(file_path)
    return True

if __name__ == "__main__":
    # If run directly, index all refined notes as a setup step
    index_all_refined_notes()
