from app.legacy.memory_manager_original import get_collection, get_embedding_model, get_refined_notes_list
from app.legacy.scope_processor_original import ScopeProcessor

def test_scope_filtering():
    print("=== STARTING MEMORY & SCOPE LAYER VALIDATION ===")
    
    # Check notes index list
    notes = get_refined_notes_list()
    print(f"Total refined notes discovered: {len(notes)}")
    for note in notes[:3]:
        print(f"- Title: {note['title']} | Path: {note['path']} | Domain: {note['domain']} | Tags: {note['tags']}")
        
    if not notes:
        print("WARNING: No notes found in refined_notes. Please ensure notes exist.")
        return
        
    collection = get_collection()
    model = get_embedding_model()
    
    # Run a test query: let's query for "thinking"
    query_text = "thinking"
    query_emb = model.encode([query_text]).tolist()[0]
    
    # Retrieve top 20 items from ChromaDB raw memory
    print(f"\nQuerying Chroma for '{query_text}'...")
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=20
    )
    
    raw_list = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]
    
    for doc, meta, dist in zip(documents, metadatas, distances):
        raw_list.append({
            "document": doc,
            "metadata": meta,
            "distance": dist
        })
        
    print(f"Retrieved {len(raw_list)} raw candidates.")
    
    # -------------------------------------------------------------
    # Scenario A: Scope limited to 'refined_notes' namespace only
    # -------------------------------------------------------------
    scope_a = {
        "namespaces": ["refined_notes"]
    }
    filtered_a = ScopeProcessor.filter_results(raw_list, scope_a)
    print(f"\n[Scope A] Namespace: 'refined_notes' only")
    print(f"Filtered count: {len(filtered_a)}")
    for item in filtered_a[:3]:
        print(f"  - Title: {item['metadata'].get('title')} | Source: {item['metadata'].get('source')} | Path: {item['metadata'].get('path')}")
        
    # Verify indeed all match the source
    for item in filtered_a:
        assert item['metadata'].get('source') == 'refined_notes', f"Source mismatch! Got {item['metadata'].get('source')}"

    # -------------------------------------------------------------
    # Scenario B: Exclude terms test (e.g. exclude word "music" or "song" or "flute")
    # -------------------------------------------------------------
    scope_b = {
        "exclude_terms": ["music", "flute", "playlist"]
    }
    filtered_b = ScopeProcessor.filter_results(raw_list, scope_b)
    print(f"\n[Scope B] Exclude terms: ['music', 'flute', 'playlist']")
    print(f"Filtered count: {len(filtered_b)} (reduced from {len(raw_list)})")
    
    # Verify no documents contain excluded terms
    for item in filtered_b:
        doc_lower = item['document'].lower()
        for term in scope_b["exclude_terms"]:
            assert term not in doc_lower, f"Document leaked excluded term '{term}'!"
            
    # -------------------------------------------------------------
    # Scenario C: Include tags test
    # -------------------------------------------------------------
    scope_c = {
        "include_tags": ["#thinkingrefinement", "#distortionreduction", "#cognition"]
    }
    filtered_c = ScopeProcessor.filter_results(raw_list, scope_c)
    print(f"\n[Scope C] Include tags: {scope_c['include_tags']}")
    print(f"Filtered count: {len(filtered_c)}")
    for item in filtered_c:
        print(f"  - Title: {item['metadata'].get('title')} | Tags: {item['metadata'].get('tags')}")

    # -------------------------------------------------------------
    # Scenario D: Domain restricted (e.g. 'framework' domain only)
    # -------------------------------------------------------------
    scope_d = {
        "namespaces": ["refined_notes"],
        "domains": ["framework"]
    }
    filtered_d = ScopeProcessor.filter_results(raw_list, scope_d)
    print(f"\n[Scope D] Namespace: refined_notes AND Domain: 'framework'")
    print(f"Filtered count: {len(filtered_d)}")
    for item in filtered_d:
        print(f"  - Title: {item['metadata'].get('title')} | Domain: {item['metadata'].get('domain')}")
        assert item['metadata'].get('domain') == 'framework'

    print("\n=== ALL MEMORY & SCOPE TESTS COMPLETED SUCCESSFULY ===")

if __name__ == "__main__":
    test_scope_filtering()
