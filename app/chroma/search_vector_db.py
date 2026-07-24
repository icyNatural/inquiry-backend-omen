import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "chat_notes"

TOP_K = 10

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")

print("\n=== VECTOR SEARCH DEBUGGER ===")
print("Type a query.")
print("Commands: exit, quit\n")

while True:
    query = input("Search: ").strip()

    if not query:
        continue

    if query.lower() in ["exit", "quit"]:
        print("Goodbye.")
        break

    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n" + "=" * 100)
    print(f"RESULTS FOR: {query}")
    print("=" * 100 + "\n")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        title = meta.get("title", "Untitled")
        convo = meta.get("conversation_index", "?")
        chunk = meta.get("chunk_index", "?")

        snippet = " ".join(doc.split())

        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."

        print(f"{i}. {title}")
        print(f"   distance: {dist:.4f}")
        print(f"   location: convo {convo} | chunk {chunk}")
        print(f"   snippet: {snippet}")
        print()