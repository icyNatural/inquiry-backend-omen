import chromadb

def main():
    client = chromadb.PersistentClient(path="chroma_db")
    print("Available collections:")
    collections = client.list_collections()
    for col in collections:
        print(f"- {col.name}")
        
    try:
        col = client.get_collection("chat_notes")
        count = col.count()
        print(f"\nCollection 'chat_notes' count: {count}")
        if count > 0:
            peek = col.peek(limit=1)
            print("\nPeek at first item:")
            print("ID:", peek['ids'][0])
            print("Metadata:", peek['metadatas'][0])
            print("Document snippet:", peek['documents'][0][:200])
    except Exception as e:
        print("Error getting collection 'chat_notes':", e)

if __name__ == "__main__":
    main()
