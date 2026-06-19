import sys, json
import chromadb
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
TOP_K = 3


# Load the saved FAISS index and the chunk metadata from disk.
def load_faiss_index():
    if not Path("index.faiss").exists():
        print("No index found. Run: python embed.py first.")
        sys.exit(1)

    index  = faiss.read_index("index.faiss")
    chunks = json.loads(Path("chunks.json").read_text(encoding="utf-8"))
    return index, chunks


# Open the saved ChromaDB notes collection from disk.
def load_chroma_collection():
    if not Path("chroma_db").exists():
        print("No ChromaDB found. Run: python embed.py first.")
        sys.exit(1)
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection("notes")


# Search the FAISS index using the query embedding and print the top results.
def search_faiss(query: str, index, chunks, qvec) -> None:
    faiss.normalize_L2(qvec)
    scores, indices = index.search(qvec, TOP_K)
    print(f"\nFAISS top {TOP_K} results for: '{query}'\n" + "─"*50)
    for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1):
        chunk = chunks[idx]
        print(f"#{rank} [{score:.3f}] {chunk['source']}")
        print(f"   {chunk['text']}")
        print()


# Search the ChromaDB collection using the query embedding and print the top results.
def search_chroma(query: str, collection, qvec) -> None:
    results = collection.query(
        query_embeddings=[qvec[0].tolist()],
        n_results=TOP_K,
    )
    print(f"\nChromaDB top {TOP_K} results for: '{query}'\n" + "─"*50)
    for rank, (doc, meta, dist) in enumerate(
            zip(results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]), start=1):
        print(f"#{rank} [{dist:.3f}] {meta['source']}")
        print(f"   {doc}")
        print()


# Convert the question into an embedding and search both FAISS and ChromaDB.
def search(query: str, index, chunks, collection, model) -> None:
    qvec = model.encode([query]).astype("float32")
    search_faiss(query, index, chunks, qvec.copy())
    search_chroma(query, collection, qvec.copy())


# Load both search backends and keep asking the user for search queries.
def main():
    index, chunks = load_faiss_index()
    collection = load_chroma_collection()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Loaded {index.ntotal} FAISS chunks.")
    print(f"Loaded {collection.count()} ChromaDB chunks. Ready to search.")
    while True:
        try:
            query = input("\nSearch: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break
            search(query, index, chunks, collection, model)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
