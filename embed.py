import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

NOTES_DIR   = Path("notes")
INDEX_FILE  = Path("index.faiss")
CHUNKS_FILE = Path("chunks.json")
CHUNK_SIZE  = 3
EMBED_BATCH_SIZE = 8


# Read every .txt file from notes/ and split the text into chunks.
def read_notes() -> list[dict]:
    chunks = []
    for file in sorted(NOTES_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        sentences = [s.strip() for s in text.split(".")
                     if s.strip()]
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk_text = ". ".join(sentences[i:i+CHUNK_SIZE]) + "."
            chunks.append({
                "text": chunk_text,
                "source": file.name,
                "chunk_id": len(chunks),
            })
    return chunks


# Create embeddings, save the FAISS index, and save chunk metadata.
def build_index() -> tuple[list[dict], list[list[float]]]:
    print("Reading notes...")
    chunks = read_notes()
    print(f"Found {len(chunks)} chunks across your notes")
    if not chunks:
        raise ValueError("No text chunks found. Add .txt files to notes/ first.")

    print("Embedding chunks (this takes a few seconds)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts  = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype("float32")
    print("Building FAISS index...")
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"Done! Indexed {index.ntotal} chunks.")
    print(f"Saved: {INDEX_FILE}, {CHUNKS_FILE}")
    return chunks, embeddings.tolist()


# Save the same chunks and embeddings into a persistent ChromaDB collection.
def build_chroma_index(chunks: list[dict], embeddings: list[list[float]]) -> None:
    import chromadb

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("notes")
    collection.upsert(
        ids       = [str(c["chunk_id"]) for c in chunks],
        documents = [c["text"]      for c in chunks],
        embeddings= embeddings,
        metadatas = [{"source": c["source"]} for c in chunks],
    )
    print(f"Chroma: indexed {collection.count()} chunks")


# Search the ChromaDB notes collection and print the closest matches.
def search_chroma(query: str, n: int = 3) -> None:
    import chromadb

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("notes")
    results = collection.query(
        query_texts=[query],
        n_results=n,
    )
    print(f"\nChroma results for: '{query}'")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  [{dist:.3f}] {meta['source']}: {doc[:150]}")


# Build both FAISS and ChromaDB indexes from the notes folder.
def main() -> None:
    chunks, embeddings = build_index()
    build_chroma_index(chunks, embeddings)


if __name__ == "__main__":
    main()
