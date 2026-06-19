import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

NOTES_DIR   = Path("notes")
INDEX_FILE  = Path("index.faiss")
CHUNKS_FILE = Path("chunks.json")
CHUNK_SIZE  = 3
EMBED_BATCH_SIZE = 8
ABBREVIATIONS = ["e.g.", "i.e.", "Dr.", "Mr.", "Mrs.", "Ms.", "U.S."]


# Split text into sentences without breaking common abbreviations.
def split_sentences(text: str) -> list[str]:
    protected_text = text
    placeholders = {}
    for i, abbreviation in enumerate(ABBREVIATIONS):
        placeholder = f"__ABBR_{i}__"
        protected_text = protected_text.replace(abbreviation, placeholder)
        placeholders[placeholder] = abbreviation

    sentences = re.split(r"(?<=[.!?])\s+", protected_text)
    for placeholder, abbreviation in placeholders.items():
        sentences = [sentence.replace(placeholder, abbreviation)
                     for sentence in sentences]
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# Read every .txt file from notes/ and split the text into chunks.
def read_notes() -> list[dict]:
    chunks = []
    for file in sorted(NOTES_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        sentences = split_sentences(text)
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk_text = " ".join(sentences[i:i+CHUNK_SIZE])
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
    from chromadb.errors import NotFoundError

    client = chromadb.PersistentClient(path="chroma_db")
    try:
        client.delete_collection("notes")
    except (NotFoundError, ValueError):
        pass

    collection = client.create_collection("notes")
    collection.upsert(
        ids       = [str(c["chunk_id"]) for c in chunks],
        documents = [c["text"]      for c in chunks],
        embeddings= embeddings,
        metadatas = [{"source": c["source"]} for c in chunks],
    )
    print(f"Chroma: indexed {collection.count()} chunks")


# Build both FAISS and ChromaDB indexes from the notes folder.
def main() -> None:
    chunks, embeddings = build_index()
    build_chroma_index(chunks, embeddings)


if __name__ == "__main__":
    main()
