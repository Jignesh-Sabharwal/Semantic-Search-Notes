# Semantic Search Notes

A small Python project that turns local `.txt` notes into searchable embeddings.

It builds two vector indexes from the same notes:

- **FAISS** for fast local vector similarity search
- **ChromaDB** for a persistent vector database

When you search, `search.py` runs the same question against both FAISS and ChromaDB so you can compare the results.

## How It Works

1. Add text files inside the `notes/` folder.
2. Run `embed.py`.
3. The notes are split into small chunks.
4. Each chunk is embedded with `all-MiniLM-L6-v2`.
5. FAISS stores the vectors in `index.faiss`.
6. ChromaDB stores the vectors in `chroma_db/`.
7. `chunks.json` stores the original chunk text and source file names.
8. Run `search.py` and type a question.

## Project Structure

```text
semantic_search/
├── embed.py              # Reads notes and builds FAISS + ChromaDB indexes
├── search.py             # Searches both FAISS and ChromaDB
├── notes/                # Your source .txt notes
├── chunks.json           # Generated chunk metadata
├── index.faiss           # Generated FAISS index
└── chroma_db/            # Generated ChromaDB database
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Add Notes

Put your `.txt` files in the `notes/` folder:

```text
notes/
├── ml_basics.txt
├── backprop.txt
└── optimization.txt
```

## Build The Indexes

Run:

```bash
python embed.py
```

This creates or updates:

- `index.faiss`
- `chunks.json`
- `chroma_db/`

Example output:

```text
Reading notes...
Found 69 chunks across your notes
Embedding chunks (this takes a few seconds)...
Building FAISS index...
Done! Indexed 69 chunks.
Saved: index.faiss, chunks.json
Chroma: indexed 69 chunks
```

## Search Your Notes

Run:

```bash
python search.py
```

Then type a question:

```text
Search: gradient descent
```

The script prints results from both search backends:

```text
FAISS top 3 results for: 'gradient descent'
--------------------------------------------------

ChromaDB top 3 results for: 'gradient descent'
--------------------------------------------------
```

To exit:

```text
q
```

or:

```text
quit
```

## FAISS vs ChromaDB Scores

The two systems report scores differently:

- **FAISS score**: higher is better
- **ChromaDB distance**: lower is better

So the numbers may look different even when the returned notes are similar.

## Generated Files

These files are created by running `embed.py`:

- `index.faiss`
- `chunks.json`
- `chroma_db/`

For GitHub, you can either commit them as a small demo index or ignore them and let users generate their own indexes locally.

## Model

This project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It creates 384-dimensional embeddings and works well for small local semantic search projects.
