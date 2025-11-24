# utils/qa.py

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
import os
import pickle

# Load models once (module-level)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # embedding dimension for the chosen model

QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"  # extractive QA reader

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
qa_pipeline = pipeline("question-answering", model=QA_MODEL_NAME, tokenizer=QA_MODEL_NAME)

# We'll store indexes and related metadata in memory; optionally persist to disk.
# Structure: {doc_id: {"index": faiss_index, "embeddings": np.array, "chunks": [str, ...]}}
DOC_INDEX_STORE = {}

def chunk_text_to_sentences(text, max_chars=1000):
    """
    Break text into chunks of ~max_chars attempting to split on sentences.
    Returns list of chunk strings.
    """
    if not text:
        return []

    sentences = [s.strip() for s in text.replace("\n", " ").split('.') if s.strip()]
    chunks = []
    cur = ""
    for s in sentences:
        part = (s + ". ")
        if len(cur) + len(part) <= max_chars:
            cur += part
        else:
            if cur:
                chunks.append(cur.strip())
            cur = part
    if cur:
        chunks.append(cur.strip())
    return chunks

def build_faiss_index_for_doc(doc_id, chunks):
    """
    Build and store a FAISS index for the list of text chunks.
    doc_id is a unique identifier for this document (e.g., filename or generated id).
    """
    # compute embeddings
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    # normalize for cosine similarity using inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # create FAISS index
    index = faiss.IndexFlatIP(EMBED_DIM)  # inner product for cosine (vectors already normalized)
    index.add(embeddings.astype('float32'))

    # store in memory
    DOC_INDEX_STORE[doc_id] = {
        "index": index,
        "embeddings": embeddings.astype('float32'),
        "chunks": chunks
    }

    return True

def retrieve_top_k(doc_id, question, k=3):
    """
    Return top-k chunks (text) most similar to the question for the given doc_id.
    """
    if doc_id not in DOC_INDEX_STORE:
        return []

    # embed and normalize question
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    index = DOC_INDEX_STORE[doc_id]["index"]

    # search
    D, I = index.search(q_emb.astype('float32'), k)
    indices = I[0].tolist()
    # Build list of (chunk, score)
    scores = D[0].tolist()
    chunks = DOC_INDEX_STORE[doc_id]["chunks"]
    results = []
    for idx, score in zip(indices, scores):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append({"chunk": chunks[idx], "score": float(score), "idx": int(idx)})
    return results

def answer_question_by_retriever_reader(doc_id, question, top_k=3):
    """
    Retrieve top-k chunks and run QA reader over each, then return the best answer (by score).
    """
    candidates = retrieve_top_k(doc_id, question, k=top_k)
    if not candidates:
        return {"answer": "", "score": 0.0, "source_chunk_idx": None}

    best_answer = {"answer": "", "score": 0.0, "source_chunk_idx": None, "context": ""}

    for c in candidates:
        context = c["chunk"]
        try:
            out = qa_pipeline(question=question, context=context)
        except Exception as e:

            continue


        score = float(out.get("score", 0.0))
        answer = out.get("answer", "").strip()
        if score > best_answer["score"] and answer:
            best_answer = {
                "answer": answer,
                "score": score,
                "source_chunk_idx": c["idx"],
                "context": context
            }

    return best_answer


def save_index(doc_id, base_dir="models"):
    if doc_id not in DOC_INDEX_STORE:
        return False
    meta = DOC_INDEX_STORE[doc_id]
    os.makedirs(base_dir, exist_ok=True)

    with open(os.path.join(base_dir, f"{doc_id}_meta.pkl"), "wb") as f:
        pickle.dump({
            "chunks": meta["chunks"],
            "embeddings": meta["embeddings"]
        }, f)

    faiss.write_index(meta["index"], os.path.join(base_dir, f"{doc_id}.index"))
    return True

def load_index(doc_id, base_dir="models"):
    meta_path = os.path.join(base_dir, f"{doc_id}_meta.pkl")
    idx_path = os.path.join(base_dir, f"{doc_id}.index")
    if not os.path.exists(meta_path) or not os.path.exists(idx_path):
        return False
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    index = faiss.read_index(idx_path)
    DOC_INDEX_STORE[doc_id] = {
        "index": index,
        "embeddings": meta["embeddings"],
        "chunks": meta["chunks"]
    }
    return True
