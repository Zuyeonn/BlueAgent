import json
import faiss
from sentence_transformers import SentenceTransformer

def load_embedding_model():
    return SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def load_rag_index(corpus_path="rag_corpus.json"):
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    embedder = load_embedding_model()
    embeddings = embedder.encode(corpus, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return embedder, index, corpus
