import numpy as np
from functools import partial
from joblib import Parallel, delayed

def cosine_similarity(v1, v2):
    """
    Computes cosine similarity between 2 vector embeddings.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def retrieve_top_k_chunks_for_query(model, query: str, chunks: list[str], top_k: int = 5):
    """
    Embeds a question (query), and some chunked documents (chunks),
    and returns the top_k chunks with highest cosine similarity with the query.
    """
    # Embed query
    query_embedding = model.embed(query)
    # Embed chunks
    chunks_embeddings = Parallel(n_jobs=8, prefer="threads")(delayed(model.embed)(chunk) for chunk in chunks)
    # Compute cosine similarities between query and every chunk
    similarities = map(partial(cosine_similarity, query_embedding), chunks_embeddings)
    # Return chunks with highest cosine similarity with query
    return [x for x, _ in sorted(zip(chunks, similarities))][::-1][:top_k]
