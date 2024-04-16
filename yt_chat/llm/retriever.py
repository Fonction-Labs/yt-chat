import numpy as np
from functools import partial
from joblib import Parallel, delayed

from .openai import client

def cosine_similarity(emb_1, embd_2):
    return np.dot(emb_1, embd_2) / (np.linalg.norm(emb_1) * np.linalg.norm(embd_2))

def retriever(docs, question, n: int = 4):
    doc_embeddings = Parallel(n_jobs=8, prefer="threads")(delayed(get_embedding)(doc) for doc in docs)
    question_embedding = get_embedding(question)
    similarities = map(partial(cosine_similarity, question_embedding), doc_embeddings)
    return [x for x, _ in sorted(zip(docs, similarities))][::-1][:n]

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input=[text], model=model).data[0].embedding
