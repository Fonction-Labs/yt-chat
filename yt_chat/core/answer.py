import uuid

from qdrant_client.http.models import PointStruct

from flib.utils.chunk_text import ChunkSettings, get_text_chunks
from flib.utils.qdrant import SingleCollectionQdrantClient
from flib.utils.images import load_image

def embed_and_store_chunks(embedding_model, chunks: list[str], metas: list[dict], qdrant_client: SingleCollectionQdrantClient) -> None:
    # Embed chunks
    embeddings = embedding_model.run_batch(chunks, parallel=True)

    # Store embedded chunks
    qdrant_client.upsert(
        qdrant_client.collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"meta": meta, "text": chunk},
            )
            for chunk, meta, embedding in zip(chunks, metas, embeddings)
        ],
    )

def embed_and_store_text(text: str, meta: dict, embedding_model, chunk_settings: ChunkSettings, qdrant_client: SingleCollectionQdrantClient):
    # Cut transcript text into chunks
    chunks = get_text_chunks(
        text, chunk_settings.chunk_size, chunk_settings.chunk_overlap
    )
    metas = [meta for i in range(len(chunks))] # TODO: fix? We copy the same meta for all chunks of the same text

    # Embed and store chunks
    embed_and_store_chunks(embedding_model, chunks, metas, qdrant_client)

def retrieve_top_k_chunks_for_query(
        embedding_model, query: str, qdrant_client: SingleCollectionQdrantClient, top_k: int, cosine_threshold: float = -1.,
) -> list[str, dict]:
    """
    Embeds a query with a given model, and return top_k chunks ((text, meta) tuples) in the collection
    with highest embedded vector similarity.
    """
    # Embed query
    query_embedding = embedding_model.run(query)

    # Retrieve top_k chunks with highest similarity in the collection
    query_results = qdrant_client.search(qdrant_client.collection_name, query_vector=query_embedding, limit=top_k)

    # Filter for cosine
    query_results = [result for result in query_results if result.score >= cosine_threshold]

    return [(result.payload["text"], result.payload["meta"]) for result in query_results]

def answer_query(query: str, model, embedding_model,
                 qdrant_client: SingleCollectionQdrantClient,
                 generate_hypothetical_prompt,
                 generate_context_prompt,
                 top_k: int,
                 cosine_threshold: float = -1.,
                 use_hypothetical: bool = True):
    """
    Answers a question (query), using a model and embeddings stored in a Qdrant database.

    If use_hypothetical is set to True, this will formulate and use a fake,
    hypothetical answer to the question in order to perform the embeddings retrieval
    (this can improve the quality of retrievals, at the cost of one additional LLM request per query).
    """
    # Saves the original query into a new variable
    original_query = query

    # If use_hypothetical, formulate a fake answer to the question to improve embeddings retrieval
    if use_hypothetical:
        prompt = generate_hypothetical_prompt(question=query)
        query = model.run(prompt, temperature=0.7)

    # TODO: if nothing is retrieved (due to cosine threshold), answer "could not find context" in chat

    # Retrieve top_k chunks for query
    top_k_chunks_for_query = retrieve_top_k_chunks_for_query(
        embedding_model, query, qdrant_client, top_k=top_k, cosine_threshold=cosine_threshold,
    )

    if len(top_k_chunks_for_query) == 0:
        return ("Sorry, I could not retrieve context to answer your question.", [])

    top_k_texts_for_query, top_k_metas_for_query = zip(*top_k_chunks_for_query)

    # Merge top_k chunks into a single string for context
    context = " ".join(top_k_texts_for_query)

    images = None
    if top_k_metas_for_query:
        images = [load_image(meta["jpg_path"]) for meta in top_k_metas_for_query]

    # Generate LLM "context"-type prompt
    prompt = generate_context_prompt(question=original_query, context=context)

    # Get LLM response
    bot_response = model.run(prompt=prompt, images=images, temperature=0.)

    return bot_response, top_k_metas_for_query
