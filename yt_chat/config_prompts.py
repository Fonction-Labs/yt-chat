# TODO:
#if images is not None:
#    prompt += """Some additional page images (which corresponds to the Context text content) will be provided to help you answer the Question. They may contain figures or tables.\n"""

# ------ GENERAL PROMPTS (CONTEXT AND HYPOTHETICAL) ------

def generate_openai_context_prompt(question: str, context: str) -> str:
    # Custom prompt (not sure this is optimal)
    return f"""Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.
               Question: {question}\n\n
               Context: {context}\n\n
               Answer:\n"""

def generate_mistral_context_prompt(question: str, context: str) -> str:
    # Prompt from https://docs.mistral.ai/guides/basic-RAG/
    return f"""Context information is below.
               ---------------------
               {context}
               ---------------------
               Given the context information and not prior knowledge, answer the query.
               Query: {question}
               Answer:
            """

def generate_openai_hypothetical_prompt(question: str) -> str:
    # Prompt from: https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde
    return f"""Given a question, generate a paragraph of text that answers the question.    Question: {question}    Paragraph:"""

def generate_mistral_hypothetical_prompt(question: str) -> str:
    # Same prompt as OpenAI with additional "the same way it was formulated"
    return f"""Given a question, generate a paragraph of text that answers the question the same way it was formulated.    Question: {question}    Paragraph:"""
