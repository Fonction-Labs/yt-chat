# TODO: handle original language for Mistral prompts

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
    return f"""Given a question, generate a paragraph of text that answers the question. Use the original language of the question to answer.    Question: {question}    Paragraph:"""

def generate_mistral_hypothetical_prompt(question: str) -> str:
    # Same prompt as OpenAI with additional "the same way it was formulated"
    return f"""Given a question, generate a paragraph of text that answers the question the same way it was formulated.    Question: {question}    Paragraph:"""

# ------ YOUTUBE SUMMARIES PROMPTS ------
# (these are all custom prompts)

def generate_openai_summarize_transcript_prompt(transcript: str) -> str:
    return f"""Write a detailed summary of the following youtube video transcript. Use the original language of the transcript to write your summary.
                "{transcript}".
                CONCISE SUMMARY:
            """

def generate_mistral_summarize_transcript_prompt(transcript: str) -> str:
    return f"""I want you to write a concise summary of an extract of a youtube video transcript. Do NOT mention things you are not sure or do not know. If you don't know or lack context, do not make up things. Only use information provided by the transcript to write your concise summary.
            TRANSCRIPT: "{transcript}".
            ### CONCISE SUMMARY:
            """

def generate_openai_summarize_summaries_prompt(summaries: str) -> str:
    return f"""The following is set of summaries:
              "{summaries}"
              Take these and distill it into a final, consolidated summary of 3 to 5 paragraphs with some subtitles for each paragraph. Use the original language of the summaries to write your final summary.
              Helpful Answer:
              """

def generate_mistral_summarize_summaries_prompt(summaries: str) -> str:
    return f"""
            The following will be a set of summaries from a transcript. Take these and distill it into a final, consolidated summary of the main points. Construct it as a well organized summary of the main points and should be between 3 and 5 paragraphs.
            SUMMARIES: "{summaries}".
            ### DISTILLED ANSWER:
            """
