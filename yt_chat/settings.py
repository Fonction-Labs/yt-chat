from pydantic import BaseModel

from yt_chat.llm.models import OpenAILLM, OllamaLLM

MODELS = {
    # First in this dictionary will be the default for the cli
    "gpt-3.5-turbo": OpenAILLM("gpt-3.5-turbo", "text-embedding-3-small"),
    "mistral": OllamaLLM("mistral"),
}

MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE = {
        "gpt-3.5-turbo": 4096,
        "mistral": 4096,
        }

# ------ GENERAL CONTEXT MESSAGES ------
def generate_openai_context_message(question, context):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"""Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.
                           Question: {question}\n\n
                           Context: {context}\n\n
                           Answer:\n""",
        },
    ]


def generate_mistral_context_message(question, context):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"""Context information is below.
                           ---------------------
                           {context}
                           ---------------------
                           Given the context information and not prior knowledge, answer the query.
                           Query: {question}
                           Answer:
                           """,
        },
    ]

# ------ YOUTUBE SUMMARIES MESSAGES ------
def generate_openai_summarize_transcript_message(transcript: str):
    return [{
        "role": "user",
        "content": f"""
            Write a detailed summary of the following youtube video transcript.
                "{transcript}".
            CONCISE SUMMARY:
            """,
    }]


def generate_mistral_summarize_transcript_message(transcript: str):
    return [{
        "role": "user",
        "content": f"""
            I want you to write a concise summary of an extract of a youtube video transcript. Do NOT mention things you are not sure or do not know. If you don't know or lack context, do not make up things. Only use information provided by the transcript to write your concise summary.
            TRANSCRIPT: "{transcript}".
            ### CONCISE SUMMARY:
            """,
    }]


def generate_openai_summarize_summaries_message(summaries: str):
    return [{
        "role": "user",
        "content": f"""
            The following is set of summaries:
            "{summaries}"
            Take these and distill it into a final, consolidated detailed summary of 3 to 5 paragraphs with some subtitles for each paragraph (with markdown format).
            Helpful Answer:
            """,
    }]


def generate_mistral_summarize_summaries_message(summaries: str):
    return [{
        "role": "user",
        "content": f"""
            The following will be a set of summaries from a transcript. Take these and distill it into a final, consolidated summary of the main points. Construct it as a well organized summary of the main points and should be between 3 and 5 paragraphs.
            SUMMARIES: "{summaries}".
            ### DISTILLED ANSWER:
            """,
    }]

MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC = {
    "gpt-3.5-turbo": generate_openai_context_message,
    "mistral": generate_mistral_context_message,
}

MODEL_TO_GENERATE_SUMMARIZE_TRANSCRIPT_MESSAGES_FUNC = {
    "gpt-3.5-turbo": generate_openai_summarize_transcript_message,
    "mistral": generate_mistral_summarize_transcript_message,
}

MODEL_TO_GENERATE_SUMMARIZE_SUMMARIES_MESSAGES_FUNC = {
    "gpt-3.5-turbo": generate_openai_summarize_summaries_message,
    "mistral": generate_mistral_summarize_summaries_message,
}


