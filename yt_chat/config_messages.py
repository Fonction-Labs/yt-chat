from PIL import Image
from yt_chat.utils.images import encode_image_base64

def message(prompt: str, images: list[Image] = None):

    if images is not None:
        images = [encode_image_base64(image) for image in images]

    if images is None:
        images = []

    return [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}} for image in images]},
    ]

# ------ GENERAL MESSAGES (CONTEXT AND HYPOTHETICAL) ------
def generate_openai_context_message(question: str, context: str, images: list[Image] = None):
    # Custom prompt (not sure this is optimal)
    # GPT-4 can handle images
    prompt = """Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.\n"""
    if images is not None:
        prompt += """Some additional page images (which corresponds to the Context text content) will be provided to help you answer the Question. They may contain figures or tables.\n"""
    prompt += """Question: {question}\n\n
                 Context: {context}\n\n
                 Answer:\n"""

    return message(prompt, images)



def generate_mistral_context_message(question, context):
    # Prompt from https://docs.mistral.ai/guides/basic-RAG/
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


def generate_openai_hypothetical_message(question):
    # Prompt from: https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"""Given a question, generate a paragraph of text that answers the question.    Question: {question}    Paragraph:""",
        },
    ]


def generate_mistral_hypothetical_message(question):
    # Same prompt as OpenAI with additional "the same way it was formulated"
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"""Given a question, generate a paragraph of text that answers the question the same way it was formulated.    Question: {question}    Paragraph:""",
        },
    ]

# ------ YOUTUBE SUMMARIES MESSAGES ------
# (these are all custom prompts)

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
