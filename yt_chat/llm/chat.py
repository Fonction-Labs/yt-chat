from functools import partial
import numpy as np
from tenacity import retry, wait_exponential

from .retriever import retriever
from .openai import client

def answer_question_video(docs, question, n=4):
    context = " ".join(retriever(docs, question, n))
    answer = answer_question(question, context)
    return answer

# Main function to answer question
def answer_question(question, context, model="gpt-3.5-turbo"):
    messages = get_prompt(question, context)
    response = api_call(messages, model)
    return response.choices[0].message.content

# Function to get prompt messages
def get_prompt(question, context):
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

@retry(wait=wait_exponential(multiplier=1, min=2, max=6))
def api_call(messages, model):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stop=["\n\n"],
        max_tokens=100,
        temperature=0.0,
    )
