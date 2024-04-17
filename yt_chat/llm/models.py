import os
from warnings import warn

import ollama
from joblib import Parallel, delayed
from openai import OpenAI
from tenacity import retry, wait_exponential

from yt_chat.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from functools import partial

def parallel(method):
    def wrapper(self, *args, **kwargs):
        items = args[0] # Assuming args[0] is the list of items to parallelize
        n_jobs = kwargs.get('n_jobs', 8)  # Default value of n_jobs is 8

        # Calculate total number of items to process
        total_items = len(items)

        # Create tqdm_joblib context manager for progress bar
        with tqdm_joblib(tqdm(desc="Parallelizing batches...", total=total_items)):
            # Call the original method with provided arguments
            result = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(method)(self, item, *args[1:], **kwargs) for item in items
            )
            return result
    return wrapper


class OpenAILLM:
    def __init__(self, model_name: str, embedding_model_name: str):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def embed(self, prompt: str):
        return self.client.embeddings.create(input=[prompt], model=self.embedding_model_name).data[0].embedding

    @parallel
    def embed_batch_parallel(self, prompts: list[str], n_jobs: int = 8):
        return self.embed(prompts)

    def predict(self, prompt: str, temperature: float):
        return self.predict_messages(messages=[{"role": "user", "content": prompt}], temperature=temperature)

    @parallel
    def predict_batch_parallel(self, prompts: list[str], temperature: float, n_jobs: int = 8):
        return self.predict(prompts, temperature)

    #@retry(wait=wait_exponential(multiplier=1, min=2, max=6)) # TODO: fix?
    def predict_messages(self, messages: list[dict[str, str]], temperature: float):
        return self.client.chat.completions.create(model=self.model_name,
                                                   messages=messages,
                                                   max_tokens=None,
                                                   temperature=temperature,
                                                   ).choices[0].message.content

    @parallel
    def predict_messages_batch_parallel(self, list_messages: list[list[dict[str, str]]], temperature: float, n_jobs: int = 8):
        return self.predict_messages(list_messages, temperature)

class OllamaLLM:
    # TODO: VLLM for when deploying at-scale
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(self, prompt: str):
        ollama.pull(self.model_name)
        return ollama.embeddings(model=self.model_name, prompt=prompt)["embedding"]

    def embed_batch_parallel(self, prompts: list[str]):
        warn("Unfortunately, ollama does not handle parallelization of requests. No parallelization will occur.")
        return [self.embed(prompt) for prompt in tqdm(prompts)]

    def predict(self, prompt: str, temperature: float):
        ollama.pull(self.model_name)
        return self.predict_messages(messages=[{"role": "user", "content": prompt}], temperature=temperature)

    def predict_batch_parallel(self, prompts: list[str], temperature: float):
        warn("Unfortunately, ollama does not handle parallelization of requests. No parallelization will occur.")
        return [self.predict(prompt, temperature) for prompt in tqdm(prompts)]

    def predict_messages(self, messages: list[dict[str, str]], temperature: float):
        warn("Unfortunately, ollama does not handle changing dynamically parameters like temperature (llamma.cpp does.")
        return ollama.chat(model=self.model_name, messages=messages)["message"]["content"]

    def predict_messages_batch_parallel(self, list_messages: list[list[dict[str, str]]], temperature: float):
        warn("Unfortunately, ollama does not handle parallelization of requests. No parallelization will occur.")
        return [self.predict_messages(messages, temperature) for messages in tqdm(list_messages)]
