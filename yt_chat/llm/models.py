import os
from warnings import warn

import ollama
from openai import OpenAI
from tenacity import retry, wait_exponential

class OpenAILLM:
    def __init__(self, model_name: str, embedding_model_name: str):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def embed(self, prompt: str):
        return self.client.embeddings.create(input=[prompt], model=embedding_model_name).data[0].embedding

    def predict(self, prompt: str, temperature: float):
        return predict_messages(messages=[{"role": "user", "content": prompt}], temperature=temperature)

    # TODO: fix
    #@retry(wait=wait_exponential(multiplier=1, min=2, max=6))
    def predict_messages(self, messages: list[dict[str, str]], temperature: float):
        return self.client.chat.completions.create(model=self.model_name,
                                                   messages=messages,
                                                   max_tokens=None,
                                                   temperature=temperature,
                                                   ).choices[0].message.content

class OllamaLLM:
    # TODO: VLLM for when deploying at-scale
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(self, prompt: str):
        ollama.pull(self.model_name)
        return ollama.embeddings(model=self.model_name, prompt=prompt)["embedding"]

    def predict(self, prompt: str, temperature: float):
        ollama.pull(self.model_name)
        return predict_messages(messages=[{"role": "user", "content": prompt}], temperature=temperature)

    def predict_messages(self, messages: list[dict[str, str]], temperature: float):
        warn("Unfortunately, ollama does not handle changing dynamically parameters like temperature (llamma.cpp does.")
        return ollama.chat(model=self.model_name, messages=messages)["response"]["message"]["content"]
