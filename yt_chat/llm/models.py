from warnings import warn

import ollama
from openai import OpenAI
from tenacity import retry, wait_exponential

from yt_chat.utils.parallel import parallel
from tqdm import tqdm

from yt_chat.config import Config

class BaseLLM:
    def __init__(self, model_name: str):
        self.model_ref_name = Config.MODEL_TO_MODEL_REF_NAME[model_name]
        self.embedding_model_name = Config.MODEL_TO_EMBEDDING_MODEL_NAME.get(model_name)
        self.embedding_vector_size = Config.MODEL_TO_EMBEDDING_VECTOR_SIZE.get(model_name)
        self.context_window_token_size = Config.MODEL_TO_CONTEXT_WINDOW_TOKEN_SIZE.get(model_name)
        self.generate_context_messages_func = Config.MODEL_TO_GENERATE_CONTEXT_MESSAGES_FUNC.get(model_name)
        self.generate_hypothetical_messages_func = Config.MODEL_TO_GENERATE_HYPOTHETICAL_MESSAGES_FUNC.get(model_name)

        # Specific to yt-chat below
        self.generate_summarize_transcript_messages_func = (
            Config.MODEL_TO_GENERATE_SUMMARIZE_TRANSCRIPT_MESSAGES_FUNC.get(model_name)
        )
        self.generate_summarize_summaries_messages_func = (
            Config.MODEL_TO_GENERATE_SUMMARIZE_SUMMARIES_MESSAGES_FUNC.get(model_name)
        )

class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)

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

    # @retry(wait=wait_exponential(multiplier=1, min=2, max=6)) # TODO: fix?
    def predict_messages(self, messages: list[dict[str, str]], temperature: float):
        return (
            self.client.chat.completions.create(
                model=self.model_ref_name,
                messages=messages,
                #max_tokens=None,
                temperature=temperature,
            )
            .choices[0]
            .message.content
        )

    @parallel
    def predict_messages_batch_parallel(
        self,
        list_messages: list[list[dict[str, str]]],
        temperature: float,
        n_jobs: int = 8,
    ):
        return self.predict_messages(list_messages, temperature)


class OllamaLLM(BaseLLM):
    # TODO: VLLM for when deploying at-scale
    def __init__(self, model_name: str, api_key: str):
        # Note: API key is unused for local models
        super().__init__(model_name)

    def embed(self, prompt: str):
        ollama.pull(self.model_ref_name)
        return ollama.embeddings(model=self.model_ref_name, prompt=prompt)["embedding"]

    def embed_batch_parallel(self, prompts: list[str]):
        warn("Unfortunately, ollama does not handle parallelization of requests. No parallelization will occur.")
        return [self.embed(prompt) for prompt in tqdm(prompts)]

    def predict(self, prompt: str, temperature: float):
        ollama.pull(self.model_ref_name)
        return self.predict_messages(messages=[{"role": "user", "content": prompt}], temperature=temperature)

    def predict_batch_parallel(self, prompts: list[str], temperature: float):
        warn("Unfortunately, ollama does not handle parallelization of requests. No parallelization will occur.")
        return [self.predict(prompt, temperature) for prompt in tqdm(prompts)]

    def predict_messages(self, messages: list[dict[str, str]], temperature: float):
        warn("Unfortunately, ollama does not handle changing dynamically parameters like temperature (llamma.cpp does.")
        return ollama.chat(model=self.model_ref_name, messages=messages)["message"]["content"]

    def predict_messages_batch_parallel(self, list_messages: list[list[dict[str, str]]], temperature: float):
        warn("Unfortunately, ollama does not handle parallelization of requests. No parallelization will occur.")
        return [self.predict_messages(messages, temperature) for messages in tqdm(list_messages)]
