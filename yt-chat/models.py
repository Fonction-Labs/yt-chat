from langchain_openai import ChatOpenAI
from llama_cpp import Llama
from langchain_community.llms import CTransformers


class ChatOpenAILLM:
    def __init__(self, model_name, temperature):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def predict(self, prompt, temperature):
        return self.llm.invoke(prompt).content


class LlamaLLM:
    def __init__(self, model_path, n_ctx, n_threads, n_gpu_layers):
        #self.llm = Llama.from_pretrained(
        #    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        #    filename="*Q5_K_S.gguf",
        #    n_ctx=n_ctx,
        #    n_threads=n_threads,
        #    n_gpu_layers=n_gpu_layers,
        #)

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )

    def predict(self, prompt, temperature):
        return self.llm.create_completion(
            f"<s>[INST] {prompt} [/INST]", max_tokens=None, temperature=temperature
        )["choices"][0]["text"]
