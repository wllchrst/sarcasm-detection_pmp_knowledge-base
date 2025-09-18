import httpx
from llm.base_llm import BaseLLM
from helpers import env_helper
from ollama import Client

timeout_seconds = 180

class OllamaLLM(BaseLLM):
    def __init__(self, llm_model: str = None):
        super().__init__(llm_model)
        self.HOST = env_helper.OLLAMA_HOST
        self.client = Client(host=self.HOST, timeout=timeout_seconds)

    def answer(self, system_prompt: str, prompt: str) -> str:
        try:
            response = self.client.chat(self.llm_model, think=False, stream=False, messages=[
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])

            return response.message.content
        except httpx.ReadTimeout as timeout:
            raise TimeoutError(
                f'Ollama request exceeded timeout limit for model {self.model_name} with error: {timeout}')
        except Exception as e:
            raise RuntimeError(f"Error when processing in ollama: {str(e)}")
