from google import genai
from google.genai import types


class GeminiLLM:
    """Adaptador da Gemini API para o contrato usado pelo projeto."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        client=None,
    ):
        if not model_name.strip():
            raise ValueError("O nome do modelo não pode estar vazio.")

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = client or genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        prompt = prompt.strip()

        if not prompt:
            raise ValueError("O prompt não pode estar vazio.")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )

        answer = response.text

        if not answer or not answer.strip():
            raise RuntimeError("O modelo não retornou uma resposta textual.")

        return answer.strip()
