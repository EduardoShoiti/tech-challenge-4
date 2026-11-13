from typing import Protocol


class LLM(Protocol):
    """Contrato mínimo esperado pelo pipeline RAG."""

    def generate(self, prompt: str) -> str:
        """Gera uma resposta textual para o prompt recebido."""
        ...
