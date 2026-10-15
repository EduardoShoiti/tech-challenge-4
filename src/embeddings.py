import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def load_embedding_model(
    model_name: str = MODEL_NAME
) -> SentenceTransformer:
    """
    Carrega o modelo utilizado para gerar embeddings.
    """

    model = SentenceTransformer(model_name)

    return model


def generate_embeddings(
    documents: list[dict],
    model: SentenceTransformer,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """
    Gera embeddings para uma lista de documentos.

    Parameters
    ----------
    documents:
        Documentos criados pelo document_builder.

    model:
        SentenceTransformer carregado.

    batch_size:
        Quantidade de textos processados por lote.

    normalize_embeddings:
        Normaliza os vetores. Útil para comparação
        baseada em similaridade.

    Returns
    -------
    np.ndarray
        Matriz no formato:

        (quantidade_documentos, dimensao_embedding)
    """

    if not documents:
        raise ValueError(
            "Nenhum documento foi fornecido."
        )

    texts = [
        document["text"]
        for document in documents
    ]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    return embeddings