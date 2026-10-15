from sentence_transformers import SentenceTransformer


def retrieve_reviews(
    question: str,
    collection,
    embedding_model: SentenceTransformer,
    top_k: int = 5,
    where: dict | None = None,
):
    """
    Recupera as reviews semanticamente mais
    próximas da pergunta.

    Parameters
    ----------
    question:
        Pergunta ou consulta do usuário.

    collection:
        Collection do ChromaDB.

    embedding_model:
        Mesmo modelo usado para gerar os embeddings
        dos documentos.

    top_k:
        Quantidade de documentos recuperados.

    where:
        Filtro opcional por metadados.

    Returns
    -------
    dict
        Resultado retornado pelo ChromaDB.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "A pergunta não pode estar vazia."
        )

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_args = {
        "query_embeddings": [
            query_embedding.tolist()
        ],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where is not None:
        query_args["where"] = where

    results = collection.query(
        **query_args
    )

    return results