from pathlib import Path

import chromadb
import numpy as np


DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "olist_reviews"


def create_vector_store(
    documents: list[dict],
    embeddings: np.ndarray,
    persist_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 5000,
):
    """
    Cria uma Vector Store persistente usando ChromaDB.

    Armazena:
        - ID
        - texto
        - metadados
        - embedding

    Parameters
    ----------
    documents:
        Lista de documentos.

    embeddings:
        Matriz de embeddings correspondente aos documentos.

    persist_path:
        Pasta onde o ChromaDB será persistido.

    collection_name:
        Nome da collection.

    batch_size:
        Quantidade de documentos inseridos por lote.

    Returns
    -------
    chromadb.Collection
    """

    if len(documents) != len(embeddings):
        raise ValueError(
            "A quantidade de documentos é diferente "
            "da quantidade de embeddings."
        )

    Path(persist_path).mkdir(
        parents=True,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=persist_path
    )

    collection = client.get_or_create_collection(
        name=collection_name
    )

    total = len(documents)

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start + batch_size,
            total
        )

        batch_documents = documents[start:end]
        batch_embeddings = embeddings[start:end]

        ids = [
            document["id"]
            for document in batch_documents
        ]

        texts = [
            document["text"]
            for document in batch_documents
        ]

        metadatas = [
            document["metadata"]
            for document in batch_documents
        ]

        vectors = (
            batch_embeddings
            .tolist()
        )

        # upsert permite executar novamente sem
        # criar registros duplicados.
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )

        print(
            f"Indexados {end}/{total} documentos"
        )

    print(
        f"\nVector Store criada com "
        f"{collection.count()} documentos."
    )

    return collection