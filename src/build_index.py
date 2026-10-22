from pathlib import Path

import pandas as pd

from document_builder import build_documents
from embeddings import (
    load_embedding_model,
    generate_embeddings
)
from build_vector_store import create_vector_store
from retriever import retrieve_reviews


DATA_PATH = Path(
    "data/processed/df_reviews_rag.parquet"
)

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "olist_reviews"


def load_data(
    data_path: Path = DATA_PATH
) -> pd.DataFrame:
    """
    Carrega o dataset já processado para o RAG.
    """

    print("Carregando dataset...")

    df = pd.read_parquet(data_path).head(50)

    print(
        f"Dataset carregado com "
        f"{len(df):,} registros."
    )

    return df


def build_rag_index():
    """
    Executa todo o pipeline de indexação:

    DataFrame
        -> documentos
        -> embeddings
        -> ChromaDB

    Returns
    -------
    collection
        Collection criada no ChromaDB.

    model
        Modelo de embeddings carregado.
    """

    # Carrega dataset
    df = load_data()

    # Cria documentos
    print("\nCriando documentos...")

    documents = build_documents(
        df=df,
        id_column="review_id",
        text_column="review_text",
        metadata_columns=[
            "review_score",
        ]
    )

    print(
        f"{len(documents):,} documentos criados."
    )

    print(documents[0:5])

    # Carrega modelo
    print("\nCarregando modelo de embeddings...")

    model = load_embedding_model()

    # Gera embeddings
    print("\nGerando embeddings...")

    embeddings = generate_embeddings(
        documents=documents,
        model=model,
        batch_size=64,
    )

    print(
        f"Embeddings gerados: "
        f"{embeddings.shape}"
    )

    print(embeddings[0:5])

    # Cria Vector Store
    print("\nCriando Vector Store...")

    collection = create_vector_store(
        documents=documents,
        embeddings=embeddings,
        persist_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )

    print("\nPipeline concluído com sucesso.")

    return collection, model


def test_retrieval(
    collection,
    model
):
    """
    Executa uma busca simples para validar se o índice vetorial está funcionando.
    """

    question = (
        "Quais problemas os clientes relatam "
        "sobre atraso na entrega?"
    )

    print("\nTeste de retrieval")
    print(f"Pergunta: {question}")

    results = retrieve_reviews(
        question=question,
        collection=collection,
        embedding_model=model,
        top_k=5,
    )

    for i, (
        text,
        metadata,
        distance
    ) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1
    ):

        print(
            f"\n--- Resultado {i} ---"
        )

        print(
            f"Distância: {distance:.4f}"
        )

        print(
            f"Metadata: {metadata}"
        )

        print(
            f"Review: {text}"
        )


if __name__ == "__main__":

    build_rag_index()
    collection, model = build_rag_index()

    test_retrieval(
        collection=collection,
        model=model
    )