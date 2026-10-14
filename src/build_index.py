from pathlib import Path

import pandas as pd

from document_builder import build_documents


DATA_PATH = Path(
    "data/processed/df_reviews_rag.parquet"
)


def load_data(
    data_path: Path = DATA_PATH
) -> pd.DataFrame:
    """
    Carrega o dataset já processado para o RAG.
    """

    print("Carregando dataset...")

    df = pd.read_parquet(data_path)

    print(
        f"Dataset carregado com "
        f"{len(df):,} registros."
    )

    return df


def build_rag_index():
    
    df = load_data()

    
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

    

if __name__ == "__main__":

    build_rag_index()
