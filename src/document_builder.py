import pandas as pd
import numpy as np


def _to_python_type(value):
    """
    Converte valores do Pandas/Numpy para tipos simples do Python,
    compatíveis com os metadados do ChromaDB.
    """

    if pd.isna(value):
        return None

    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.to_datetime(value).date())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def build_documents(
    df: pd.DataFrame,
    id_column: str = "review_id",
    text_column: str = "review_text",
    metadata_columns: list[str] | None = None,
) -> list[dict]:
    """
    Converte um DataFrame em documentos para o pipeline RAG.

    Cada documento possui:

        {
            "id": "...",
            "text": "...",
            "metadata": {...}
        }

    Parameters
    ----------
    df:
        DataFrame processado.

    id_column:
        Coluna que identifica unicamente a review.

    text_column:
        Coluna contendo o texto final da review.

    metadata_columns:
        Colunas que serão armazenadas como metadados.

    Returns
    -------
    list[dict]
        Lista de documentos.
    """

    if metadata_columns is None:
        metadata_columns = []

    required_columns = [
        id_column,
        text_column,
        *metadata_columns
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colunas não encontradas no DataFrame: "
            f"{missing_columns}"
        )

    # ChromaDB exige IDs únicos.
    if df[id_column].duplicated().any():
        duplicated = (
            df.loc[
                df[id_column].duplicated(keep=False),
                id_column
            ]
            .astype(str)
            .unique()[:10]
        )

        raise ValueError(
            f"A coluna '{id_column}' possui IDs duplicados. "
            f"Exemplos: {duplicated}. "
            "Verifique os joins antes de construir os documentos."
        )

    documents = []

    for _, row in df.iterrows():

        text = row[text_column]

        # Ignora documentos vazios por segurança.
        if pd.isna(text):
            continue

        text = str(text).strip()

        if not text:
            continue

        metadata = {}

        for column in metadata_columns:

            value = _to_python_type(row[column])

            # Não envia None ao Chroma.
            if value is not None:
                metadata[column] = value

        document = {
            "id": str(row[id_column]),
            "text": text,
            "metadata": metadata,
        }

        documents.append(document)

    return documents