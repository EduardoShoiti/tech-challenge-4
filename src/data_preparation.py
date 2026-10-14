from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "olist_order_reviews_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "df_reviews_rag.parquet"

GENERIC_REVIEWS = [
    "bom",
    "muito bom",
    "ótimo",
    "otimo",
    "ótima",
    "otima",
    "excelente",
    "perfeito",
    "gostei",
    "adorei",
    "muito obrigado",
    "boa",
    "maravilha",
    "amei",
    "ok",
    "satisfeito",
    "satisfeita",
    "recomendo",
    "médio",
    "medio",
    "ruim",
    "péssimo",
    "pessimo",
    "horrível",
    "horrivel",
    "não gostei",
    "nao gostei",
]


def prepare_reviews(df_order_reviews: pd.DataFrame) -> pd.DataFrame:

    df_order_reviews = df_order_reviews.copy()

    df_order_reviews["review_title"] = (
        df_order_reviews["review_comment_title"].fillna("").str.strip()
    )
    df_order_reviews["review_message"] = (
        df_order_reviews["review_comment_message"].fillna("").str.strip()
    )

    df_order_reviews["review_text"] = (
        df_order_reviews["review_title"].str.strip().fillna("")
        + ". "
        + df_order_reviews["review_message"].str.strip().fillna("")
    ).str.strip(" .")

    df_order_reviews["count_chars_review_text"] = (
        df_order_reviews["review_text"].str.len().fillna(0).astype(int)
    )
    df_order_reviews["count_words_review_text"] = (
        df_order_reviews["review_text"].str.split().str.len().fillna(0).astype(int)
    )

    df_order_reviews["normalized_review_text"] = (
        df_order_reviews["review_text"]
        .fillna("")
        .str.lower()
        .str.replace(r"[^a-zA-Zá-úÁ-Úà-ùâ-ûã-õç0-9\s]", "", regex=True)
        .str.strip()
    )

    conditions = [
        df_order_reviews["count_chars_review_text"] == 0,
        df_order_reviews["count_words_review_text"] <= 1,
        df_order_reviews["normalized_review_text"].isin(GENERIC_REVIEWS),
    ]
    df_order_reviews["use_in_rag_reason"] = np.select(
        conditions,
        ["no_text", "one_word_review", "generic_review"],
        default="valid",
    )
    df_order_reviews["use_in_rag"] = df_order_reviews["use_in_rag_reason"] == "valid"

    df_reviews_rag = df_order_reviews[df_order_reviews["use_in_rag"]].copy()
    df_reviews_rag.drop_duplicates(subset=["review_text"], keep="first", inplace=True)
    return df_reviews_rag


def run(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    
    df_order_reviews = pd.read_csv(input_path)
    df_reviews_rag = prepare_reviews(df_order_reviews)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_reviews_rag.to_parquet(output_path, index=False)
    return df_reviews_rag


if __name__ == "__main__":
    run()
