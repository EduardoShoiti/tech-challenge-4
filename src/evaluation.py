import json

import pandas as pd

from retriever import retrieve_with_validation


def load_evaluation_questions(
    path="evaluation/retrieval_questions.json"
):
    """
    Carrega as perguntas utilizadas na avaliação
    do retrieval.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def evaluate_retrieval(
    questions,
    collection,
    embedding_model,
    top_k=5,
    top_n=3
):
    """
    Executa as perguntas de avaliação e cria
    uma tabela com as métricas do retrieval.
    """

    evaluation_results = []

    for item in questions:

        question = item["question"]

        result = retrieve_with_validation(
            question=question,
            collection=collection,
            embedding_model=embedding_model,
            top_k=top_k,
            top_n=top_n,

            # Calibrando os thresholds
            max_best_distance=None,
            max_mean_distance=None
        )

        evaluation_results.append({

            "id":
                item["id"],

            "question":
                question,

            "expected_scope":
                item["expected_scope"],

            "best_distance":
                result["best_distance"],

            "mean_distance":
                result["mean_distance"],

            "distances":
                result["results"]["distances"][0]
        })

    return pd.DataFrame(
        evaluation_results
    )


def print_evaluation_summary(
    df_evaluation
):
    """
    Exibe resumo estatístico das distâncias
    por tipo de pergunta.
    """

    print(
        "\n=== RESULTADOS ===\n"
    )

    print(
        df_evaluation[
            [
                "question",
                "expected_scope",
                "best_distance",
                "mean_distance"
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== RESUMO POR GRUPO ===\n"
    )

    summary = (
        df_evaluation
        .groupby("expected_scope")
        [
            [
                "best_distance",
                "mean_distance"
            ]
        ]
        .agg([
            "min",
            "mean",
            "median",
            "max"
        ])
    )

    print(summary)