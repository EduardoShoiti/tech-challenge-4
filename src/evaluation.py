import json
import re
import unicodedata

import pandas as pd

from config import MAX_BEST_DISTANCE, MAX_MEAN_DISTANCE
from retriever import retrieve_with_validation


def load_evaluation_questions(
    path="evaluation/retrieval_questions.json"
):
    """Carrega as perguntas utilizadas na avaliação do retrieval."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    """Normaliza texto para a comparação simples de termos esperados."""

    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def calculate_relevance_metrics(
    documents,
    expected_terms
):
    """
    Calcula métricas de relevância por correspondência de termos.

    Essa avaliação é uma heurística simples e reproduzível. Um documento é
    considerado relevante quando contém ao menos um dos termos esperados.
    """

    if not expected_terms:
        return {
            "hit_at_k": None,
            "precision_at_k": None,
            "reciprocal_rank": None,
            "relevant_documents": None,
        }

    normalized_terms = [
        normalize_text(term)
        for term in expected_terms
    ]
    relevant_positions = []

    for position, document in enumerate(documents, start=1):
        normalized_document = normalize_text(document)

        if any(
            term in normalized_document
            for term in normalized_terms
        ):
            relevant_positions.append(position)

    total_documents = len(documents)
    relevant_documents = len(relevant_positions)

    return {
        "hit_at_k": int(relevant_documents > 0),
        "precision_at_k": (
            relevant_documents / total_documents
            if total_documents
            else 0.0
        ),
        "reciprocal_rank": (
            1 / relevant_positions[0]
            if relevant_positions
            else 0.0
        ),
        "relevant_documents": relevant_documents,
    }


def evaluate_retrieval(
    questions,
    collection,
    embedding_model,
    top_k=5,
    top_n=3,
    max_best_distance=MAX_BEST_DISTANCE,
    max_mean_distance=MAX_MEAN_DISTANCE,
):
    """Executa as perguntas e reúne métricas de escopo e relevância."""

    evaluation_results = []

    for item in questions:
        question = item["question"]
        result = retrieve_with_validation(
            question=question,
            collection=collection,
            embedding_model=embedding_model,
            top_k=top_k,
            top_n=top_n,
            max_best_distance=max_best_distance,
            max_mean_distance=max_mean_distance,
        )

        documents = result["results"]["documents"][0]
        relevance = calculate_relevance_metrics(
            documents=documents,
            expected_terms=item.get("expected_terms", []),
        )
        predicted_scope = (
            "in_scope"
            if result["has_sufficient_evidence"]
            else "out_of_scope"
        )

        evaluation_results.append({
            "id": item["id"],
            "question": question,
            "expected_scope": item["expected_scope"],
            "predicted_scope": predicted_scope,
            "scope_correct": predicted_scope == item["expected_scope"],
            "best_distance": result["best_distance"],
            "mean_distance": result["mean_distance"],
            "hit_at_k": relevance["hit_at_k"],
            "precision_at_k": relevance["precision_at_k"],
            "reciprocal_rank": relevance["reciprocal_rank"],
            "relevant_documents": relevance["relevant_documents"],
            "distances": result["results"]["distances"][0],
        })

    return pd.DataFrame(evaluation_results)


def calculate_classification_metrics(
    expected_scopes,
    predicted_scopes
):
    """Calcula métricas considerando in_scope como a classe positiva."""

    pairs = list(zip(expected_scopes, predicted_scopes))
    true_positive = sum(
        expected == "in_scope" and predicted == "in_scope"
        for expected, predicted in pairs
    )
    false_positive = sum(
        expected == "out_of_scope" and predicted == "in_scope"
        for expected, predicted in pairs
    )
    false_negative = sum(
        expected == "in_scope" and predicted == "out_of_scope"
        for expected, predicted in pairs
    )
    true_negative = sum(
        expected == "out_of_scope" and predicted == "out_of_scope"
        for expected, predicted in pairs
    )

    total = len(pairs)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": (
            (true_positive + true_negative) / total
            if total
            else 0.0
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def calibrate_thresholds(df_evaluation):
    """Encontra thresholds candidatos usando as distâncias já calculadas."""

    best_candidates = sorted(
        df_evaluation["best_distance"].dropna().unique()
    )
    mean_candidates = sorted(
        df_evaluation["mean_distance"].dropna().unique()
    )
    candidates = []

    for best_threshold in best_candidates:
        for mean_threshold in mean_candidates:
            predicted_scopes = [
                (
                    "in_scope"
                    if best_distance <= best_threshold
                    and mean_distance <= mean_threshold
                    else "out_of_scope"
                )
                for best_distance, mean_distance in zip(
                    df_evaluation["best_distance"],
                    df_evaluation["mean_distance"],
                )
            ]
            metrics = calculate_classification_metrics(
                expected_scopes=df_evaluation["expected_scope"],
                predicted_scopes=predicted_scopes,
            )
            candidates.append({
                "max_best_distance": float(best_threshold),
                "max_mean_distance": float(mean_threshold),
                **metrics,
            })

    return max(
        candidates,
        key=lambda candidate: (
            candidate["f1"],
            candidate["precision"],
            candidate["accuracy"],
            -candidate["max_best_distance"],
            -candidate["max_mean_distance"],
        ),
    )


def print_evaluation_summary(df_evaluation):
    """Exibe resultados, métricas de escopo, relevância e calibração."""

    print("\n=== RESULTADOS ===\n")
    print(
        df_evaluation[
            [
                "id",
                "question",
                "expected_scope",
                "predicted_scope",
                "best_distance",
                "mean_distance",
                "hit_at_k",
                "precision_at_k",
                "reciprocal_rank",
            ]
        ].to_string(index=False)
    )

    classification = calculate_classification_metrics(
        expected_scopes=df_evaluation["expected_scope"],
        predicted_scopes=df_evaluation["predicted_scope"],
    )
    print("\n=== MÉTRICAS DE ESCOPO ===\n")
    print(pd.Series(classification).to_string())

    relevance = df_evaluation[
        df_evaluation["expected_scope"] == "in_scope"
    ]
    print("\n=== MÉTRICAS DE RELEVÂNCIA ===\n")
    print(
        relevance[
            ["hit_at_k", "precision_at_k", "reciprocal_rank"]
        ].mean().to_string()
    )

    calibrated = calibrate_thresholds(df_evaluation)
    print("\n=== THRESHOLDS CANDIDATOS ===\n")
    print(pd.Series(calibrated).to_string())
