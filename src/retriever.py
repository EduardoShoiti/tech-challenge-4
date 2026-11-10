from config import (
    MAX_BEST_DISTANCE,
    MAX_MEAN_DISTANCE,
    RETRIEVAL_TOP_K,
    EVIDENCE_TOP_N
)

def evaluate_evidence(
    distances,
    max_best_distance=None,
    max_mean_distance=None,
    top_n=3
):

    if not distances:
        return {
            "best_distance": None,
            "mean_distance": None,
            "has_sufficient_evidence": False
        }

    selected = distances[:top_n]

    best_distance = min(selected)

    mean_distance = (
        sum(selected)
        / len(selected)
    )

    if (
        max_best_distance is None
        or max_mean_distance is None
    ):

        sufficient = None

    else:

        sufficient = (
            best_distance
            <= max_best_distance
            and
            mean_distance
            <= max_mean_distance
        )

    return {
        "best_distance":
            best_distance,

        "mean_distance":
            mean_distance,

        "has_sufficient_evidence":
            sufficient
    }


def retrieve_reviews(
    question,
    collection,
    embedding_model,
    top_k=5,
    where=None
):

    question = question.strip()

    if not question:
        raise ValueError(
            "A pergunta não pode estar vazia."
        )

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_args = {
        "query_embeddings": [
            query_embedding.tolist()
        ],

        "n_results":
            top_k,

        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if where is not None:
        query_args["where"] = where

    return collection.query(
        **query_args
    )


def retrieve_with_validation(
    question,
    collection,
    embedding_model,
    top_k=RETRIEVAL_TOP_K,
    top_n=EVIDENCE_TOP_N,
    max_best_distance=MAX_BEST_DISTANCE,
    max_mean_distance=MAX_MEAN_DISTANCE,
    where=None
):

    results = retrieve_reviews(
        question=question,
        collection=collection,
        embedding_model=embedding_model,
        top_k=top_k,
        where=where
    )

    evidence = evaluate_evidence(
        distances=results["distances"][0],
        top_n=top_n,
        max_best_distance=max_best_distance,
        max_mean_distance=max_mean_distance
    )

    return {
        "question":
            question,

        "best_distance":
            evidence["best_distance"],

        "mean_distance":
            evidence["mean_distance"],

        "has_sufficient_evidence":
            evidence[
                "has_sufficient_evidence"
            ],

        "results":
            results
    }