from llm import LLM
from prompts import FALLBACK_ANSWER, build_rag_prompt
from retriever import retrieve_with_validation


def extract_evidences(results: dict) -> list[dict]:
    """Converte o retorno aninhado do Chroma em uma lista simples."""

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "id": document_id,
            "text": text,
            "metadata": metadata,
            "distance": distance,
        }
        for document_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        )
    ]


def answer_question(
    question: str,
    collection,
    embedding_model,
    llm: LLM,
    **retrieval_options,
) -> dict:
    """Executa retrieval, validação de evidência e geração da resposta."""

    retrieval = retrieve_with_validation(
        question=question,
        collection=collection,
        embedding_model=embedding_model,
        **retrieval_options,
    )

    response = {
        "question": question,
        "answer": FALLBACK_ANSWER,
        "has_sufficient_evidence": retrieval["has_sufficient_evidence"],
        "best_distance": retrieval["best_distance"],
        "mean_distance": retrieval["mean_distance"],
        "evidences": [],
    }

    if not retrieval["has_sufficient_evidence"]:
        return response

    evidences = extract_evidences(retrieval["results"])
    prompt = build_rag_prompt(
        question=question,
        evidences=evidences,
    )

    response["answer"] = llm.generate(prompt)
    response["evidences"] = evidences

    return response
