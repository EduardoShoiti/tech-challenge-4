import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from build_vector_store import load_vector_store
from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
)
from embeddings import load_embedding_model
from gemini_llm import GeminiLLM
from rag_pipeline import answer_question


def main():
    parser = argparse.ArgumentParser(
        description="Consulta avaliações da Olist usando RAG."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Pergunta sobre as avaliações dos clientes.",
    )
    args = parser.parse_args()

    llm = GeminiLLM(
        model_name=LLM_MODEL_NAME,
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
    )
    embedding_model = load_embedding_model()
    collection = load_vector_store(
        persist_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )

    result = answer_question(
        question=" ".join(args.question),
        collection=collection,
        embedding_model=embedding_model,
        llm=llm,
    )

    print(f"\nResposta:\n{result['answer']}")

    if result["evidences"]:
        print("\nEvidências recuperadas:")

        for index, evidence in enumerate(result["evidences"], start=1):
            score = evidence["metadata"].get("review_score", "-")
            print(
                f"\n[Review {index}] "
                f"nota={score} "
                f"distância={evidence['distance']:.4f}\n"
                f"{evidence['text']}"
            )


if __name__ == "__main__":
    main()
