from evaluation import (
    load_evaluation_questions,
    evaluate_retrieval,
    print_evaluation_summary
)

from embeddings import (
    load_embedding_model
)

from build_vector_store import (
    load_vector_store
)

from config import (
    CHROMA_PATH,
    COLLECTION_NAME
)


# Carregar modelo
model = load_embedding_model()


# Abrir Vector Store existente
collection = load_vector_store(
    persist_path=CHROMA_PATH,
    collection_name=COLLECTION_NAME
)


# Carregar perguntas
questions = load_evaluation_questions()


# Executar avaliação
df_evaluation = evaluate_retrieval(
    questions=questions,
    collection=collection,
    embedding_model=model,
    top_k=5,
    top_n=3
)


# Exibir resultados
print_evaluation_summary(
    df_evaluation
)