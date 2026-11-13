import os


EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "olist_reviews"

MAX_BEST_DISTANCE = 0.65
MAX_MEAN_DISTANCE = 0.66

RETRIEVAL_TOP_K = 5
EVIDENCE_TOP_N = 3

LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "gemini-3.1-flash-lite"
)
LLM_TEMPERATURE = 0.2
LLM_MAX_OUTPUT_TOKENS = 1024
