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