# Paths
DATA_DIR = "data"
PRODUCTS_PATH = f"{DATA_DIR}/products.csv"
ORDERS_PATH = f"{DATA_DIR}/orders.csv"
CHROMA_DIR = f"{DATA_DIR}/chroma_index"

# Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG
LLM_MODEL_RAG = "mistral"
TEMPERATURE_RAG = 0.2

# Agent
LLM_MODEL_AGENT = "mistral:instruct"
TEMPERATURE_AGENT = 0
