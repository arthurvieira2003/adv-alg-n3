import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configurações do projeto"""
    
    # Google Gemini API
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    # Modelo LLM
    LLM_MODEL = "gemini-2.5-flash-lite-preview-06-17"
    LLM_TEMPERATURE = 0.1
    
    # Embeddings
    EMBEDDING_MODEL = "models/embedding-001"
    
    # Paths
    DATA_DIR = "data"
    GRAPH_FILE = "knowledge_graph.json"
    
    # Graph settings
    MAX_NODES = 1000
    MAX_RELATIONSHIPS = 5000
    
    # RAG settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 5