import os
from dotenv import load_dotenv
import logging

load_dotenv(override=True)

CONFIG = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'), # Your API key for accessing OpenAI's API. This key is required for using the LLM services.
    'CSV_DIRECTORY': os.getenv('CSV_DIRECTORY', '../biokg_nebula'), # The directory path where your CSV files containing triplets are stored. If not provided, default to relative path.
    'PERSIST_DIR': './storage_graph', # The directory path where the graph storage context will be persisted.
    'TEST_MODE': bool(int(os.getenv('TEST_MODE', 1)))  # Set to 1 for test mode, 0 for full processing
}

NEBULA_GRAPH_CONFIG = {
    'GRAPHD_HOST': os.getenv('GRAPHD_HOST', '127.0.0.1'), # The hostname or IP address of the NebulaGraph server. 
    'GRAPHD_PORT': os.getenv('GRAPHD_PORT', '9669'), # The Port number of the NebulaGraph server.
    'NEBULA_USER': os.getenv('NEBULA_USER', 'root'), # The username for accessing NebulaGraph
    'NEBULA_PASSWORD': os.getenv('NEBULA_PASSWORD', 'nebula'), # The password for accessing NebulaGraph
    'SPACE_NAME': os.getenv("SPACE_NAME", 'biokg_db'), # The name of the space (database) in NebulaGraph where the triplets will be stored.
    'INCLUDE_EMBEDDINGS': False # Note: This is a expensive and time consuming operation. Setting it to False by default.
}
# SentenceSplitter(chunk_size=512, chunk_overlap=20)
if CONFIG["TEST_MODE"] == 0:
    NEBULA_GRAPH_CONFIG['INCLUDE_EMBEDDINGS'] = False
    logging.info("""
                Setting INCLUDE_EMBEDDINGS to False in test mode. Since this is a time consuming operation, setting it to False by default in test mode.    
                Manually set it to True if you want to include embeddings within this condition and in the NEBULA_GRAPH_CONFIG
                """
            )

TEXT_PROCESSING_CONFIG = {
    'CHUNK_SIZE': 1000,
    'CHUNK_OVERLAP': 20,
    'MAX_TRIPLETS_PER_CHUNK': 10, # The maximum number of triplets to process in a single chunk
}

LLM_CONFIG = {
    'COMPLETION_MODEL_NAME': 'gpt-4o-mini',
    'EMBED_MODEL_NAME': 'text-embedding-3-small',
    'NUM_TOKENS': 1024,
    'TEMPERATURE': 0.0,
    'CONTEXT_WINDOW': 3900
}

KG_PROPERTIES = {
    'TAGS': ['entity'],
    'REL_PROP_NAMES': ['relationship'],
    'EDGE_TYPES': ['relationship']
}


    
    