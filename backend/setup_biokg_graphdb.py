import os
import csv
import logging
import sys
import multiprocessing
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.core.storage import StorageContext
from llama_index.core import KnowledgeGraphIndex
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import Settings

from config import CONFIG, KG_PROPERTIES, LLM_CONFIG, TEXT_PROCESSING_CONFIG, NEBULA_GRAPH_CONFIG

from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import traceback
import time

load_dotenv(override=True)


TAGS = KG_PROPERTIES['TAGS']
REL_PROP_NAMES = KG_PROPERTIES['REL_PROP_NAMES']
EDGE_TYPES = KG_PROPERTIES['EDGE_TYPES']

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up LLM and service context
Settings.llm = OpenAI(model=LLM_CONFIG['COMPLETION_MODEL_NAME'], temperature=LLM_CONFIG['TEMPERATURE'], api_key=CONFIG['OPENAI_API_KEY'])
Settings.embed_model = OpenAIEmbedding(model=LLM_CONFIG['EMBED_MODEL_NAME'], api_key=CONFIG['OPENAI_API_KEY'])
Settings.node_parser = SentenceSplitter(chunk_size=TEXT_PROCESSING_CONFIG['CHUNK_SIZE'], chunk_overlap=TEXT_PROCESSING_CONFIG['CHUNK_OVERLAP'])
Settings.num_output = LLM_CONFIG['NUM_TOKENS']
Settings.context_window = LLM_CONFIG['CONTEXT_WINDOW']


# NebulaGraph connection setup
nebula_config = Config()
nebula_config.max_connection_pool_size = 10
connection_pool = ConnectionPool()
ok = connection_pool.init([(NEBULA_GRAPH_CONFIG['GRAPHD_HOST'], NEBULA_GRAPH_CONFIG["GRAPHD_PORT"])], nebula_config)

if not ok:
    logging.error("Failed to initialize the connection pool.")
    sys.exit(1)

session = connection_pool.get_session(NEBULA_GRAPH_CONFIG['NEBULA_USER'], NEBULA_GRAPH_CONFIG['NEBULA_PASSWORD'])

# Create space and schema if not exists
try:
    session.execute(f'CREATE SPACE IF NOT EXISTS {NEBULA_GRAPH_CONFIG["SPACE_NAME"]}(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);')
    session.execute(f'USE {NEBULA_GRAPH_CONFIG["SPACE_NAME"]};')
    session.execute('CREATE TAG IF NOT EXISTS entity(name string);')
    session.execute('CREATE EDGE IF NOT EXISTS relationship(relationship string);')
    session.execute('CREATE TAG INDEX IF NOT EXISTS entity_index ON entity(name(256));')
    # Wait to ensure the space is ready. Note: If the space name is not created, it takes some time to be create the space and load the schiema.
    time.sleep(10)
except Exception as e:
    logging.error(f"Error creating space or schema: {e}")
    sys.exit(1)

def initialize_kg_index():
    tags = TAGS
    edge_types, rel_prop_names = EDGE_TYPES, REL_PROP_NAMES
    
    graph_store = NebulaGraphStore(
        space_name=NEBULA_GRAPH_CONFIG['SPACE_NAME'],
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,  # Associate each edge type with the default property name
        tags=tags,
    )
    
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    kg_index = KnowledgeGraphIndex(
        [],
        storage_context=storage_context,
        llm=Settings.llm, # Note: This is default, but specifying for visibility
        embed_model=Settings.embed_model,  # Note: This is default, but specifying for visibility
        max_triplets_per_chunk=TEXT_PROCESSING_CONFIG['MAX_TRIPLETS_PER_CHUNK'],
        include_embeddings=NEBULA_GRAPH_CONFIG['INCLUDE_EMBEDDINGS'],
        space_name=NEBULA_GRAPH_CONFIG['SPACE_NAME'],
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        show_progress=True
    )
    
    return kg_index

def read_triplets_from_file(file_path):
    """
    Reads triplets from a single CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list of tuple: A list of triplets, where each triplet is a tuple of strings.
    """
    triplets = []
    try:
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 3:
                    triplets.append(tuple(row))
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
    return triplets

def chunk_triplets(triplets, chunk_size):
    """
    Divides a list of triplets into smaller chunks.

    Args:
        triplets (list of tuple): The list of triplets to be chunked.
        chunk_size (int): The maximum size of each chunk.

    Returns:
        list of list of tuple: A list of chunks, where each chunk is a list of triplets.
    """
    chunks = []
    chunk = []
    for triplet in triplets:
        chunk.append(triplet)
        if len(chunk) >= chunk_size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks

def load_triplets_from_directory(directory):
    """
    Loads triplets from all CSV files in the specified directory.

    Args:
        directory (str): The directory containing CSV files.

    Returns:
        list of tuple: A list of triplets read from all CSV files in the directory.
    """
    all_triplets = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory, filename)
                triplets = read_triplets_from_file(file_path)
                all_triplets.extend(triplets)
    except Exception as e:
        logging.error(f"Error loading triplets from directory {directory}: {e}")
    return all_triplets

def assign_chunks_with_ids(chunks):
    """
    Assigns an ID to each chunk for tracking.

    Args:
        chunks (list of list of tuple): The list of chunks to which IDs will be assigned.

    Returns:
        list of tuple: A list of tuples, where each tuple contains a chunk and its ID.
    """
    return [(chunk, chunk_id) for chunk_id, chunk in enumerate(chunks)]

def load_triplets_in_chunks():
    """
    Main function to load and chunk triplets.

    Returns:
        list of tuple: A list of tuples, where each tuple contains a chunk and its ID. If in TEST_MODE, only the first chunk is returned.
    """
    triplets = load_triplets_from_directory(CONFIG['CSV_DIRECTORY'])
    logging.info(f"Loaded {len(triplets)} triplets from CSV files.")
    
    if not triplets:
        logging.info("No triplets found to process.")
        return []
    
    existing_triplets = set(fetch_all_triplets(session, NEBULA_GRAPH_CONFIG['SPACE_NAME']))
    logging.info(f"Fetched {len(existing_triplets)} existing triplets from NebulaGraph for comparison.")

    filtered_triplets = [triplet for triplet in triplets if triplet not in existing_triplets]
    logging.info(f"Filtered out {len(triplets) - len(filtered_triplets)} duplicate triplets. {len(filtered_triplets)} new triplets to process.")
    
    chunks = chunk_triplets(filtered_triplets, TEXT_PROCESSING_CONFIG['CHUNK_SIZE'])

    if not filtered_triplets:
        logging.info("No new triplets to upsert.")
        return []
    
    if CONFIG['TEST_MODE']:
        # Only return the first chunk for testing
        return assign_chunks_with_ids(chunks[:1])
    else:
        # Return all chunks
        return assign_chunks_with_ids(chunks)
    
def fetch_all_triplets(session, space_name=NEBULA_GRAPH_CONFIG['SPACE_NAME']):
    """
    Fetches all the triplets from the NebulaGraph database.

    Args:
        session: The NebulaGraph session object used to execute queries.
        space_name (str): The name of the space (database) from which to fetch triplets.

    Returns:
        list of tuple: A list of triplets, where each triplet is a tuple (subject, relationship, object).
    """
    logging.info(f"Fetching all triplets from space: {space_name}")

    query = f"""
    USE {space_name};
    MATCH (subject:entity)-[rel:relationship]->(object:entity)
    RETURN properties(subject)['name'] AS subject_name, rel.relationship AS relationship, properties(object)['name'] AS object_name;
    """

    try:
        result = session.execute(query)
        if not result.is_succeeded():
            raise Exception(f"Query failed: {result.error_msg()}")

        result_list = result.as_primitive()
        triplets = []

        for item in result_list:
            # Extract values from the dictionary
            subject = item.get('subject_name', 'Unknown')
            relationship = item.get('relationship', 'Unknown')
            object = item.get('object_name', 'Unknown')
            triplets.append((subject, relationship, object))

        logging.info(f"Fetched {len(triplets)} triplets from NebulaGraph.")
        return triplets
        
    except Exception as e:
        logging.error(f"Error fetching triplets: {e}")
        return []

def process_chunk(chunk, chunk_id):
    """
    Processes a single chunk of triplets.

    Args:
        chunk (list of tuple): The chunk of triplets to process.
        chunk_id (int): The ID of the chunk.

    Returns:
        None
    """
    logging.info(f"Processing chunk {chunk_id} with {len(chunk)} triplets")
    try:
        # Set to track seen triplets
        seen_triplets = set()
        
        kg_index = initialize_kg_index()
        
        for triplet in chunk:
            if triplet in seen_triplets:
                logging.info(f"Skipping duplicate triplet: {triplet}")
                continue
            kg_index.upsert_triplet(triplet, include_embeddings=NEBULA_GRAPH_CONFIG['INCLUDE_EMBEDDINGS'])
            seen_triplets.add(triplet)
        logging.info(f"Finished processing chunk {chunk_id}")
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_id}: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    triplet_chunks = load_triplets_in_chunks()

    if not triplet_chunks:
        sys.exit(0)

    num_processes = min(multiprocessing.cpu_count(), len(triplet_chunks))
    logging.info(f"Starting multiprocessing pool with {num_processes} processes...")

    with multiprocessing.Pool(num_processes) as pool:
        try:
            pool.starmap(process_chunk, triplet_chunks)
        except KeyboardInterrupt:
            logging.info("Process interrupted.")
        except Exception as e:
            logging.error(f"An error occurred in the multiprocessing pool: {e}")
            traceback.print_exc()

    logging.info("Upserting of triplets is complete.")

    if not os.path.exists(CONFIG['PERSIST_DIR']):
        os.makedirs(CONFIG['PERSIST_DIR'])
        logging.info(f"Directory '{CONFIG['PERSIST_DIR']}' created.")
    
    kg_index = initialize_kg_index()

    logging.info(f"Persisting graph storage context to '{CONFIG['PERSIST_DIR']}'...")
    kg_index.storage_context.persist(persist_dir=CONFIG['PERSIST_DIR'])
    logging.info("Persisting complete.")
