import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from utils import *  # Assuming necessary utility functions are imported
from config import *  # Assuming necessary configurations are imported
from main import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your actual CSV file that contains node data
df = pd.read_csv('/Users/kunjrathod/MedRAG/scripts/NebulaGraph Studio result.csv')

from functools import lru_cache

@lru_cache(maxsize=None)
def cached_query_graph(question):
    logger.debug(f"Querying graph with question: {question}")
    try:
        response = query_graph(question, kg_index_query_engine)
        logger.debug(f"Query response received")
        return response
    except Exception as e:
        logger.error(f"Error querying the graph: {e}")
        return None

def analyze_query_times(df, degree_range, num_samples=10):
    nodes = df[(df['total_degree'] >= degree_range[0]) & (df['total_degree'] <= degree_range[1])].sample(n=min(num_samples, len(df)))
    logger.debug(f"Selected {len(nodes)} nodes for degree range {degree_range}")
    
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:  # Increased from 5 to 10
        futures = [executor.submit(time_query_and_answer, row['node'], row['total_degree']) for _, row in nodes.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {degree_range} degree nodes"):
            result = future.result()
            if result:
                results.append(result)
    return results

def time_query_and_answer(node_name, degree):
    question = f"Tell me about the node {node_name}"
    
    try:
        # Measure the retrieval time
        retrieval_start = time.time()
        response_graph_rag = cached_query_graph(question)
        retrieval_time = time.time() - retrieval_start

        if response_graph_rag is None:
            logger.error(f"Error querying node {node_name}")
            return None

        # Measure the initial answer generation time
        generation_start = time.time()
        context, initial_answer, subgraph = process_graph_response(response_graph_rag)
        generation_time = (time.time() - generation_start) * 1000  # Convert to milliseconds

        # Measure the enriched answer generation time
        enrichment_start = time.time()
        gene_context, alphafold_context, drug_context = get_combined_info_with_metadata(initial_answer)
        enriched_answer = enrich_answer(initial_answer, gene_context, alphafold_context, drug_context)
        enrichment_time = (time.time() - enrichment_start) * 1000  # Convert to milliseconds

        total_time = retrieval_time + (generation_time / 1000) + (enrichment_time / 1000)  # Convert back to seconds for total time

        return node_name, degree, total_time, retrieval_time, generation_time, enrichment_time

    except Exception as e:
        logger.error(f"Error processing node {node_name}: {e}")
        return None

if __name__ == "__main__":
    # Call analyze_query_times for low-degree nodes
    low_degree_results = analyze_query_times(df, (0, 100))
    
    # Display performance metrics
