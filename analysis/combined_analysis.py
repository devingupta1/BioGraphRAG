import logging
import time
import pandas as pd
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
from functools import lru_cache
from utils import *  # Assuming necessary utility functions are imported
from config import *  # Assuming necessary configurations are imported
from main import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your actual CSV file that contains node data
df = pd.read_csv('/Users/kunjrathod/MedRAG/scripts/NebulaGraph Studio result.csv')

# Define degree ranges as instance variables
LOW_DEGREE_RANGE = (0, 100)
MID_DEGREE_RANGE = (101, 699)
HIGH_DEGREE_RANGE = (700, 2000)

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
    # Use ProcessPoolExecutor for mid and high degree ranges
    with ProcessPoolExecutor(max_workers=10) as executor:  # Increased from 5 to 10
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

def calculate_statistics(times):
    stats = {}
    stats['min'] = min(times)
    stats['max'] = max(times)
    stats['mean'] = statistics.mean(times)
    stats['median'] = statistics.median(times)
    stats['std_dev'] = statistics.stdev(times)
    return stats

def display_performance(times, degree_range):
    logging.info(f"\n{degree_range} Degree Nodes Performance")
    logging.info(f"{'Node':<20}{'Degree':<10}{'Retrieval Time (s)':<20}{'Generation Time (ms)':<20}{'Enrichment Time (ms)':<20}{'Total Time (s)':<20}")
    
    retrieval_times, generation_times, enrichment_times, total_times = [], [], [], []

    for node, degree, total_time, retrieval_time, generation_time, enrichment_time in times:
        logging.info(f"{node:<20}{degree:<10}{retrieval_time:<20.2f}{generation_time:<20.2f}{enrichment_time:<20.2f}{total_time:<20.2f}")
        retrieval_times.append(retrieval_time)
        generation_times.append(generation_time)
        enrichment_times.append(enrichment_time)
        total_times.append(total_time)

    logging.info("\nStatistics:")
    for time_type, times_list in [("Retrieval", retrieval_times), ("Generation", generation_times), ("Enrichment", enrichment_times), ("Total", total_times)]:
        if times_list:
            stats = calculate_statistics(times_list)
            logging.info(f"{time_type} Time - Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, Std Dev: {stats['std_dev']:.2f}")
        else:
            logging.info(f"{time_type} Time - No data available.")
    
    return retrieval_times, generation_times, enrichment_times, total_times

if __name__ == "__main__":
    logging.info("Starting analysis...")

    low_degree_results = analyze_query_times(df, LOW_DEGREE_RANGE)
    mid_degree_results = analyze_query_times(df, MID_DEGREE_RANGE)  # This will now use multiprocessing
    high_degree_results = analyze_query_times(df, HIGH_DEGREE_RANGE)  # This will now use multiprocessing

    display_performance(low_degree_results, "Low")
    display_performance(mid_degree_results, "Mid")
    display_performance(high_degree_results, "High")

    logging.info("Analysis complete.")
