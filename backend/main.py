from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import os 
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import load_index_from_storage
from config import CONFIG, KG_PROPERTIES, LLM_CONFIG, TEXT_PROCESSING_CONFIG, NEBULA_GRAPH_CONFIG

from llama_index.core.storage import StorageContext
from llama_index.graph_stores.nebula import NebulaGraphStore

from utils import invoke_openai_completion, process_graph_response, prepare_final_response, get_combined_info_with_metadata, enrich_answer


load_dotenv(override=True)

# Set up LLM and service context
Settings.llm = LlamaOpenAI(model=LLM_CONFIG["COMPLETION_MODEL_NAME"], temperature=LLM_CONFIG["TEMPERATURE"], api_key=CONFIG["OPENAI_API_KEY"])

Settings.embed_model = LlamaOpenAIEmbedding(model=LLM_CONFIG["EMBED_MODEL_NAME"], api_key=CONFIG["OPENAI_API_KEY"])
Settings.node_parser = SentenceSplitter(chunk_size=TEXT_PROCESSING_CONFIG['CHUNK_SIZE'], chunk_overlap=TEXT_PROCESSING_CONFIG['CHUNK_OVERLAP'])
Settings.num_output = LLM_CONFIG["NUM_TOKENS"]
Settings.context_window = LLM_CONFIG["CONTEXT_WINDOW"]

app = FastAPI()


class Question(BaseModel):
    question: str

logging.basicConfig(level=logging.INFO)

space_name = NEBULA_GRAPH_CONFIG['SPACE_NAME']

edge_types, rel_prop_names, tags = KG_PROPERTIES['EDGE_TYPES'], KG_PROPERTIES['REL_PROP_NAMES'], KG_PROPERTIES['TAGS']

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

storage_context = StorageContext.from_defaults(persist_dir='./storage_graph', graph_store=graph_store)
kg_index = load_index_from_storage(
    storage_context=storage_context,
    max_triplets_per_chunk=TEXT_PROCESSING_CONFIG['MAX_TRIPLETS_PER_CHUNK'],
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    verbose=True
)

kg_index_query_engine = kg_index.as_query_engine(
    retriever_mode="keyword",
    verbose=True,
    response_mode="tree_summarize",
)


@app.post("/generate-answer/")	
async def generate_answer(question: Question):	
    try:	
        answer = invoke_openai_completion(user_query=question.question, system_prompt="You are an assistant clinical AI Assistant that provides accurate medical information.")
        return {"answer": answer}
    except Exception as e:	
        raise HTTPException(status_code=500, detail=str(e))

def query_graph(question, kg_index_query_engine):
    """
    Queries the knowledge graph to get an answer based on the provided question.

    Args:
        question (str): The user's question to query the knowledge graph.
        kg_index_query_engine: The query engine used to retrieve information from the graph.

    Returns:
        A response from the knowledge graph, or None if an error occurs.
    """
    try: 
        return kg_index_query_engine.query(question)
    except Exception as e:
        logging.error(f"Error querying the graph. {e}")
        return None

@app.post("/generate-answer-graph-rag/")
async def generate_answer_graph_rag(question: Question):
    """
    API endpoint to generate an answer using the knowledge graph and external enrichment.
    Args:
        question (Question): A Pydantic model containing the user's question.

    Returns:
        A JSON response containing the final enriched answer and metadata.
    """
    
    try:
        logging.info(f"Received question: {question.question}")

        # Step 1: Query the graph for the initial answer
        response_graph_rag = query_graph(question.question, kg_index_query_engine)
        if response_graph_rag is None:
            answer = invoke_openai_completion(user_query=question.question, system_prompt="You are an assistant clinical AI Assistant that provides accurate medical information.")
            final_answer = answer
            return {
                "answer": final_answer + "\n\n(This answer is not generated from Knowledge Graph)",
                "metadata": {}
            }

        # Step 2: Extract initial context and metadata from the graph response
        context, initial_answer, subgraph = process_graph_response(response_graph_rag)

        # Step 3: Enrich the answer using the pipeline
        gene_context, alphafold_context, drug_context = get_combined_info_with_metadata(initial_answer)
        enriched_answer = enrich_answer(initial_answer, gene_context, alphafold_context, drug_context)
        
        # Step 4: Prepare the final structured answer
        final_answer = prepare_final_response(
            initial_answer=initial_answer,
            enriched_answer=enriched_answer,
            context=context,
            gene_context=gene_context,
            alphafold_context=alphafold_context,
            drug_context=drug_context
        )

        return {
            "answer": final_answer,
            "metadata": {
                "context": context,
                "subgraph": subgraph
            }
        }

    except Exception as e:
        logging.error(f"Error: {e}")
        return {
            "answer": f"Sorry, an error occurred while processing your request. {e}",
            "metadata": {}
        }



