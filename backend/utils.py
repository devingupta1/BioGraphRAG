import requests
import re
import os
import logging
from config import CONFIG, LLM_CONFIG
from openai import OpenAI

openai_client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])


# Configure logging
logging.basicConfig(level=logging.INFO)

def invoke_openai_completion(user_query, system_prompt: str = "You are a helpful assistant."):
    try:
        response = openai_client.chat.completions.create(
                model=LLM_CONFIG["COMPLETION_MODEL_NAME"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=LLM_CONFIG["NUM_TOKENS"],
                temperature=LLM_CONFIG["TEMPERATURE"],
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None

def fetch_data_from_api(url, headers=None):
    """
    Fetches data from the provided API URL and returns the JSON response.
    """
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    logging.error(f"Failed to fetch data from {url}, status code {response.status_code}")
    return None

def fetch_uniprot_id(gene_name):
    """
    Fetches the UniProt ID for a given gene name.
    """
    url = f"https://rest.uniprot.org/uniprotkb/search?query={gene_name}&fields=accession&format=json"
    data = fetch_data_from_api(url)
    if data and 'results' in data and data['results']:
        return data['results'][0]['primaryAccession']
    logging.error(f"UniProt ID not found for {gene_name}")
    return None

def fetch_ensembl_gene_id(gene_name):
    """
    Fetches the Ensembl gene ID for a given gene name.
    """
    url = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{gene_name}?content-type=application/json"
    data = fetch_data_from_api(url)
    if data:
        return data[0]['id']
    logging.error(f"Ensembl gene ID not found for {gene_name}")
    return None

def fetch_ensembl_gene_info(gene_id):
    """
    Fetches detailed gene information from Ensembl using the gene ID.
    """
    url = f"https://rest.ensembl.org/lookup/id/{gene_id}?expand=1&content-type=application/json"
    data = fetch_data_from_api(url)
    if data:
        return {
            'Gene name': data.get('display_name'),
            'Description': data.get('description'),
            'Organism': data.get('species'),
            'Biotype': data.get('biotype'),
            'Location': f"{data.get('seq_region_name')}:{data.get('start')}-{data.get('end')}",
            'Strand': data.get('strand'),
            'Ensembl Link': f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={gene_id}"
        }
    logging.error(f"Ensembl gene details not found for {gene_id}")
    return {}

def fetch_and_extract_uniprot_function(uniprot_id):
    """
    Fetches UniProt data for a given UniProt ID and extracts the "Function" section.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
    response = requests.get(url)
    if response.status_code == 200:
        function_section = []
        capture = False
        for line in response.text.split('\n'):
            if line.startswith("CC   -!- FUNCTION:"):
                capture = True
                function_section.append(line.replace("CC   -!- FUNCTION:", "").strip())
            elif capture and line.startswith("CC       "):
                function_section.append(line.replace("CC       ", "").strip())
            elif capture:
                break
        if function_section:
            return ' '.join(function_section)
        logging.error("'Function' section not found in the UniProt data")
        return None
    logging.error(f"Failed to fetch UniProt data for {uniprot_id}, status code {response.status_code}")
    return None

def get_alphafold_link(gene_name):
    """
    Generates the AlphaFold link for a given gene name by first fetching its UniProt ID.
    """
    uniprot_id = fetch_uniprot_id(gene_name)
    if uniprot_id:
        url = f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return url
        logging.error(f"Failed to fetch AlphaFold link, status code {response.status_code}")
    return None

def fetch_drug_info(drug_name):
    """
    Fetches drug information from the NIH RxNav API for a given drug name.
    """
    url = f"https://mor.nlm.nih.gov/RxNav/search?searchBy=String&searchTerm={drug_name}"
    response = requests.get(url)
    if response.status_code == 200:
        return url
    logging.error(f"Failed to fetch drug information for {drug_name}, status code {response.status_code}")
    return None

def generate_combined_paragraph(gene_name, ensembl_info, uniprot_function):
    """
    Combines Ensembl and UniProt information into a structured paragraph.
    """
    if ensembl_info and uniprot_function:
        return {
            'paragraph': (
                f"{ensembl_info['Gene name']} ({ensembl_info['Organism']}) is a gene located on chromosome {ensembl_info['Location']}. "
                f"It is classified as a {ensembl_info['Biotype']} gene. The gene plays a crucial role in {ensembl_info['Description']}. "
                f"Functionally, it {uniprot_function}. For more information, you can visit the Ensembl page: {ensembl_info['Ensembl Link']}."
            )
        }
    return "Gene information not found."

def extract_metadata(answer):
    """
    Extracts metadata such as genes, diseases, proteins, and drugs from the provided answer using OpenAI.
    """
    prompt = f"""Extract the following metadata from the answer in the format:
    Genes: ...
    Diseases: ...
    Proteins: ...
    Drugs: ...

    Answer: {answer}"""

    metadata = invoke_openai_completion(user_query=prompt, system_prompt="You are an assistant that extracts metadata.")

    if metadata:
        logging.debug(f"Metadata extracted: {metadata}")
    return metadata

def process_metadata(metadata):
    """
    Processes the extracted metadata into a structured dictionary format.
    """
    if metadata:
        return {
            'genes': re.findall(r'Genes: (.*)', metadata)[0].split(", ") if 'Genes: ' in metadata else [],
            'diseases': re.findall(r'Diseases: (.*)', metadata)[0].split(", ") if 'Diseases: ' in metadata else [],
            'proteins': re.findall(r'Proteins: (.*)', metadata)[0].split(", ") if 'Proteins: ' in metadata else [],
            'drugs': re.findall(r'Drugs: (.*)', metadata)[0].split(", ") if 'Drugs: ' in metadata else [],
        }
    return {}
def get_combined_info_with_metadata(answer):
    """
    Combines gene and drug information with metadata extracted from the answer.
    """
    metadata = extract_metadata(answer)
    processed_metadata = process_metadata(metadata)

    gene_context = {}
    alphafold_context = {}
    for gene in processed_metadata.get('genes', []):
        uniprot_id = fetch_uniprot_id(gene)
        if uniprot_id:
            ensembl_gene_id = fetch_ensembl_gene_id(gene)
            if ensembl_gene_id:
                ensembl_info = fetch_ensembl_gene_info(ensembl_gene_id)
                uniprot_function = fetch_and_extract_uniprot_function(uniprot_id)
                if ensembl_info and uniprot_function:
                    gene_context[gene] = generate_combined_paragraph(gene, ensembl_info, uniprot_function)
                    alphafold_context[gene] = get_alphafold_link(gene)

    drug_context = {drug: fetch_drug_info(drug) for drug in processed_metadata.get('drugs', [])}

    return gene_context, alphafold_context, drug_context

def enrich_answer(original_answer, gene_context, alphafold_context, drug_context):
    """
    Generates an enriched answer using the original answer and additional context information.
    """
    context_info = f"""
    Gene Context: {gene_context}
    AlphaFold Context: {alphafold_context}
    Drug Context: {drug_context}
    """

    prompt = f"""You are an assistant that combines information to enrich an answer. Given the original answer and the context information, generate an enriched structured answer.

    Original Answer: {original_answer}

    Context Information: {context_info}

    Enriched Answer:"""

    return invoke_openai_completion(user_query=prompt, system_prompt="You are an assistant that enriches answers using provided context.")

def process_graph_response(response_graph_rag):
    """
    Processes the response from the graph and extracts relevant context and subgraph information.

    Args:
        response_graph_rag: The raw response from the knowledge graph.

    Returns:
        A tuple containing:
            - context (list): The context extracted from the graph response.
            - initial_answer (str): The initial answer from the graph.
            - subgraph (list): A list of triples representing the subgraph.
    """
    try:
        context = response_graph_rag.source_nodes[-1].node.metadata.get('kg_rel_texts', [])
        initial_answer = response_graph_rag.response
        subgraph = extract_triples_as_tuples_from_string(context)
        logging.debug(f"Processed graph response with {len(subgraph)} triples.")
        return context, initial_answer, subgraph
    except Exception as e:
        logging.error(f"Error processing graph response: {e}")
        raise

import re


def extract_triples_as_tuples_from_string(context):
    # Regular expression to capture entities and relationships, accounting for periods and special characters
    pattern = re.compile(
        r"(?P<e1>[^{} ]+){name: [^}]+} (?P<dir1><-|->)"
        r"\[relationship:{relationship: (?P<rel1>[^}]+)}\](?P<dir2>-|->) "
        r"(?P<e2>[^{} ]+){name: [^}]+}"
        r"(?: (?P<dir3><-|->)"
        r"\[relationship:{relationship: (?P<rel2>[^}]+)}\](?P<dir4>-|->) "
        r"(?P<e3>[^{} ]+){name: [^}]+})?"
    )
    triples = []
    for path in context:
        match = pattern.search(path)
        if match:
            # Handle first triple
            e1 = match.group('e1')
            dir1 = match.group('dir1')
            rel1 = match.group('rel1')
            dir2 = match.group('dir2')
            e2 = match.group('e2')

            triple1 = get_triple(e1, dir1, rel1, dir2, e2)
            if triple1:
                triples.append(triple1)

            # Handle second triple (if exists)
            if match.group('rel2') and match.group('e3'):
                e2 = match.group('e2')
                dir3 = match.group('dir3')
                rel2 = match.group('rel2')
                dir4 = match.group('dir4')
                e3 = match.group('e3')

                triple2 = get_triple(e2, dir3, rel2, dir4, e3)
                if triple2:
                    triples.append(triple2)
        else:
            logging.warning(f"No match found for path: {path}")
    return triples

def get_triple(e1, dir1, rel, dir2, e2):
    if dir1 == '-' and dir2 == '->':
        # e1 -[rel]-> e2
        return (e1, rel, e2)
    elif dir1 == '<-' and dir2 == '-':
        # e1 <-[rel]- e2
        return (e2, rel, e1)
    elif dir1 == '-' and dir2 == '-':
        # e1 -[rel]- e2 (undirected)
        return (e1, rel, e2)
    else:
        # Handle unexpected direction combinations
        logging.warning(f"Unexpected direction combination: {dir1}[rel]{dir2} between {e1} and {e2}")
        return None

def prepare_final_response(initial_answer, enriched_answer, context, gene_context, alphafold_context, drug_context):
    """
    Prepares the final answer by combining the initial graph response and enriched context.

    Args:
        initial_answer (str): The original answer from the knowledge graph.
        enriched_answer (str): The answer enriched with external data.
        context (list): The context extracted from the graph response.
        gene_context (dict): Gene information used for enrichment.
        alphafold_context (dict): AlphaFold protein structure links.
        drug_context (dict): Drug information for enrichment.

    Returns:
        A string containing the fully enriched and structured answer.
    """
    final_answer_parts = [
        f"**Answer from Knowledge Graph:**\n{initial_answer}"
    ]

    # If the enriched answer adds new information, include it
    if enriched_answer != initial_answer:
        final_answer_parts.append(f"\n\n**Enriched Answer->**\n{enriched_answer}")

    # Add the context
    if context:
        final_answer_parts.append("\n\n**Context**:" + "".join([f"\n- {x}" for x in context]))

    # Include gene, protein, and drug information
    additional_info_added = False

    # Gene information
    if gene_context:
        gene_details = [f"\n- {gene}: {info['paragraph']}" for gene, info in gene_context.items()]
        final_answer_parts.append("\n\n**Gene Information**:" + "".join(gene_details))
        additional_info_added = True

    # AlphaFold protein structure links
    if alphafold_context:
        protein_links = [f"\n- {gene}: [View Protein Structure]({link})" for gene, link in alphafold_context.items() if link]
        final_answer_parts.append("\n\n**AlphaFold Protein Structures**:" + "".join(protein_links))
        additional_info_added = True

    # Drug information (skip if URL is invalid or "Not mentioned")
    if drug_context:
        drug_links = []
        for drug, link in drug_context.items():
            if link and "Not mentioned" not in link and link.strip() != "":  # Check the condition
                drug_links.append(f"\n- {drug}: [More Info]({link})")
        
        if drug_links:
            final_answer_parts.append("\n\n**Drug Information**:" + "".join(drug_links))
            additional_info_added = True

    if not additional_info_added:
        final_answer_parts.append("\n\nNo additional information was found.")

    return "".join(final_answer_parts)

def prepare_final_response(initial_answer, enriched_answer, context, gene_context, alphafold_context, drug_context):
    """
    Prepares the final answer by combining the initial graph response and enriched context.

    Args:
        initial_answer (str): The original answer from the knowledge graph.
        enriched_answer (str): The answer enriched with external data.
        context (list): The context extracted from the graph response.
        gene_context (dict): Gene information used for enrichment.
        alphafold_context (dict): AlphaFold protein structure links.
        drug_context (dict): Drug information for enrichment.

    Returns:
        A string containing the fully enriched and structured answer.
    """
    final_answer_parts = [
        f"**Answer from Knowledge Graph:**\n{initial_answer}"
    ]

    # If the enriched answer adds new information, include it
    if enriched_answer != initial_answer:
        final_answer_parts.append(f"\n\n**Enriched Answer->**\n{enriched_answer}")

    # Add the context
    if context:
        final_answer_parts.append("\n\n**Context**:" + "".join([f"\n- {x}" for x in context]))

    # Include gene, protein, and drug information
    additional_info_added = False

    # Gene information
    if gene_context:
        gene_details = [f"\n- {gene}: {info['paragraph']}" for gene, info in gene_context.items()]
        final_answer_parts.append("\n\n**Gene Information**:" + "".join(gene_details))
        additional_info_added = True

    # AlphaFold protein structure links
    if alphafold_context:
        protein_links = [f"\n- {gene}: [View Protein Structure]({link})" for gene, link in alphafold_context.items() if link]
        final_answer_parts.append("\n\n**AlphaFold Protein Structures**:" + "".join(protein_links))
        additional_info_added = True

    # Drug information (skip if URL is invalid or "Not mentioned")
    if drug_context:
        drug_links = []
        for drug, link in drug_context.items():
            if link and "Not mentioned" not in link and link.strip() != "":  # Check the condition
                drug_links.append(f"\n- {drug}: [More Info]({link})")
        
        if drug_links:
            final_answer_parts.append("\n\n**Drug Information**:" + "".join(drug_links))
            additional_info_added = True

    if not additional_info_added:
        final_answer_parts.append("\n\nNo additional information was found.")

    return "".join(final_answer_parts)