import openai
import streamlit as st
from openai import OpenAI
import requests
import re
import sys

# Initialize OpenAI API key
client = os.getenv('OPENAI_API_KEY')

# Function to fetch UniProt ID from gene name
def fetch_uniprot_id(gene_name):
    url = f"https://rest.uniprot.org/uniprotkb/search?query={gene_name}&fields=accession&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data and 'results' in data and len(data['results']) > 0:
            uniprot_id = data['results'][0]['primaryAccession']
            return uniprot_id
    print(f"Error: Failed to fetch UniProt ID for {gene_name}")
    return None

# Function to fetch Ensembl gene ID
def fetch_ensembl_gene_id(gene_name):
    url = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{gene_name}?content-type=application/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            gene_id = data[0]['id']
            return gene_id
    print(f"Error: Failed to fetch Ensembl gene ID for {gene_name}")
    return None

# Function to fetch Ensembl gene information
def fetch_ensembl_gene_info(gene_id):
    url = f"https://rest.ensembl.org/lookup/id/{gene_id}?expand=1&content-type=application/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        gene_info = {
            'Gene name': data.get('display_name'),
            'Description': data.get('description'),
            'Organism': data.get('species'),
            'Biotype': data.get('biotype'),
            'Location': data.get('seq_region_name') + ":" + str(data.get('start')) + "-" + str(data.get('end')),
            'Strand': data.get('strand'),
            'Ensembl Link': f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={gene_id}"
        }
        return gene_info
    print(f"Error: Failed to fetch Ensembl gene details for {gene_id}")
    return None

# Function to fetch UniProt data
def fetch_uniprot_data(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    print(f"Error: Failed to fetch UniProt data for {uniprot_id}")
    return None

# Function to extract the "Function" section from UniProt data
def extract_function_section(uniprot_data):
    function_section = []
    capture = False
    for line in uniprot_data.split('\n'):
        if line.startswith("CC   -!- FUNCTION:"):
            capture = True
            function_section.append(line.replace("CC   -!- FUNCTION:", "").strip())
        elif capture and line.startswith("CC       "):
            function_section.append(line.replace("CC       ", "").strip())
        elif capture and not line.startswith("CC       "):
            break
    if function_section:
        return ' '.join(function_section)
    print("Error: 'Function' section not found in the UniProt data")
    return None

# Function to fetch AlphaFold link
def get_alphafold_link(gene_name):
    uniprot_id = fetch_uniprot_id(gene_name)
    if not uniprot_id:
        print(f"Error: Could not find UniProt ID for gene name {gene_name}")
        return None

    url = f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return url
    else:
        print(f"Error: Failed to fetch AlphaFold link, status code {response.status_code}")
        return None

# Function to fetch drug information from NIH RxNav API
def fetch_drug_info(drug_name):
    url = f"https://mor.nlm.nih.gov/RxNav/search?searchBy=String&searchTerm={drug_name}"
    response = requests.get(url)
    if response.status_code == 200:
        return url
    else:
        print(f"Error: Failed to fetch drug information for {drug_name}, status code {response.status_code}")
        return None

# Function to combine Ensembl and UniProt information into a paragraph
def generate_combined_paragraph(gene_name, ensembl_info, uniprot_function, uniprot_info):
    if ensembl_info and uniprot_function:
        paragraph = (
            f"{ensembl_info['Gene name']} ({ensembl_info['Organism']}) is a gene located on chromosome {ensembl_info['Location']}. "
            f"It is classified as a {ensembl_info['Biotype']} gene. The gene plays a crucial role in {ensembl_info['Description']}. "
            f"Functionally, it {uniprot_function}. For more information, you can visit the Ensembl page: {ensembl_info['Ensembl Link']}.")
        context = {
            'paragraph': paragraph,
            'uniprot_info': uniprot_info
        }
        return context
    return "Gene information not found."

# Function to extract metadata using OpenAI API
def extract_metadata(answer):
    try:
        prompt = f"""Extract the following metadata from the answer in the format:
        Genes: ...
        Diseases: ...
        Proteins: ...
        Drugs: ...

        Answer: {answer}"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts metadata."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.5
        )

        metadata = response.choices[0].message.content.strip()
        print(f"[DEBUG] Metadata extracted: {metadata}")
        return metadata

    except openai.AuthenticationError:
        print("Authentication error: please check your OpenAI API key.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to process extracted metadata
def process_metadata(metadata):
    if metadata is None:
        return {}

    genes = re.findall(r'Genes: (.*)', metadata)
    diseases = re.findall(r'Diseases: (.*)', metadata)
    proteins = re.findall(r'Proteins: (.*)', metadata)
    drugs = re.findall(r'Drugs: (.*)', metadata)

    processed_data = {
        'genes': genes[0].split(", ") if genes else [],
        'diseases': diseases[0].split(", ") if diseases else [],
        'proteins': proteins[0].split(", ") if proteins else [],
        'drugs': drugs[0].split(", ") if drugs else [],
    }
    print(f"[DEBUG] Processed metadata: {processed_data}")
    return processed_data

# Main function to get combined gene and drug information and metadata
def get_combined_info_with_metadata(answer):
    metadata = extract_metadata(answer)
    processed_metadata = process_metadata(metadata)

    gene_context = {}
    alphafold_context = {}
    for gene in processed_metadata['genes']:
        uniprot_id = fetch_uniprot_id(gene)
        if uniprot_id:
            uniprot_info = {
                'uniprot_id': uniprot_id,
                'entry_name': f"{gene}_ENTRY",
                'url': f"https://www.uniprot.org/uniprot/{uniprot_id}"
            }
            print(f"[DEBUG] UniProt info fetched: {uniprot_info}")
            ensembl_gene_id = fetch_ensembl_gene_id(gene)
            if ensembl_gene_id:
                ensembl_info = fetch_ensembl_gene_info(ensembl_gene_id)
                uniprot_data = fetch_uniprot_data(uniprot_id)
                if uniprot_data:
                    uniprot_function = extract_function_section(uniprot_data)
                    combined_context = generate_combined_paragraph(gene, ensembl_info, uniprot_function, uniprot_info)
                    gene_context[gene] = combined_context
                    alphafold_link = get_alphafold_link(gene)
                    alphafold_context[gene] = alphafold_link
                else:
                    gene_context[gene] = "UniProt data not found."
            else:
                gene_context[gene] = "Ensembl gene ID not found."
        else:
            gene_context[gene] = "UniProt ID not found."

    drug_context = {}
    for drug in processed_metadata['drugs']:
        drug_info = fetch_drug_info(drug)
        drug_context[drug] = drug_info

    return gene_context, alphafold_context, drug_context

# Function to generate enriched answer using OpenAI API
def enrich_answer(original_answer, gene_context, alphafold_context, drug_context):
    context_info = f"""
    Gene Context: {gene_context}
    AlphaFold Context: {alphafold_context}
    Drug Context: {drug_context}
    """

    prompt = f"""You are an assistant that combines information to enrich an answer. Given the original answer and the context information, generate an enriched structured answer.

    Original Answer: {original_answer}

    Context Information: {context_info}

    Enriched Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that enriches answers using provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    enriched_answer = response.choices[0].message.content.strip()
    return enriched_answer

def get_initial_answer(question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
def main():
    st.title("Question Answering System")
    st.write("Enter your question below:")

    question = st.text_input("Your Question:")
    if st.button("Get Answer"):
        with st.spinner("Getting the initial answer..."):
            initial_answer = get_initial_answer(question)

        st.subheader("Initial Answer:")
        st.write(initial_answer)

        with st.spinner("Enriching the answer..."):
            gene_context, alphafold_context, drug_context = get_combined_info_with_metadata(initial_answer)
            enriched_answer = enrich_answer(initial_answer, gene_context, alphafold_context, drug_context)

        st.subheader("Enriched Answer:")
        st.write(enriched_answer)

        # Display AlphaFold links
        if alphafold_context:
            st.subheader("AlphaFold Links:")
            for gene, link in alphafold_context.items():
                if link:
                    st.markdown(f"[AlphaFold structure for {gene}]({link})")
                else:
                    st.write(f"No AlphaFold structure available for {gene}")

        # Extract and display debug information if available
        debug_info = enriched_answer.split('[DEBUG]')  # Assuming the debug info is marked with [DEBUG]
        if len(debug_info) > 1:
            st.subheader("Debug Information:")
            for info in debug_info[1:]:
                st.text(info.strip())

if __name__ == "__main__":
    main()