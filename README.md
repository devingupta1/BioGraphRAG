# BioGraphRAG: GraphRAG for Biomedical Knowledge Graphs
This repository is introduces BioGraphRAG, a platform to integrate biomedical knowledge graphs with LLMs via GraphRAG architecture. Currently it supports BioKG knoweledge grpahs, with plans of supporting other open-source biomedical knowledge graphs in the future.


# Prerequisites
Before you begin, ensure that you have the following installed on your system:
- NebulaGraph: This is the graph database where the triplets will be stored (see https://docs.nebula-graph.io/3.4.0/2.quick-start/1.quick-start-workflow/ for instructions to install) 
- Docker Desktop: To run NebulaGraph in a container (see https://docs.docker.com/desktop/ for installation instructions).
- Git: For version control and cloning the repository.
- Conda: Recommended for managing Python environments (see https://conda.io/projects/conda/en/latest/user-guide/install/index.html for instructions to install)

# Installation
## Clone the Repository

```
git clone https://github.com/devingupta1/BioGraphRAG.git
cd BioGraphRAG
```


# Get the data

```
brew install git-lfs
git lfs install

```
## Create a Conda Environment and Install Dependencies

```
conda create -n biographrag python=3.10
conda activate biographrag
pip install -r requirements.txt
```

## Configure Environment Variables
Create a .env file in the root of the project with the following variables:
```
OPENAI_API_KEY=""
OPENAI_EMBEDDING_ENGINE=""

# Chain auth
CHAINLIT_AUTH_SECRET="secret" # default is secret

# Nebula Graph
NEBULA_USER=""
NEBULA_PASSWORD=""
NEBULA_ADDRESS=""
GRAPHD_HOST=""
GRAPHD_PORT=""
SPACE_NAME= "" # SPACE_NAME is the name of the space you want to use in Nebula Graph to store the graphdb.

# User passwords (modify these)
PASSWORD_DIYA="<PASSWORD_DIYA>" 
PASSWORD_KUNJ="<PASSWORD_KUNJ>"
PASSWORD_ROHIN="<PASSWORD_ROHIN>"
PASSWORD_SIDDHARTH="<PASSWORD_SIDDHARTH>"
PASSWORD_NIRAJ="<PASSWORD_NIRAJ>"
PASSWORD_DEVIN="<PASSWORD_DEVIN>"
```

# Set Up Nebula Graph using Docker Desktop

To setup nebulagraph using the docker desktop, follow the instructions in the following link: https://docs.nebula-graph.io/3.4.0/2.quick-start/1.quick-start-workflow/. A youtube tutorial in the official nebula graph documentation is a quick way to set this up: https://youtu.be/8zKer-4RXEg


# Running the Script
Run the setup_biokg_graphdb.py script in your directory in the terminal to upsert triplets into the Nebula Graph:
```
cd backend
python setup_biokg_graphdb.py
```

# Sample Command
Once you have executed the above command, you can verify the triples are upserted to the Nebula Graph Studio by executing the following sample commands in the console to interact with your graph data:
- Retrieve a limited number of edges from the graph:
```
MATCH ()-[e]->() RETURN e LIMIT 100;
```


# Initial Setup

```
cd scripts
python create_entity_type_mappings.py
```


# To run the frontend
```
chainlit run frontend/app.py -w
```

# To run the backend
```
cd backend
uvicorn main:app --reload --port 8001
```