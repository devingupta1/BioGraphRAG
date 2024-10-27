import chainlit as cl
import httpx
import os
from dotenv import load_dotenv
from typing import Optional
import json

import plotly.graph_objects as go
import networkx as nx


load_dotenv(override=True)

# Endpoint to your FastAPI backend
FASTAPI_ENDPOINT = os.getenv("FASTAPI_ENDPOINT", "http://127.0.0.1:8001/generate-answer-graph-rag/")
TIMEOUT=120 # seconds

# :TODO: In future create a better implementation for user database
# Dictionary to store user credentials loaded from the .env file
users = {
    "diya": os.getenv("PASSWORD_DIYA"),
    "kunj": os.getenv("PASSWORD_KUNJ"),
    "rohin": os.getenv("PASSWORD_ROHIN"),
    "siddharth": os.getenv("PASSWORD_SIDDHARTH"),
    "niraj": os.getenv("PASSWORD_NIRAJ"),
    "devin": os.getenv("PASSWORD_DEVIN")
}
# Color coding based on entity type
entity_types = {
    "gene": "#ff7f0e",       # Orange
    "protein": "#1f77b4",    # Blue
    "disease": "#2ca02c",    # Green
    "drug": "#d62728",       # Red
    "pathway": "#9467bd",    # Purple
    "cell": "#e377c2",       # Pink
    "genetic_disorder": "#bcbd22", # Yellow-Green
    "unknown": "#7f7f7f"     # Gray
}

with open("biokg_nebula/entity_to_entity_type.json", "r") as f:
    entity_to_entity_type_map = json.load(f)

# Get the entity type based on the mapping
def get_entity_type(entity, entity_to_entity_type_map):
    return entity_to_entity_type_map.get(entity, 'unknown')


# Create graph using NetworkX
def create_networkx_graph(triples):
    G = nx.DiGraph()
    for triple in triples:
        source, relationship, target = triple
        G.add_edge(source, target, label=relationship)
    return G

# Plot biomedical graph using Plotly
def plot_biomedical_graph(G, entity_to_entity_type_map):
    pos = nx.spring_layout(G)
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
                            marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=[], size=10,
                                        colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'), line_width=2))

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])
        
        entity_type = get_entity_type(node, entity_to_entity_type_map)
        node_trace['marker']['color'] += tuple([entity_types.get(entity_type, entity_types['unknown'])])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig
# Add password authentication callback
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if users.get(username) == password:
        return cl.User(identifier=username, metadata={"role": "user", "provider": "credentials"})
    else:
        return None

@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    if app_user:
        await cl.Message(f"Hello {app_user.identifier}").send()
    else:
        await cl.Message("Please login to access the chatbot").send()

@cl.on_message
async def on_message(message: cl.Message):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(FASTAPI_ENDPOINT, json={"question": message.content}, timeout=TIMEOUT)
            response_json = response.json()
            answer = response_json.get("answer")
            metadata = response_json.get("metadata", {})
            
            if metadata:
                triples = metadata.get("subgraph", [])
                if triples:
                    G = create_networkx_graph(triples)
                    fig = plot_biomedical_graph(G, entity_to_entity_type_map)
                    elements = [cl.Plotly(name="Biomedical Graph", figure=fig, display="inline")]
                    await cl.Message(content=f"{answer}\n\n**Context**", elements=elements).send()
                else:
                    await cl.Message(content=f"{answer}\n\nNo graph data available.").send()
            else:
                await cl.Message(content=answer).send()

    except httpx.HTTPStatusError as http_err:
        # Handle specific HTTP errors
        error_message = f"HTTP error occurred: {http_err.response.status_code} - {http_err.response.text}"
        await cl.Message(content=error_message).send()
    except httpx.RequestError as req_err:
        # Handle request errors such as timeouts
        error_message = f"Request error occurred: {str(req_err)}"
        await cl.Message(content=error_message).send()
    except Exception as e:
        # Handle any other exceptions
        error_message = f"An unexpected error occurred: {str(e)}"
        await cl.Message(content=error_message).send()

if __name__ == "__main__":
    cl.run(on_message, watch=True)
