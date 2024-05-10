import gc
import multiprocessing as mp
import pandas as pd
import networkx as nx
from tqdm import tqdm
from bs4 import BeautifulSoup
import json
import os

# Function to extract HTML data into a graph with node limit
def extract_html_data(html, parent, graph, max_nodes):
    stack = [(parent, html)]
    while stack and len(graph.nodes) < max_nodes:
        parent, element = stack.pop()
        for child in element.children:
            if len(graph.nodes) >= max_nodes:
                break
            if child.name is not None:
                tag = child.name
                attributes = child.attrs
                text = child.get_text()
                text = text[:1024]
                idx = len(graph.nodes)
                graph.add_node(idx, tag=tag, attributes=attributes, text=text, label=tag)
                graph.add_edge(parent, idx)
                stack.append((idx, child))

# Process each row to create a graph from HTML with node limit
def process_row(row, max_nodes=1024):
    url = row['url']
    label = row['label']
    html = row['html']
    soup = BeautifulSoup(html, 'html.parser')
    graph = nx.DiGraph()
    graph.add_node(0, tag='root', attributes={}, text='', label='root')
    extract_html_data(soup, 0, graph, max_nodes)
    return (url, label, graph)

# Save graph data and metadata to disk
def save_graph_data(args):
    index, (url, label, graph) = args
    graph_name = f'graph_{index}'
    json_graph = json.dumps(nx.node_link_data(graph))
    graph_path = f'./data/preprocessed_data/graphs/{graph_name}.json'
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    with open(graph_path, 'w') as f:
        f.write(json_graph)
    return {'graph_name': graph_name, 'url': url, 'label': label, 'num_nodes': len(graph.nodes)}

# Wrapper to provide `max_nodes` argument to the processing function
def process_row_with_max_nodes(args):
    row, max_nodes = args
    return process_row(row, max_nodes)

# Use multiprocessing Pool for parallel processing
def process_batch(batch, batch_index, max_nodes=1024):
    args = [(row, max_nodes) for _, row in batch.iterrows()]
    with mp.Pool(mp.cpu_count()) as pool:
        graphs = list(tqdm(pool.imap(process_row_with_max_nodes, args), total=len(batch)))
    with mp.Pool(mp.cpu_count()) as pool:
        graph_data = list(tqdm(pool.imap(save_graph_data, enumerate(graphs, start=batch_index)), total=len(graphs)))

    del graphs
    gc.collect()
    return pd.DataFrame(graph_data)

# Process the graphs in batches to handle memory limitations
def parallel_process_graphs_in_batches(batch_size=1000, max_nodes=1024):
    html_df = pd.read_parquet("./data/20000_html_data_balanced.parquet")
    total_batches = (len(html_df) + batch_size - 1) // batch_size
    all_metadata = []

    batch_start = 0
    batch_index = 0
    for batch_num in range(total_batches):
        batch_end = min(batch_start + batch_size, len(html_df))
        batch = html_df.iloc[batch_start:batch_end]
        print(f'Processing batch {batch_num + 1}/{total_batches}')
        batch_metadata = process_batch(batch, batch_index, max_nodes)
        all_metadata.append(batch_metadata)
        batch_start = batch_end
        batch_index += len(batch)

        del batch, batch_metadata
        gc.collect()

        print()

    return pd.concat(all_metadata, ignore_index=True)


if __name__ == '__main__':
    print('Number of CPUs:', mp.cpu_count())
    # Execute the processing and saving of graphs with a node limit
    graph_data_df = parallel_process_graphs_in_batches(batch_size=128, max_nodes=1024)
    # Save metadata to a Parquet file
    graph_data_df.to_parquet('./data/preprocessed_data/graph_data.parquet')
    print('Graph metadata saved to ./data/preprocessed_data/graph_data.parquet')
