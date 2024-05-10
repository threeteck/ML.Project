import gc
from collections import Counter
import json
import os
from torch_geometric.data import Data, InMemoryDataset
from pyvis.network import Network
from datasets import Dataset
import torchtext
from tqdm import tqdm
import multiprocessing as mp
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_node_attr(node, tokenizer):
    '''
    Preprocess node attributes for node_link_data format. Turn all attributes to strings. Add 'text' attribute for text nodes. Tokenize attributes.
    :param node:
    :return:
    '''
    node['attr_str'] = []
    for key, value in node['attributes'].items():
        node['attr_str'].append(f'{key}= \'{value}\'')
    if node['text'] is not None and node['text'] != '' and node['tag'] != 'script':
        node['attr_str'].append(f'text= \'{node["text"]}\'')
    node['attr_str'] = ' '.join(node['attr_str'])
    # replace multiple spaces with single space
    node['attr_str'] = ' '.join(node['attr_str'].split())
    # replace multiple \n with single \n
    node['attr_str'] = '\n'.join(node['attr_str'].split('\n')).lower()
    node['attr_tokens'] = tokenizer(node['attr_str'])[:128]
    del node['attributes']
    del node['text']
    del node['attr_str']
    del node['label']
    node['tag'] = node['tag'].lower()
    return node


def preprocess_graph_attr(graph, tokenizer):
    '''
    Preprocess graph attributes for node_link_data format. Turn all attributes to strings. Tokenize attributes.
    :param graph:
    :return:
    '''
    for i, node in enumerate(graph['nodes']):
        node = preprocess_node_attr(node, tokenizer)
        graph['nodes'][i] = node
    return graph


def load_graph(path, tokenizer, verbose=False):
    with open(path, 'r') as f:
        graph = json.load(f)
        nx_graph = nx.node_link_graph(graph)
        nx_data = nx.node_link_data(nx_graph)
        nx_data = preprocess_graph_attr(nx_data, tokenizer)
        if verbose:
            print(f'Nodes in nx graph: {len(nx_graph.nodes)} | Links in nx graph: {len(nx_graph.edges)}')
    return nx_data

def save_graph(graph, path):
    with open(path, 'w') as f:
        json.dump(graph, f)

def load_single_graph(args):
    path, label, url, tokenizer, output_folder = args
    file_name = os.path.basename(path)
    if os.path.exists(os.path.join(output_folder, file_name)):
        with open(os.path.join(output_folder, file_name), 'r') as f:
            nx_data = json.load(f)
        return (nx_data, label, url, file_name)
    
    nx_data = load_graph(path, tokenizer, verbose=False)
    save_graph(nx_data, os.path.join(output_folder, file_name))
    return (nx_data, label, url, file_name)


def load_graphs(metadata, folder, tokenizer, output_folder):
    pool_args = [(os.path.join(folder, row['graph_name'] + '.json'), row['label'], row['url'], tokenizer, output_folder) for _, row in
                 metadata.iterrows()]

    with mp.Pool(mp.cpu_count()) as pool:
        nx_datas = list(tqdm(pool.imap(load_single_graph, pool_args), total=len(pool_args)))
    return nx_datas

def load_graphs_single_thread(metadata, folder, tokenizer, output_folder):
    nx_datas = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        graph_name = row['graph_name']
        url = row['url']
        label = row['label']
        path = f'./data/preprocessed_data/graphs/{graph_name}.json'
        if not os.path.exists(path):
            continue

        nx_data = load_single_graph((path, label, url, tokenizer, output_folder))
        nx_datas.append(nx_data)
    return nx_datas


def generator_node_attributes(graphs):
    '''
    Generator of tokens for node attributes for vocab building.
    :param graphs:
    :return:
    '''
    for graph, _, _, _ in graphs:
        for node in graph['nodes']:
            yield {
                'tag': [node['tag']],
                'tokens': node['attr_tokens']
            }

def build_vocab(graphs):
    '''
    Build vocabulary for node tags and attributes.
    :param graphs:
    :return:
    '''
    data = Dataset.from_generator(generator_node_attributes, gen_kwargs={'graphs': graphs})
    special_tokens = ["<unk>", "<pad>"]
    attr_vocab = torchtext.vocab.build_vocab_from_iterator(
        data['tokens'],
        min_freq=10,
        specials=special_tokens
    )
    attr_unk_index = attr_vocab["<unk>"]
    attr_pad_index = attr_vocab["<pad>"]
    attr_vocab.set_default_index(attr_unk_index)

    tag_vocab = torchtext.vocab.build_vocab_from_iterator(
        data['tag'],
        min_freq=1,
        specials=special_tokens
    )
    tag_unk_index = tag_vocab["<unk>"]
    tag_pad_index = tag_vocab["<pad>"]
    tag_vocab.set_default_index(tag_unk_index)

    attr_vocab_obj = {
        'vocab': attr_vocab,
        'unk_index': attr_unk_index,
        'pad_index': attr_pad_index
    }
    tag_vocab_obj = {
        'vocab': tag_vocab,
        'unk_index': tag_unk_index,
        'pad_index': tag_pad_index
    }

    return attr_vocab_obj, tag_vocab_obj


def preprocess_node_text(node, tag_vocab, attr_vocab):
    node['tag_id'] = tag_vocab['vocab'][node['tag']]
    node['attr_token_ids'] = attr_vocab['vocab'].lookup_indices(node['attr_tokens'])
    del node['tag']
    del node['attr_tokens']
    return node

def preprocess_graph(nx_data, tag_vocab, attr_vocab):
    graph, label, url, file_name = nx_data
    for i, node in enumerate(graph['nodes']):
        node = preprocess_node_text(node, tag_vocab, attr_vocab)
        graph['nodes'][i] = node
    return (graph, label, url, file_name)

def preprocess_single_graph(args):
    graph, tag_vocab, attr_vocab, output_folder = args
    file_name = graph[3]
    if os.path.exists(os.path.join(output_folder, file_name)):
        with open(os.path.join(output_folder, file_name), 'r') as f:
            nx_data = json.load(f)
        return nx_data
    
    nx_data = preprocess_graph(graph, tag_vocab, attr_vocab)
    save_graph(nx_data, os.path.join(output_folder, file_name))
    return nx_data
    

def preprocess_graphs(graphs, tag_vocab, attr_vocab, output_folder):
    pool_args = [(graph, tag_vocab, attr_vocab, output_folder) for graph in graphs]
    with mp.Pool(mp.cpu_count()//2) as pool:
        new_graphs = list(tqdm(pool.imap(preprocess_single_graph, pool_args), total=len(pool_args)))
    return new_graphs

def preprocess_graphs_single_thread(graphs, tag_vocab, attr_vocab, output_folder):
    new_graphs = []
    for graph in tqdm(graphs, total=len(graphs)):
        new_graph = preprocess_single_graph((graph, tag_vocab, attr_vocab, output_folder))
        new_graphs.append(new_graph)
    return new_graphs


# Function for converting a single graph
def get_pyg_data(nx_data):
    '''
    Convert networkx graph to PyTorch Geometric Data object.
    :param graph:
    :return:
    '''
    graph = nx_data[0]
    file_name = nx_data[3]
    edge_index = []
    for edge in graph['links']:
        edge_index.append([edge['source'], edge['target']])
        edge_index.append([edge['target'], edge['source']])  # make it undirected
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(edge_index=edge_index)
    return data, file_name

# Parallelized function for converting a list of graphs
def get_pyg_dataset(graphs):
    '''
    Convert list of networkx graphs to PyTorch Geometric Dataset object in parallel.
    :param graphs:
    :return:
    '''
    with mp.Pool(mp.cpu_count()) as pool:
        data_list = list(tqdm(pool.imap(get_pyg_data, graphs), total=len(graphs)))
    return data_list


class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, split, pyg_data, nx_datas, attr_vocab, pad_len=128, transform=None, pre_transform=None):
        self.pyg_data = pyg_data
        self.nx_datas = nx_datas
        self.attr_vocab = attr_vocab
        self.pad_len = pad_len
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.split}_graph_data.pt']

    def process_graph(self, args):
        pyg_data, nx_data, attr_vocab, pad_len = args
        graph, label, url, file_name = nx_data
        data, _ = pyg_data
        data.num_nodes = len(graph['nodes'])
        data.y = torch.tensor([label], dtype=torch.long)
        data.tags = torch.tensor([node['tag_id'] for node in graph['nodes']], dtype=torch.long)
        ids = [torch.tensor(node['attr_token_ids'], dtype=torch.long) for node in graph['nodes']]
        ids = [F.pad(id, (0, pad_len - len(id)), 'constant', attr_vocab['pad_index']) for id in ids]
        data.attrs = torch.stack(ids)
        return data

    def process(self):
        data_list = []
        for i in tqdm(range(len(self.pyg_data)), total=len(self.pyg_data)):
            data = self.process_graph((self.pyg_data[i], self.nx_datas[i], self.attr_vocab, self.pad_len))
            data_list.append(data)

        print(f'Saving {self.split} data to {self.processed_paths[0]}')
        self.save(data_list, self.processed_paths[0])


def build_or_load_vocab(nx_datas, build=True):
    if build:
        attr_vocab, tag_vocab = build_vocab(nx_datas)
        with open('./data/preprocessed_data/attr_vocab.pkl', 'wb') as f:
            pickle.dump(attr_vocab, f)
        with open('./data/preprocessed_data/tag_vocab.pkl', 'wb') as f:
            pickle.dump(tag_vocab, f)
        print('Vocab saved to disk.')
    else:
        with open('./data/preprocessed_data/attr_vocab.pkl', 'rb') as f:
            attr_vocab = pickle.load(f)
        with open('./data/preprocessed_data/tag_vocab.pkl', 'rb') as f:
            tag_vocab = pickle.load(f)
        print('Vocab loaded from disk.')
    return attr_vocab, tag_vocab

def get_nx_data_from_metadata(metadata):
    nx_datas = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        graph_name = row['graph_name']
        url = row['url']
        label = row['label']
        path = f'./data/preprocessed_data/graphs/{graph_name}.json'
        if not os.path.exists(path):
            continue

        nx_datas.append((None, label, url, graph_name + '.json'))
    return nx_datas

def split_data(metadata):
    metadata['label'] = metadata['label'].astype(int)
    train, test = train_test_split(metadata, test_size=0.3, stratify=metadata['label'], random_state=42)
    return train, test


def create_dataset(metadata, tokenizer, split):
    print(f'First stage preprocessing {len(metadata)} graphs...')
    output_folder = './data/preprocessed_data/processed_graphs'
    nx_datas = load_graphs_single_thread(metadata, './data/preprocessed_data/graphs', tokenizer, output_folder)
    print(f'Preprocessed {len(nx_datas)} graphs.')

    gc.collect()

    print(f'\nBuilding vocab...')
    attr_vocab, tag_vocab = build_or_load_vocab(nx_datas, build=False)
    print(f'Attr vocab size: {len(attr_vocab["vocab"])} | Tag vocab size: {len(tag_vocab["vocab"])}')

    gc.collect()

    print('\nSecond stage preprocessing...')
    output_folder = './data/preprocessed_data/processed_graphs_ids'
    nx_datas = preprocess_graphs(nx_datas, tag_vocab, attr_vocab, output_folder)
    print('Preprocessing done.')

    gc.collect()

    print('\nThird stage preprocessing...')
    pyg_data = get_pyg_dataset(nx_datas)
    print('PyG data created.')

    gc.collect()

    print('\nPreparing dataset...')
    dataset = CustomGraphDataset('./data/preprocessed_data', split, pyg_data, nx_datas, attr_vocab, pad_len=128)
    print('Dataset created.')
    return dataset, attr_vocab, tag_vocab

def main():
    metadata = pd.read_parquet('./data/preprocessed_data/graph_data_fixed.parquet')
    metadata = metadata.drop_duplicates(subset='url').reset_index(drop=True)
    metadata = metadata.dropna().reset_index(drop=True)
    tokenizer = torchtext.data.utils.get_tokenizer('spacy')

    train, test = split_data(metadata)

    print('Creating test dataset...')
    test_dataset, _, _ = create_dataset(test, tokenizer, 'test')

    print()

    print('Creating training dataset...')
    train_dataset, attr_vocab, tag_vocab = create_dataset(train, tokenizer, 'train')


def delete_duplicate_hashes(folder, metadata):
    # delete duplicate graphs and corresponding rows in metadata
    import hashlib
    import glob
    import os

    print(f'Deleting duplicate hashes in {folder}...')
    hashes = {}
    for file in tqdm(glob.glob(f'{folder}/*.json')):
        with open(file, 'r') as f:
            data = f.read()
            hash = hashlib.sha256(data.encode()).hexdigest()
            if hash not in hashes:
                hashes[hash] = [file]
            else:
                hashes[hash].append(file)

    # for each hash, that has multiple files, delete all but the first one in files
    deleted_files = []
    for hash, files in hashes.items():
        if len(files) > 1:
            for file in files[1:]:
                os.remove(file)
                file_name = os.path.basename(file).replace('.json', '')
                deleted_files.append(file_name)

    # delete corresponding rows in metadata
    metadata = metadata[~metadata['graph_name'].isin(deleted_files)]
    metadata.to_parquet('./data/preprocessed_data/graph_data_fixed.parquet')
    print('Deleted duplicate hashes.')
    return metadata


if __name__ == '__main__':
    main()