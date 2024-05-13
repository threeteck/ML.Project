"""
Full pipeline to run the model from url to prediction for a single URL.
"""

from time import sleep
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests
import gc
import networkx as nx
from tqdm import tqdm
from bs4 import BeautifulSoup
import json
import os
import gc
from collections import Counter
import json
import os
from torch_geometric.data import Data, InMemoryDataset
from pyvis.network import Network
from datasets import Dataset
import torchtext
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
import json
import os
from torch_geometric.data import Data
from pyvis.network import Network
from datasets import Dataset
import torchtext
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool

def request_html(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(e)
        return None
    print('not 200')
    return None

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

# Process html to create a graph from HTML with node limit
def process_html(url, html, max_nodes=1024):
    soup = BeautifulSoup(html, 'html.parser')
    graph = nx.DiGraph()
    graph.add_node(0, tag='root', attributes={}, text='', label='root')
    extract_html_data(soup, 0, graph, max_nodes)
    graph_data = nx.node_link_data(graph)
    return (graph_data, url)

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

def index_node_text(node, tag_vocab, attr_vocab):
    node['tag_id'] = tag_vocab['vocab'][node['tag']]
    node['attr_token_ids'] = attr_vocab['vocab'].lookup_indices(node['attr_tokens'])
    del node['tag']
    del node['attr_tokens']
    return node

def index_graph_text(graph, tag_vocab, attr_vocab):
    for i, node in enumerate(graph['nodes']):
        node = index_node_text(node, tag_vocab, attr_vocab)
        graph['nodes'][i] = node
    return graph

def get_pyg_data(graph):
    '''
    Convert networkx graph to PyTorch Geometric Data object.
    :param graph:
    :return:
    '''
    edge_index = []
    for edge in graph['links']:
        edge_index.append([edge['source'], edge['target']])
        edge_index.append([edge['target'], edge['source']])  # make it undirected
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(edge_index=edge_index)
    return data


def process_graph(pyg_data, graph, attr_vocab, pad_len):
    data = pyg_data
    data.num_nodes = len(graph['nodes'])
    data.tags = torch.tensor([node['tag_id'] for node in graph['nodes']], dtype=torch.long)
    ids = [torch.tensor(node['attr_token_ids'], dtype=torch.long) for node in graph['nodes']]
    ids = [F.pad(id, (0, pad_len - len(id)), 'constant', attr_vocab['pad_index']) for id in ids]
    data.attrs = torch.stack(ids)
    return data


class DataProcessor:
    def __init__(self, tag_vocab, attr_vocab, tokenizer, pad_len):
        self.tag_vocab = tag_vocab
        self.attr_vocab = attr_vocab
        self.pad_len = pad_len
        self.tokenizer = tokenizer

    def process_data(self, graph):
        graph = preprocess_graph_attr(graph, self.tokenizer)
        graph = index_graph_text(graph, self.tag_vocab, self.attr_vocab)
        pyg_data = get_pyg_data(graph)
        data = process_graph(pyg_data, graph, self.attr_vocab, self.pad_len)
        return data


class GCN(nn.Module):
    def __init__(self, hidden_channels, num_classes, tag_vocab, attr_vocab):
        super(GCN, self).__init__()
        self.debug, self.logged_once = False, False
        self.tag_vocab = tag_vocab
        self.attr_vocab = attr_vocab
        self.embedding_dim_tag = 16
        self.embedding_dim_attr = 32
        self.hidden_channels = hidden_channels
        self.gcn_hidden_channels = hidden_channels + self.embedding_dim_tag
        self.num_classes = num_classes

        self.tag_embedding = nn.Embedding(len(tag_vocab['vocab']), self.embedding_dim_tag)
        self.attr_embedding = nn.Embedding(len(attr_vocab['vocab']), self.embedding_dim_attr)
        self.rnn = nn.LSTM(self.embedding_dim_attr, self.hidden_channels, batch_first=True)
        self.conv1 = GCNConv(self.gcn_hidden_channels, self.gcn_hidden_channels)
        self.conv2 = GCNConv(self.gcn_hidden_channels, self.gcn_hidden_channels)
        self.fc = nn.Linear(self.gcn_hidden_channels, self.num_classes)

    def log_shape(self, name, x):
        if self.debug and not self.logged_once:
            print(f'{name} shape: {x.shape}')

    def forward(self, tags, attrs, edge_index, batch):
        # tags: [num nodes]
        # attrs: [num nodes, seq len (128)]
        # edge_index: [2, num edges]

        # embed tags and attributes
        tag_embeds = self.tag_embedding(tags)  # [num nodes, embedding dim]
        attr_embeds = self.attr_embedding(attrs)  # [num nodes, seq len, embedding dim]

        self.log_shape('tag_embeds', tag_embeds)
        self.log_shape('attr_embeds', attr_embeds)

        # rnn
        _, hidden = self.rnn(attr_embeds)  # [1, num nodes, hidden channels]
        hidden = hidden[0].squeeze(0)  # [num nodes, hidden channels]
        self.log_shape('hidden2', hidden)

        # concatenate tag and attribute embeddings
        x = torch.cat([tag_embeds, hidden], dim=1)  # [num nodes, hidden channels + embedding dim]
        self.log_shape('concatenated', x)
        self.log_shape('edge_index', edge_index)

        # propagate graph convolutional neural network
        x = self.conv1(x, edge_index)  # [num nodes, hidden channels + embedding dim]
        self.log_shape('conv1', x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)  # [num nodes, hidden channels + embedding dim]
        self.log_shape('conv2', x)
        x = F.relu(x)

        # global pooling
        x = global_mean_pool(x, batch)  # [batch size, hidden channels + embedding dim]
        self.log_shape('global_pooling', x)

        # fully connected layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.fc(x)  # [batch size, num classes]
        self.log_shape('fc', x)

        if self.debug:
            self.logged_once = True

        return x


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


# return the prediction and the confidence
def predict(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.tags, data.attrs, data.edge_index, data.batch)
        out_softmax = F.softmax(out, dim=1)
        pred = out_softmax.argmax(dim=1).item()
        conf = out_softmax.max().item()
    return pred, conf


def build_model(tag_vocab, attr_vocab, model_path):
    model = GCN(hidden_channels=32, num_classes=2, tag_vocab=tag_vocab, attr_vocab=attr_vocab)
    model = load_model(model, model_path)
    return model


def run_model(url, model, processor):
    html = request_html(url)
    if html is None:
        return None, None
    graph_data, url = process_html(url, html)
    data = processor.process_data(graph_data)
    pred, conf = predict(model, data)

    return pred, conf

def main(url):
    tag_vocab_path = './data/tag_vocab.pkl'
    attr_vocab_path = './data/attr_vocab.pkl'
    model_path = '../models/model_65.pt'
    pad_len = 128

    with open(tag_vocab_path, 'rb') as f:
        tag_vocab = pickle.load(f)
    with open(attr_vocab_path, 'rb') as f:
        attr_vocab = pickle.load(f)
    tokenizer = torchtext.data.utils.get_tokenizer('spacy')
    processor = DataProcessor(tag_vocab, attr_vocab, tokenizer, pad_len)

    model = GCN(hidden_channels=32, num_classes=2, tag_vocab=tag_vocab, attr_vocab=attr_vocab)
    model = load_model(model, model_path)

    pred, conf = run_model(url, model, processor)
    return pred, conf

if __name__ == '__main__':
    # get url from args
    import sys
    url = sys.argv[1]
    pred, conf = main(url)
    print(f'Prediction: {pred}, Confidence: {conf}')

