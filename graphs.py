from re import I
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data

from utils import *

class GraphData(Data):
    def __init__(self, G, global_attrs, node_attrs, edge_attrs):
        # breakpoint()
        torch_G = from_networkx(G)
        super(GraphData, self).__init__(node_attrs, torch_G.edge_index, edge_attrs)
        self.global_attrs = global_attrs


class Graph:
    def __init__(self, G, global_attrs, node_attrs, edge_attrs):
        self.global_attrs = global_attrs
        self.node_attrs = node_attrs
        self.edge_attrs = node_attrs
        if global_attrs is None:
            self.global_attrs = torch.tensor([])
        self.G = G
        self.edge_name = {i: edge for i, edge in enumerate(self.G.edges)}
        self.edge_index = {edge: i for i, edge in enumerate(self.G.edges)}
        self.node_name = {i: node for i, node in enumerate(self.G.nodes)}
        self.node_index = {node: i for i, node in enumerate(self.G.nodes)}

        if len(self.G.nodes) == 0 or len(self.G.edges) == 0:
            return

        self.update(global_attrs, node_attrs, edge_attrs)

    def update(self, global_attrs, node_attrs, edge_attrs):
        if global_attrs is None:
            global_attrs = self.global_attrs
        if node_attrs is None:
            node_attrs = self.node_attrs
        if edge_attrs is None:
            edge_attrs = self.edge_attrs
        self.update_attr(global_attrs, node_attrs, edge_attrs)
        self.update_stat()

    def update_attr(self, global_attrs, node_attrs, edge_attrs):
        self.global_attrs = global_attrs
        if torch.is_tensor(node_attrs) or torch.is_tensor(edge_attrs):
            self.node_attrs = node_attrs
            self.edge_attrs = edge_attrs

            if torch.is_tensor(node_attrs):
                if node_attrs is not None:
                    for i, node_attr in enumerate(node_attrs):
                        self.G.nodes[self.node_name[i]]['node_attr'] = node_attrs[i]
            if torch.is_tensor(edge_attrs):
                if edge_attrs is not None:
                    for i, edge_attr in enumerate(edge_attrs):
                        self.G.edges[self.edge_name[i]]['edge_attr'] = edge_attrs[i]
            self.torch_G = from_networkx(self.G)
            self.torch_G.node_attr = torch.stack(self.torch_G.node_attr)
        else:
            self.torch_G = from_networkx(self.G)
            self.torch_G.node_attr = torch.stack(self.torch_G.node_attr)
            self.node_attrs = self.torch_G.node_attr
            self.edge_attrs = self.torch_G.edge_attr

    def update_stat(self):
        if self.node_attrs is not None:
            self.node_mean = self.node_attrs.mean(dim=0)
            self.node_std = self.node_attrs.std(dim=0)
        if self.edge_attrs is not None:
            self.edge_mean = self.edge_attrs.mean(dim=0)
            self.edge_std = self.edge_attrs.std(dim=0)

    def neighbors(self, node):
        return [x for x in self.G.neighbors(node)]

    def to(self, device):
        if self.node_attrs is not None:
            self.node_attrs = self.node_attrs.to(device)
        if self.edge_attrs is not None:
            self.edge_attrs = self.edge_attrs.to(device)
        # self.update(self.global_attrs, self.node_attrs, self.edge_attrs)
        return self

    def copy(self, global_attrs=None, node_attrs=None, edge_attrs=None):
        graph_copy = Graph.from_nx_graph(self.G.copy())
        node_attrs = self.node_attrs.clone().detach() if node_attrs is None else node_attrs
        edge_attrs = self.edge_attrs.clone().detach() if edge_attrs is None else edge_attrs
        graph_copy.update(global_attrs, node_attrs, edge_attrs)
        return graph_copy

    @staticmethod
    def from_nx_graph(G, global_attrs=None, node_attrs=None, edge_attrs=None):
        return Graph(G, global_attrs, node_attrs, edge_attrs)

    @staticmethod
    def concat(sg1, sg2):
        node_attrs = torch.cat([sg1.node_attrs, sg2.node_attrs], dim=-1)
        edge_attrs = torch.cat([sg1.edge_attrs, sg2.edge_attrs], dim=-1)
        concat_graph = Graph(sg1.G, None, node_attrs, edge_attrs)
        return concat_graph

    def concat_node(self, node_attrs):
        self.update(None, torch.cat([self.node_attrs, node_attrs], dim=-1), self.edge_attrs)

    @staticmethod
    def union(sg1, sg2):
        U = nx.union(sg1.G, sg2.G, rename=('G1-', 'G2-'))
        return Graph.from_nx_graph(U)
    
    @staticmethod
    def empty():
        return Graph.from_nx_graph(nx.DiGraph())