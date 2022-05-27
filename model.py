import torch
from torch import nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from dataset import MujocoDataset

from utils import *
import config
from graphs import Graph

class EdgeNet(nn.Module):
    def __init__(self, global_features, node_features, edge_features, output_features, features=[256, 256]):
        super(EdgeNet, self).__init__()
        features = [global_features + 2 * node_features + edge_features] + features + [output_features]
        self.net = make_nn(features)
    
    def forward(self, g, ns, nr, e):
        # g: global feature
        # ns: sender node feature
        # nr: receiver node feature
        # e: edge feature
        x = torch.cat([ns, nr, e], dim=-1)
        return self.net(x.float())


class NodeNet(nn.Module):
    def __init__(self, global_features, node_features, edge_features, output_features, features=[256, 256]):
        super(NodeNet, self).__init__()
        features = [global_features + node_features + edge_features] + features + [output_features]
        self.net = make_nn(features)
    
    def forward(self, g, n, e):
        # g: global feature
        # n: node feature
        # e: aggregated edge feature
        x = torch.cat([n, e], dim=-1)
        return self.net(x.float())


class GlobalNet(nn.Module):
    def __init__(self, global_features, node_features, edge_features, output_features, features=[256, 256]):
        super(GlobalNet, self).__init__()
        features = [global_features + node_features + edge_features] + features + [output_features]
        self.net = make_nn(features)
    
    def forward(self, g, n, e):
        # g: global feature
        # n: aggregated node feature
        # e: aggregated edge feature
        x = torch.cat([n, e], dim=-1)
        return self.net(x)


class GGRU(nn.Module):
    def __init__(self, global_features, node_features, edge_features, hidden_size=20, num_layers=2):
        super(GGRU, self).__init__()
        self.has_global = global_features > 0
        self.fg = nn.GRU(global_features, hidden_size, num_layers)
        self.fn = nn.GRU(node_features, hidden_size, num_layers)
        self.fe = nn.GRU(edge_features, hidden_size, num_layers)
    
    def forward(self, graphs, graph_h):
        global_attrs = torch.stack([graph.global_attrs for graph in graphs])
        node_attrs = torch.stack([graph.global_attrs for graph in graphs])
        edge_attrs = torch.stack([graph.global_attrs for graph in graphs])
        g_out, g_h = None, None
        if self.has_global:
            g_out, g_h = self.fg(global_attrs, graph_h.global_attrs)
        n_out, n_h = self.fn(node_attrs, graph_h.node_attrs)
        e_out, e_h = self.fe(edge_attrs, graph_h.edge_attrs)
        graph_out = graphs[-1].copy(g_out, n_out, e_out)
        graph_h = graph_h.copy(g_h, n_h, e_h)
        return graph_out, graph_h


class GN(nn.Module):
    def __init__(self, global_features, node_features, edge_features, global_out_features, node_out_features, edge_out_features):
        super(GN, self).__init__()
        self.has_global = global_features > 0
        self.fe = EdgeNet(global_features, node_features, edge_features, edge_out_features)
        self.fn = NodeNet(global_features, node_features, edge_out_features, node_out_features)
        self.fg = GlobalNet(global_features, node_out_features, edge_out_features, global_out_features)
    
    def forward(self, graph):
        # edge update
        ns = torch.stack([graph.node_attrs[graph.node_index[edge[0]]] for edge in graph.G.edges]).to(config.DEVICE)
        nr = torch.stack([graph.node_attrs[graph.node_index[edge[1]]] for edge in graph.G.edges]).to(config.DEVICE)
        e = graph.edge_attrs
        g = graph.global_attrs
        
        e_out = self.fe(g, ns, nr, e)
        
        # node update
        in_edges = torch.stack([torch.stack([e_out[graph.edge_index[edge]] for edge in graph.G.in_edges(node)]).sum(dim=0) for node in graph.G.nodes])
        n = graph.node_attrs
        n_out = self.fn(g, n, in_edges)

        # global update
        g_out = torch.tensor([])
        if self.has_global:
            e_sum = torch.sum(e_out, dim=0)
            n_sum = torch.sum(n_out, dim=0)
            g_out = self.fg(g, n_sum, e_sum)
        
        return graph.copy(g_out, n_out, e_out)


class ForwardModel(nn.Module):
    def __init__(self, global_features, node_features, edge_features, hidden_features=128):
        super(ForwardModel, self).__init__()
        self.gn1 = GN(global_features, node_features, edge_features, 0, hidden_features, hidden_features)
        self.gn2 = GN(global_features + 0, node_features + hidden_features, edge_features + hidden_features, global_features, node_features, edge_features)
    
    def forward(self, g_norm):
        g_norm = g_norm.to(config.DEVICE)
        g_1 = self.gn1(g_norm)
        g_concat = Graph.concat(g_norm, g_1)
        return self.gn2(g_concat).node_attrs
    
    def predict(self, g, norm_in, norm_out):
        g_norm = norm_in.normalize(g)
        delta_y_pred = self.forward(g_norm)
        delta_y_pred = norm_out.invnormalize(delta_y_pred)
        return g.node_attrs + delta_y_pred


class RecurrentForwardModel(nn.Module):
    def __init__(self, global_features, node_features, edge_features, hidden_features=128, gru_hidden_size=20, gru_layers=2):
        super(RecurrentForwardModel, self).__init__()
        self.ggru = GGRU(global_features, node_features, edge_features)
        self.gn = GN(global_features + 0, node_features + gru_hidden_size, edge_features + gru_hidden_size)
        
    def forward(self, g_norm, g_h):
        g_1, g_h = self.ggru(g_norm, g_h)
        g_concat = Graph.concat(g_norm, g_1)
        return self.gn2(g_concat).node_attrs, g_h
    

class GCNForwardModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=config.DM_HIDDEN_FEATURES):
        super(GCNForwardModel, self).__init__()
        layers = []
        features = [in_features] + hidden_features + [out_features]
        for i in range(len(features) - 1):
            if config.GNN_TYPE == 'graphconv':
                layers.append(gnn.GraphConv(features[i], features[i + 1]))
            elif config.GNN_TYPE == 'gatconv':
                layers.append(gnn.GATConv(features[i] * (config.N_HEADS if i > 0 else 1), features[i + 1], heads=config.N_HEADS if i < len(features) - 2 else 1, edge_dim=1))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, g_norm):
        torch_graph = g_norm.torch_G
        x, edge_index, edge_weight = torch_graph.node_attr.float(), torch_graph.edge_index, torch_graph.edge_attr.float()
        x, edge_index, edge_weight = x.to(config.DEVICE), edge_index.to(config.DEVICE), edge_weight.to(config.DEVICE)
        
        if config.GNN_TYPE == 'graphconv':
            kwargs = {'edge_weight': edge_weight}
        elif config.GNN_TYPE == 'gatconv':
            kwargs = {'edge_attr': edge_weight}

        for layer in self.layers[:-1]:
            if config.GNN_TYPE == 'graphconv': # conv
                x = layer(x, edge_index, **kwargs)
            elif config.GNN_TYPE == 'gatconv':
                x = layer(x, edge_index, **kwargs)
            x = F.relu(x) # relu
            x = F.dropout(x, training=self.training, p=config.DROPOUT) # dropout
        return self.layers[-1](x, edge_index, **kwargs)
    
    def predict(self, g, norm_in, norm_out):
        g_norm = norm_in.normalize(g)
        delta_y_pred = self.forward(g_norm)
        delta_y_pred = norm_out.invnormalize(delta_y_pred)
        return g.node_attrs + delta_y_pred.cpu()

if __name__ == '__main__':
    import dm_control.suite.swimmer as swimmer
    ds = MujocoDataset(swimmer.swimmer(6), n_runs=1)
    g, _, _ = ds[0]
    n_attrs = g.node_attrs.shape[-1]
    gcn = GCNForwardModel(n_attrs, n_attrs)
    yp = gcn(g)
    print(g.node_attrs.shape)
    print(yp.shape)