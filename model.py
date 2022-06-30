from platform import node
import torch
from torch import nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from dataset import MujocoDataset
from torch_scatter import scatter_mean

from utils import *
import config
from graphs import GraphData

class EdgeNet(nn.Module):
    def __init__(self, global_features, node_features, edge_features, output_features, features=[512]):
        super(EdgeNet, self).__init__()
        features = [global_features + 2 * node_features + edge_features] + features + [output_features]
        self.net = make_nn(features)
    
    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        x = torch.cat([src, dest, edge_attr, u[batch]], dim=-1)
        return self.net(x.float())


class NodeNet(nn.Module):
    def __init__(self, global_features, node_features, edge_features, output_features, features=[512]):
        super(NodeNet, self).__init__()
        self.node_mlp_1 = make_nn([node_features + edge_features] + features)
        self.node_mlp_2 = make_nn([node_features + features[-1] + global_features] + features + [output_features])
    
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=-1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=-1)
        return self.node_mlp_2(out)


class GlobalNet(nn.Module):
    def __init__(self, global_features, node_features, edge_features, output_features, features=[512]):
        super(GlobalNet, self).__init__()
        features = [global_features + node_features] + features + [output_features]
        self.net = make_nn(features)
    
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.net(out)

class GN(nn.Module):
    def __init__(self, global_features, node_features, edge_features,
                       global_out_features, node_out_features, edge_out_features):
        super(GN, self).__init__()
        self.layers = nn.ModuleList([
            self.make_layer(global_features, node_features, edge_features, 512, 512, 512),
            self.make_layer(512, 512, 512, 512, 512, 512),
            self.make_layer(512, 512, 512, 512, 512, 512),
            self.make_layer(512, 512, 512, 512, 512, 512),
            self.make_layer(512, 512, 512, global_out_features, node_out_features, edge_out_features),
        ])

    def forward(self, x, edge_index, edge_attr, u, batch):
        for layer in self.layers[:-1]:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)
            x, edge_attr, u = F.relu(x), F.relu(edge_attr), F.relu(u)
            x, edge_attr, u = F.dropout(x, p=config.DROPOUT), F.dropout(edge_attr, p=config.DROPOUT), F.dropout(u, p=config.DROPOUT)
        return self.layers[-1](x, edge_index, edge_attr, u, batch)

    
    def make_layer(self, global_features, node_features, edge_features, 
                   global_out_features, node_out_features, edge_out_features):
        return gnn.MetaLayer(
            EdgeNet(global_features, node_features, edge_features, edge_out_features),
            NodeNet(global_features, node_features, edge_out_features, node_out_features),
            GlobalNet(global_features, node_out_features, edge_out_features, global_out_features)
        )

class ForwardModel(nn.Module):
    def __init__(self, global_features, node_features, edge_features, out_features):
        super(ForwardModel, self).__init__()
        self.gn1 = gnn.MetaLayer(
            EdgeNet(global_features, node_features, edge_features, edge_features, config.DM_HIDDEN_FEATURES),
            NodeNet(global_features, node_features, edge_features, node_features, config.DM_HIDDEN_FEATURES),
            GlobalNet(global_features, node_features, edge_features, global_features, config.DM_HIDDEN_FEATURES)
        )
        self.gn2 = gnn.MetaLayer(
            EdgeNet(global_features * 2, node_features * 2, edge_features * 2, edge_features, config.DM_HIDDEN_FEATURES),
            NodeNet(global_features * 2, node_features * 2, edge_features, node_features, config.DM_HIDDEN_FEATURES),
            GlobalNet(global_features * 2, node_features, edge_features, global_features, config.DM_HIDDEN_FEATURES)
        )
        self.out_features = out_features
    
    def forward(self, graph):
        graph.float()
        graph = graph.to(config.DEVICE)

        x, edge_attr, u = self.gn1(graph.x, graph.edge_index, graph.edge_attr, graph.global_attrs, graph.batch)
        g1 = type(graph)(graph.edge_index, u, x, edge_attr)
        g_cat = graph.concat(g1, concat_edge=True)
        return self.gn2(g_cat.x, g_cat.edge_index, g_cat.edge_attr, g_cat.global_attrs, g_cat.batch)[0][:, :self.out_features]
        
    def predict(self, g, norm_in, norm_out, copy=True, static_graph=None):
        norm_in.normalize(g)
        g_cat = g.concat(static_graph, copy=True)
        delta_y_pred = self.forward(g_cat)
        delta_y_pred = norm_out.invnormalize(delta_y_pred)
        original = norm_in.node_normalizer.invnormalize(g.node_attrs)
        return original + delta_y_pred.cpu()


class GCNForwardModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=config.DM_HIDDEN_FEATURES):
        super(GCNForwardModel, self).__init__()
        layers = []
        features = [in_features] + hidden_features + [out_features]
        for i in range(len(features) - 1):
            layers.append(gnn.GraphConv(features[i], features[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, g_norm):
        torch_graph = g_norm.torch_G
        x, edge_index, edge_weight = g_norm.node_attrs.float(), torch_graph.edge_index, g_norm.edge_attrs.float()
        x, edge_index, edge_weight = x.to(config.DEVICE), edge_index.to(config.DEVICE), edge_weight.to(config.DEVICE)

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x) # relu
            x = F.dropout(x, training=self.training, p=config.DROPOUT) # dropout
        return self.layers[-1](x, edge_index, edge_weight=edge_weight)
    
    def predict(self, g, norm_in, norm_out, copy=True, static_node_attrs=None):
        g_norm = norm_in.normalize(g, copy=copy)
        if config.USE_STATIC_ATTRS:
            g_norm.concat_node(static_node_attrs)
        delta_y_pred = self.forward(g_norm)
        delta_y_pred = norm_out.invnormalize(delta_y_pred)
        original = g.node_attrs
        if not copy:
            original = norm_in.node_normalizer.invnormalize(g.node_attrs)
        return original + delta_y_pred.cpu()

class ConstantModel(nn.Module):
    def __init__(self):
        super(ConstantModel, self).__init__()
    
    def forward(self, x):
        return x

    def predict(self, g, norm_in, norm_out, copy=True, static_graph=None):
        return g.x

if __name__ == '__main__':
    import dm_control.suite.swimmer as swimmer
    ds = MujocoDataset(swimmer.swimmer(6), n_runs=1)
    g, _, _ = ds[0]
    n_attrs = g.node_attrs.shape[-1]
    gcn = GCNForwardModel(n_attrs, n_attrs)
    yp = gcn(g)
    print(g.node_attrs.shape)
    print(yp.shape)