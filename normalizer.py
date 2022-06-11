from turtle import update
from utils import *

class Normalizer:
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.sum = 0
        self.squared_sum = 0
        self.count = 0
        self.initialized = False

    def normalize(self, x):
        if x.is_cuda:
            self.cuda()
        else:
            self.cpu()
        if self.count == 0:
            self.update(x)
        return normalize(x, self.mean, self.std)
    
    def invnormalize(self, x):
        return invnormalize(x, self.mean, self.std)

    def update(self, x):
        self.sum += x.sum(dim=0)
        self.count += x.shape[0]
        self.mean = self.sum / self.count

        self.squared_sum += torch.sum((x - self.mean) ** 2, dim=0)
        self.std = torch.sqrt(self.squared_sum / self.count)
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cpu(self):
        return self.to('cpu')
    
    def cuda(self):
        return self.to('cuda:0')

class GraphNormalizer(Normalizer):
    def __init__(self):
        super(GraphNormalizer, self).__init__()
        self.node_normalizer = Normalizer()
        self.edge_normalizer = Normalizer()

    def normalize(self, graph, copy=True):
        if copy:
            graph = graph.copy()
        node_norm = self.node_normalizer.normalize(graph.node_attrs)
        edge_norm = self.edge_normalizer.normalize(graph.edge_attrs)
        graph.update(None, node_norm, edge_norm)
        return graph
    
    def update(self, graph):
        self.node_normalizer.update(graph.node_attrs)
        self.edge_normalizer.update(graph.edge_attrs)

    def to(self, device):
        self.node_normalizer.to(device)
        self.edge_normalizer.to(device)
        return self

if __name__ == '__main__':
    from graphs import SwimmerGraph
    from dataset import SwimmerDataset

    ds = SwimmerDataset(n_runs=1)
    sg, _ = ds[0]

    normalizer = GraphNormalizer()
    print(sg)
    normalizer.update(sg)
    sg_norm = normalizer.normalize(sg)
    print(sg_norm)
    print(sg_norm.node_attrs.mean(dim=0))
