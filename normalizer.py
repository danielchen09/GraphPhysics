from utils import *

class Normalizer:
    def __init__(self, momentum=0):
        self.momentum = momentum
        self.mean = 0
        self.std = 1
        self.initialized = False

    def normalize(self, x):
        return normalize(x, self.mean, self.std)
    
    def invnormalize(self, x):
        return invnormalize(x, self.mean, self.std)

    def update(self, x):
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        m = self.momentum
        if not self.initialized:
            m = 0
            self.initialized = True
        self.mean = m * self.mean + (1 - m) * mean
        self.std = m * self.std + (1 - m) * std

class GraphNormalizer(Normalizer):
    def __init__(self, momentum=config.MOMENTUM):
        super(GraphNormalizer, self).__init__(momentum=momentum)
        self.node_normalizer = Normalizer()
        self.edge_normalizer = Normalizer()

    def normalize(self, graph):
        graph = graph.copy()
        node_norm = self.node_normalizer.normalize(graph.node_attrs)
        edge_norm = self.edge_normalizer.normalize(graph.edge_attrs)
        graph.update(None, node_norm, edge_norm)
        return graph
    
    def update(self, graph):
        self.node_normalizer.update(graph.node_attrs)
        self.edge_normalizer.update(graph.edge_attrs)

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
