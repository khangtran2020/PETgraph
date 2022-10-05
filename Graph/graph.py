import os
import sys
import time
from contextlib import contextmanager
path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
from typing import Tuple, Dict, List, Optional, Union, Set
from collections import defaultdict
import logging
from functools import lru_cache
from Utils.fstore import FeatureStore
import numpy as np
import torch
import tqdm

logging.basicConfig(level=logging.INFO)

@contextmanager
def timeit(logger, task):
    logger.info('Started task %s ...', task)
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info('Completed task %s - %.3f sec.', task, t1 - t0)

class NaiveHetGraph(object):
    logger = logging.getLogger('native-het-g')

    def __init__(self, node_type: Dict[int, str], edge_list: Tuple[int, int, str],
                 seed_label: Dict[int, int], node_ts: Dict[int, int]):
        self.logger.setLevel(logging.INFO)
        self.node_type = node_type
        self.node_type_encode = self.get_node_type_encoder(node_type)
        self.seed_label = seed_label
        self.node_ts = node_ts
        with timeit(self.logger, 'node-enc-init'):
            self.node_encode = dict((n, i) for i, n in enumerate(node_type.keys()))
            self.node_decode = dict((i, n) for n, i in self.node_encode.items())
        nenc = self.node_encode

        with timeit(self.logger, 'edge-type-init'):
            edge_types = [a[2] for a in edge_list]
            edge_encode = dict((v, i + 1) for i, v in enumerate(set(edge_types)))
            edge_encode['_self'] = 0
            edge_decode = dict((i, v) for v, i in edge_encode.items())
            self.edge_type_encode = edge_encode
            self.edge_type_decode = edge_decode
            self.edge_list_type_encoded = [edge_encode[e] for e in edge_types]

        self.edge_list_encoded = np.zeros((2, len(edge_list)))
        for i, e in enumerate(tqdm.tqdm(edge_list, desc='edge-init')):
            self.edge_list_encoded[:, i] = [nenc[e[0]], nenc[e[1]]]

        with timeit(self.logger, 'seed-label-init'):
            self.seed_label_encoded = dict((nenc[k], v) for k, v in seed_label.items())

    def get_seed_nodes(self, ts_range) -> List:
        return list([e for e in self.seed_label.keys()
                     if self.node_ts[e] in ts_range])

    def get_node_type_encoder(self, node_type: Dict[int, str]):
        types = sorted(list(set(node_type.values())))
        return dict((v, i) for i, v in enumerate(types))

    def get_sage_sampler(self, seeds, sizes=[-1], shuffle=False, batch_size=0):
        from torch_geometric.data.sampler import NeighborSampler
        g = self
        g.node_type_encode
        edge_index = g.edge_list_encoded
        edge_index = torch.LongTensor(edge_index)

        node_idx = np.asarray([g.node_encode[e] for e in seeds])
        node_idx = torch.LongTensor(node_idx)

        if batch_size <= 0:
            batch_size = len(seeds)
        return NeighborSampler(
            sizes=sizes,
            edge_index=edge_index,
            node_idx=node_idx, num_nodes=len(g.node_type),
            batch_size=batch_size,
            num_workers=0, shuffle=shuffle
        )

class ModifiedHetGraph(object):
    logger = logging.getLogger('native-het-g')

    def __init__(self, node_type: Dict[int, str], feat_dict: Dict, edge_list: Tuple[int, int, str],
                 seed_label: Dict[int, int], node_ts: Dict[int, int]):
        self.logger.setLevel(logging.INFO)
        self.node_type = node_type
        self.node_type_encode = self.get_node_type_encoder(node_type)
        self.seed_label = seed_label
        self.node_ts = node_ts
        self.feat_dict = feat_dict
        with timeit(self.logger, 'node-enc-init'):
            self.node_encode = dict((n, i) for i, n in enumerate(node_type.keys()))
            self.node_decode = dict((i, n) for n, i in self.node_encode.items())
        nenc = self.node_encode
        with timeit(self.logger, 'edge-type-init'):
            edge_types = [a[2] for a in edge_list]
            edge_encode = dict((v, i + 1) for i, v in enumerate(set(edge_types)))
            edge_encode['_self'] = 0
            edge_decode = dict((i, v) for v, i in edge_encode.items())
            self.edge_type_encode = edge_encode
            self.edge_type_decode = edge_decode
            self.edge_list_type_encoded = [edge_encode[e] for e in edge_types]
        self.edge_list_encoded = np.zeros((2, len(edge_list)))
        for i, e in enumerate(tqdm.tqdm(edge_list, desc='edge-init')):
            self.edge_list_encoded[:, i] = [nenc[e[0]], nenc[e[1]]]
        with timeit(self.logger, 'seed-label-init'):
            self.seed_label_encoded = dict((nenc[k], v) for k, v in seed_label.items())

    def get_seed_nodes(self, ts_range) -> List:
        return list([e for e in self.seed_label.keys()
                     if self.node_ts[e] in ts_range])

    def get_node_type_encoder(self, node_type: Dict[int, str]):
        types = sorted(list(set(node_type.values())))
        return dict((v, i) for i, v in enumerate(types))

    def get_sage_sampler(self, seeds, sizes=[-1], shuffle=False, batch_size=0):
        from torch_geometric.data.sampler import NeighborSampler
        g = self
        edge_index = g.edge_list_encoded
        edge_index = torch.LongTensor(edge_index)

        node_idx = np.asarray([g.node_encode[e] for e in seeds])
        node_idx = torch.LongTensor(node_idx)

        if batch_size <= 0:
            batch_size = len(seeds)
        return NeighborSampler(
            sizes=sizes,
            edge_index=edge_index,
            node_idx=node_idx, num_nodes=len(g.node_type),
            batch_size=batch_size,
            num_workers=0, shuffle=shuffle
        )

class GraphData(object):

    def __init__(self,
                 type_adj,
                 node_gtypes,
                 node_ts, node_type, graph_edge_type,
                 node_label):
        self.type_adj = type_adj
        self.node_gtypes = node_gtypes
        self.node_type = node_type
        self.node_ts = node_ts
        self.graph_edge_type = graph_edge_type
        self.node_label = node_label

    def random_choice(self, *args, **kwargs):
        return np.random.choice(*args, **kwargs)

    def update_budget(self, budget, node_id, weight, ts):
        bu = budget[node_id]
        bu[0] += weight
        bu[1] = ts

    def add_budget(self, node_id, ts, node_ts, budget, width, ts_max, node_src):
        for g_tp in self.node_gtypes[node_id]:
            adj = self.type_adj[g_tp]
            next_ids = adj[node_id]
            next_size = len(next_ids)
            if next_size > width:
                next_ids = self.random_choice(
                    next_ids, width, replace=False)
            for next_id in next_ids:
                if next_id in node_ts:
                    continue
                next_ts = self.node_ts.get(next_id, ts)
                if next_ts > ts_max:
                    continue
                self.update_budget(
                    budget, next_id, 1.0 / next_size, next_ts)
                node_src[next_id] = node_id, g_tp

    def sample(self, seeds, depth, width, ts_max):

        node_ts = {}  # node_id -> ts
        budget = defaultdict(lambda: [0., 0])  # node_id -> (weight, ts)
        node_src = {}

        # init
        for seed in seeds:
            ts = self.node_ts.get(seed, -1)
            node_ts[seed] = ts
            self.add_budget(seed, ts, node_ts, budget,
                            width=width, ts_max=ts_max,
                            node_src=node_src)

        # sample
        for _ in range(depth):
            # if not budget:

            #     raise ValueError('Budget is empty at depth %d' % _)
            sampled_nodes = list(budget.keys())
            values = np.array(list(budget.values()))
            budget.clear()
            if len(sampled_nodes) > width:
                score = values[:, 0] ** 2
                score /= np.sum(score)
                sampled_nodes = self.random_choice(
                    sampled_nodes, width, p=score, replace=False)
            for i, n in enumerate(sampled_nodes):
                node_ts[n] = values[i, 1]
            if _ + 1 < depth:
                for i, node in enumerate(sampled_nodes):
                    self.add_budget(
                        node, ts=values[i, 1], node_ts=node_ts,
                        budget=budget, width=width, ts_max=ts_max,
                        node_src=node_src)
        return node_ts, list((k, *node_src[k]) for k in node_ts.keys()
                             if k in node_src)