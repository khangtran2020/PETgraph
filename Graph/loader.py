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
from Graph.graph import NaiveHetGraph, ModifiedHetGraph
from Utils.fstore import FeatureStore

logging.basicConfig(level=logging.INFO)

import numpy as np
import torch
from torch_geometric.data import Data as PygData
import tqdm


@contextmanager
def timeit(logger, task):
    logger.info('Started task %s ...', task)
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info('Completed task %s - %.3f sec.', task, t1 - t0)


class NaiveHetDataLoader(object):
    logger = logging.getLogger('native-het-dl')

    def __init__(self, width: Union[int, List], depth: int,
                 g: NaiveHetGraph, ts_range: Set, batch_size: int, n_batch: int, seed_epoch: bool,
                 shuffle: bool, num_workers: int, method: str, cache_result: bool = False):

        self.g = g
        self.ts_range = ts_range

        if seed_epoch:
            batch_size = sum(batch_size)
            n_batch = int(np.ceil(len(self.seeds) / batch_size))
        else:
            assert len(batch_size) == len(self.label_seed)

        self.seed_epoch = seed_epoch
        self.batch_size = batch_size
        self.n_batch = n_batch
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.depth = depth
        self.width = width if isinstance(list, tuple) else [width] * depth
        assert len(self.width) == depth

        self.method = method
        self.cache_result = cache_result
        self.cache = None

    @property
    @lru_cache()
    def seeds(self):
        return self.g.get_seed_nodes(self.ts_range)

    @property
    @lru_cache()
    def label_seed(self) -> Dict[int, int]:
        seeds = set(self.seeds)
        label_seed = defaultdict(list)
        for sd, lbl in self.g.seed_label.items():
            if sd in seeds:
                label_seed[lbl].append(sd)
        return label_seed

    def sample_seeds(self) -> List:
        if self.seed_epoch:
            return self.g.get_seed_nodes(self.ts_range)

        rval = []
        lbl_sd = self.label_seed
        for i, bz in enumerate(self.batch_size):
            cands = lbl_sd[i]
            rval.extend(
                np.random.choice(
                    cands, bz, replace=len(cands) < bz))
        return rval

    def __iter__(self):
        import gc;
        gc.collect()
        if self.method in {'sage', 'sage-merged',
                           'dw-sage', 'dw-sage-merged',
                           'dw0-ntw1-sage', 'dw1-ntw0-sage'}:
            yield from self.iter_sage()
        else:
            raise NotImplementedError(
                'unknown sampling method %s' % self.method)

    def iter_sage(self):
        if self.cache_result and self.cache and len(self.cache) == len(self):
            self.logger.info('DL loaded from cache')
            for e in self.cache:
                yield e
        else:
            from torch.utils.data import DataLoader
            g = self.g
            seeds = self.sample_seeds()
            seeds_encoded = [g.node_encode[e] for e in seeds]
            sampler = self.get_sage_neighbor_sampler(
                seeds=seeds)
            bz = sum(self.batch_size) if not self.seed_epoch else self.batch_size
            dl = DataLoader(
                seeds_encoded, batch_size=bz, shuffle=self.shuffle)

            if self.cache_result:
                self.cache = []
            for encoded_seeds in dl:
                batch_size, encoded_node_ids, adjs = sampler.sample(encoded_seeds)
                encoded_node_ids = encoded_node_ids.cpu().numpy()
                edge_ids = self.convert_sage_adjs_to_edge_ids(adjs)
                encoded_seeds = encoded_seeds.numpy()
                if self.cache_result:
                    self.cache.append([encoded_seeds, encoded_node_ids, edge_ids])
                yield encoded_seeds, encoded_node_ids, edge_ids

    def __len__(self):
        return self.n_batch

    def convert_sage_adjs_to_edge_ids(self, adjs):
        from torch_geometric.data.sampler import Adj
        if isinstance(adjs, Adj):
            adjs = [adjs]

        if '-merged' not in self.method:
            return [a[1].cpu().numpy() for a in adjs]

        e_ids = np.concatenate([a[1].cpu().numpy() for a in adjs])
        e_ids = np.unique(e_ids)
        return [e_ids]

    @lru_cache()
    def get_node_type_weight_tensor(self):
        g = self.g
        assert isinstance(g, NaiveHetGraph)
        node_types = np.asarray([g.node_type[g.node_decode[i]] for i in range(len(g.node_type))])
        weight = dict(zip(*np.unique(node_types, return_counts=True)))
        weight = dict((k, 1.0 / v) for k, v in weight.items())
        rval = np.asarray([weight[k] for k in node_types])
        return torch.FloatTensor(rval)

    def get_sage_neighbor_sampler(self, seeds):
        from torch_geometric.data.sampler import NeighborSampler
        from Graph.sampler import DegreeWeightedNeighborSampler
        g = self.g
        g.node_type_encode
        edge_index = g.edge_list_encoded
        edge_index = torch.LongTensor(edge_index)

        node_idx = np.asarray([
            g.node_encode[e] for e in seeds
            if g.node_ts[e] in self.ts_range])

        node_idx = torch.LongTensor(node_idx)

        if self.method in {'sage', 'sage-merged'}:
            return NeighborSampler(
                sizes=self.width,
                edge_index=edge_index,
                node_idx=node_idx, num_nodes=len(g.node_type),
                batch_size=sum(self.batch_size) if not self.seed_epoch else self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle
            )
        elif self.method in {
            'dw-sage', 'dw-sage-merged',
            'dw0-ntw1-sage',
            'dw1-ntw0-sage',
        }:

            if self.method == 'dw1-ntw0-sage':
                weights = 1.0
            else:
                weights = self.get_node_type_weight_tensor()

            if self.method == 'dw0-ntw1-sage':
                enable_degree_weight = False
            else:
                enable_degree_weight = True
            return DegreeWeightedNeighborSampler(
                node_type_weights=weights,
                enable_degree_weight=enable_degree_weight,
                sizes=self.width,
                edge_index=edge_index,
                node_idx=node_idx, num_nodes=len(g.node_type),
                batch_size=sum(self.batch_size) if not self.seed_epoch else self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
            )
        raise NotImplementedError('unknown method %s' % self.method)


class ParallelHetDataLoader(object):
    logger = logging.getLogger('native-het-dl')

    def __init__(self, rank: int, world_size: int, width: Union[int, List], depth: int,
                 g: ModifiedHetGraph, ts_range: Set, batch_size, n_batch: int, seed_epoch: bool,
                 shuffle: bool, num_workers: int, method: str, cache_result: bool = False, pin_memory: bool = False,
                 seed: int = 0):

        self.g = g
        self.ts_range = ts_range
        if seed_epoch:
            batch_size = sum(batch_size)
            n_batch = int(np.ceil(len(self.seeds) / batch_size))
        else:
            assert len(batch_size) == len(self.label_seed)
        self.seed_epoch = seed_epoch
        self.batch_size = batch_size
        self.n_batch = int(n_batch / world_size)
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.depth = depth
        self.width = width if isinstance(list, tuple) else [width] * depth
        assert len(self.width) == depth
        self.method = method
        self.cache_result = cache_result
        self.cache = None
        self.rank = rank
        self.world_size = world_size
        self.pin_memory = pin_memory
        self.seed = seed
        self.feat_dict = g.feat_dict

    @property
    @lru_cache()
    def seeds(self):
        return self.g.get_seed_nodes(self.ts_range)

    @property
    @lru_cache()
    def label_seed(self) -> Dict[int, int]:
        seeds = set(self.seeds)
        label_seed = defaultdict(list)
        for sd, lbl in self.g.seed_label.items():
            if sd in seeds:
                label_seed[lbl].append(sd)
        return label_seed

    def sample_seeds(self) -> List:
        if self.seed_epoch:
            return self.g.get_seed_nodes(self.ts_range)

        rval = []
        lbl_sd = self.label_seed
        for i, bz in enumerate(self.batch_size):
            cands = lbl_sd[i]
            rval.extend(
                np.random.choice(
                    cands, bz, replace=len(cands) < bz))
        return rval

    def __iter__(self):
        import gc;
        gc.collect()
        if self.method in {'sage', 'sage-merged',
                           'dw-sage', 'dw-sage-merged',
                           'dw0-ntw1-sage', 'dw1-ntw0-sage'}:
            yield from self.iter_sage()
        else:
            raise NotImplementedError(
                'unknown sampling method %s' % self.method)

    def iter_sage(self):
        if self.cache_result and self.cache and len(self.cache) == len(self):
            self.logger.info('DL loaded from cache')
            for e in self.cache:
                yield e
        else:
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler
            g = self.g
            seeds = self.sample_seeds()
            seeds_encoded = [g.node_encode[e] for e in seeds]
            neighbor_sampler = self.get_sage_neighbor_sampler(seeds=seeds)
            sampler = DistributedSampler(dataset=seeds_encoded, num_replicas=self.world_size, rank=self.rank,
                                         shuffle=self.shuffle)
            bz = int(sum(self.batch_size) / self.world_size) if not self.seed_epoch else int(
                self.batch_size / self.world_size)
            dl = DataLoader(seeds_encoded, batch_size=bz, shuffle=self.shuffle, pin_memory=self.pin_memory,
                            sampler=sampler, num_workers=self.num_workers)

            if self.cache_result:
                self.cache = []
            for encoded_seeds in dl:
                batch_size, encoded_node_ids, adjs = neighbor_sampler.sample(encoded_seeds)
                encoded_node_ids = encoded_node_ids.cpu().numpy()
                edge_ids = self.convert_sage_adjs_to_edge_ids(adjs)
                encoded_seeds = encoded_seeds.numpy()
                # print(encoded_seeds)
                batch = (encoded_seeds, encoded_node_ids, edge_ids)
                x, y = self.prepare_batch(batch=batch)
                if self.cache_result:
                    self.cache.append([x, y])
                yield x, y

    def __len__(self):
        return self.n_batch

    def prepare_batch(self, batch):
        g = self.g
        encoded_seeds, encoded_ids, edge_ids = batch
        encoded_seeds = set(encoded_seeds)
        encode_to_new = dict((e, i) for i, e in enumerate(encoded_ids))
        mask = np.asarray([e in encoded_seeds for e in encoded_ids])
        decoded_ids = [g.node_decode[e] for e in encoded_ids]

        x = np.asarray([g.get_feat(e) for e in decoded_ids])
        x = torch.FloatTensor(x)
        edge_list = [g.edge_list_encoded[:, idx] for idx in edge_ids]
        f = lambda x: encode_to_new[x]
        f = np.vectorize(f)
        edge_list = [f(e) for e in edge_list]
        edge_list = [torch.LongTensor(e) for e in edge_list]
        y = np.asarray([
            -1 if e not in encoded_seeds else g.seed_label_encoded[e]
            for e in encoded_ids
        ])
        y = torch.LongTensor(y)
        mask = torch.BoolTensor(mask)
        y = y[mask]

        node_type_encode = g.node_type_encode
        node_type = [node_type_encode[g.node_type[e]] for e in decoded_ids]
        node_type = torch.LongTensor(np.asarray(node_type))
        edge_type = [[g.edge_list_type_encoded[eid] for eid in list_] for list_ in edge_ids]
        edge_type = [torch.LongTensor(np.asarray(e)) for e in edge_type]
        return ((mask, x, edge_list, node_type, edge_type), y)

    def convert_sage_adjs_to_edge_ids(self, adjs):
        from torch_geometric.data.sampler import Adj
        if isinstance(adjs, Adj):
            adjs = [adjs]

        if '-merged' not in self.method:
            return [a[1].cpu().numpy() for a in adjs]

        e_ids = np.concatenate([a[1].cpu().numpy() for a in adjs])
        e_ids = np.unique(e_ids)
        return [e_ids]

    @lru_cache()
    def get_node_type_weight_tensor(self):
        g = self.g
        assert isinstance(g, NaiveHetGraph)
        node_types = np.asarray([g.node_type[g.node_decode[i]] for i in range(len(g.node_type))])
        weight = dict(zip(*np.unique(node_types, return_counts=True)))
        weight = dict((k, 1.0 / v) for k, v in weight.items())
        rval = np.asarray([weight[k] for k in node_types])
        return torch.FloatTensor(rval)

    def get_sage_neighbor_sampler(self, seeds):
        from torch_geometric.data.sampler import NeighborSampler
        from Graph.sampler import DegreeWeightedNeighborSampler
        g = self.g
        g.node_type_encode
        edge_index = g.edge_list_encoded
        edge_index = torch.LongTensor(edge_index)

        node_idx = np.asarray([
            g.node_encode[e] for e in seeds
            if g.node_ts[e] in self.ts_range])

        node_idx = torch.LongTensor(node_idx)

        if self.method in {'sage', 'sage-merged'}:
            return NeighborSampler(
                sizes=self.width,
                edge_index=edge_index,
                node_idx=node_idx, num_nodes=len(g.node_type),
                batch_size=sum(self.batch_size) if not self.seed_epoch else self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle
            )
        elif self.method in {
            'dw-sage', 'dw-sage-merged',
            'dw0-ntw1-sage',
            'dw1-ntw0-sage',
        }:

            if self.method == 'dw1-ntw0-sage':
                weights = 1.0
            else:
                weights = self.get_node_type_weight_tensor()

            if self.method == 'dw0-ntw1-sage':
                enable_degree_weight = False
            else:
                enable_degree_weight = True
            return DegreeWeightedNeighborSampler(
                node_type_weights=weights,
                enable_degree_weight=enable_degree_weight,
                sizes=self.width,
                edge_index=edge_index,
                node_idx=node_idx, num_nodes=len(g.node_type),
                batch_size=sum(self.batch_size) if not self.seed_epoch else self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
            )
        raise NotImplementedError('unknown method %s' % self.method)


class DataLoader(object):
    logger = logging.getLogger('data-loader')

    @classmethod
    def create_from_graph_data(cls, name, gd, feat, num_feat, time_range,
                               n_repeat, n_batch, enable_cache,
                               seed_set, seed_epoch,
                               width, depth, batch_size, impl='eff_hgt'):
        return cls(
            name=name, g=gd,
            feat=feat, num_feat=num_feat,
            time_range=time_range,
            link_ts=gd.node_ts, link_label=gd.node_label,
            n_repeat=n_repeat, n_batch=n_batch,
            enable_cache=enable_cache,
            seed_set=seed_set, seed_epoch=seed_epoch,
            width=width, depth=depth, batch_size=batch_size, impl=impl)

    def __init__(self, name, g, feat, num_feat, time_range,
                 link_ts, link_label, n_repeat, n_batch, enable_cache,
                 seed_set, seed_epoch,
                 width=8, depth=2,
                 batch_size=[64, 16], impl='pyhgt'):

        self.batch_size = batch_size

        self.g = g
        self.feat = feat
        self.num_feat = num_feat
        self.time_range = time_range

        self.depth = depth
        self.width = width

        # ts
        self.link_ts = link_ts

        # label
        self.link_label = link_label
        self.label_link = defaultdict(set)
        for k, v in self.link_label.items():
            self.label_link[v].add(k)

        self.count = 0
        self.n_repeat = n_repeat
        self.n_batch = n_batch
        self.enable_cache = enable_cache
        self.cache = None

        self.seed_epoch = seed_epoch
        self.seed_set = seed_set

        self.name = name
        self.pbar = None
        self.impl = impl
        assert impl in ('eff_hgt', 'pyhgt')

        self.default_feat = np.zeros((self.num_feat,))

    def __len__(self):
        return self.n_repeat * self.n_batch

    def _hgt_extract(self, layer_data, graph):
        logger = self.logger

        feature = {}
        times = {}
        indxs = {}
        texts = []
        for _type in layer_data:
            idxs = np.array(list(layer_data[_type].keys()))
            tims = np.array(list(layer_data[_type].values()))[:, 1]

            values = [None] * len(idxs)
            cnt_na = 0
            for ii, i in enumerate(idxs):
                node = self.g.node_bacward[_type][i]
                val = self.feat.get(node['id'], default_value=None)
                if val is None:
                    cnt_na += 1
                    val = self.default_feat
                values[ii] = val
            if _type == 'node_link_id' and cnt_na > 0:
                logger.error(
                    'Feature extraction for node %s - cnt_na %d/%d - %f',
                    _type, cnt_na, len(idxs), cnt_na / len(idxs)
                )
                raise RuntimeError('nan in feature!')

            feature[_type] = np.vstack(values)
            assert len(idxs) == feature[_type].shape[0]

            times[_type] = tims
            indxs[_type] = idxs
        return feature, times, indxs, texts

    def sample(self):
        logger = self.logger

        assert self.time_range is not None
        logger.info('sample seed-epoch %s', self.seed_epoch)
        if self.seed_epoch:
            candidates = list(e for e in self.seed_set
                              if self.time_range.get(self.link_ts[e], False))
            batch_size = int(np.ceil(len(candidates) * 1.0 / self.n_batch))
            np.random.shuffle(candidates)
            target_ids_chunk = [
                candidates[i * batch_size:(i + 1) * batch_size]
                for i in range(int(np.ceil(len(candidates) * 1.0 / batch_size)))]
            logger.info('batch-size %d, #batch %d.', batch_size, len(target_ids_chunk))
        else:
            candidates0 = list(
                e for e in self.label_link[0]
                if self.time_range.get(self.link_ts[e], False)
                and e in self.seed_set
            )
            candidates1 = list(
                e for e in self.label_link[1]
                if self.time_range.get(self.link_ts[e], False)
                and e in self.seed_set
            )

            size0, size1 = self.batch_size
            size0 = min([size0, len(candidates0)])
            size1 = min([size1, len(candidates1)])

            target_ids_chunk = []
            times = self.n_batch

            for i in range(times):
                target0_ids = np.random.choice(candidates0, size0, replace=False)
                target1_ids = np.random.choice(candidates1, size1, replace=False)
                target_ids_chunk.append(np.concatenate([target0_ids, target1_ids]))

        cache = []
        for target_ids in tqdm.tqdm(target_ids_chunk,
                                    desc=f'{self.name}-sample'):
            method = getattr(self, 'sample_impl_' + self.impl)
            cache.append(method(target_ids))
        return cache

    def __iter__(self):

        self.count %= self.n_batch * self.n_repeat
        if self.count % len(self) == 0:
            self.pbar = tqdm.tqdm(desc=self.name, total=len(self))
            if not self.enable_cache:
                self.cache = None

        if self.cache is None:
            self.cache = self.sample()
        yield self.cache[self.count % self.n_batch]

        self.count += 1
        self.pbar.update(1)
