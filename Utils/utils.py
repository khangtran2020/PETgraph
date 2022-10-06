import os
import sys
path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
import time
import logging
import numpy as np
from contextlib import contextmanager
from Graph.graph import GraphData, NaiveHetGraph, ModifiedHetGraph
from collections import defaultdict
from ignite.utils import convert_tensor
import torch
from tqdm import tqdm
import pandas as pd

@contextmanager
def timeit(logger, task):
    logger.info('Started task %s ...', task)
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info('Completed task %s - %.3f sec.', task, t1-t0)

def feature_mock(layer_data, graph, feature_dim=16):
    feature = {}
    times = {}
    indxs = {}
    texts = []
    for ntype, data in layer_data.items():
        if not data:
            continue
        idx = np.array(list(data.keys()))
        idx_data = np.array(list(data.values()))
        ts = idx_data[:, 1]

        feature[ntype] = np.zeros((len(idx), feature_dim))

        times[ntype] = ts
        indxs[ntype] = idx
    return feature, times, indxs, texts


def create_naive_het_graph_from_edges(df):
    logger = logging.getLogger('factory-naive-het-graph')
    logger.setLevel(logging.INFO)

    with timeit(logger, 'node-type-init'):
        view = df[['MessageId', 'Days']].drop_duplicates()
        node_ts = dict((k, v) for k, v in view.itertuples(index=False))
        df['src_type'] = 0
        view = df[['MessageId', 'src_type']].drop_duplicates()
        node_type = dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        )
        df['sender_bank_type'] = 1
        view = df[['Sender', 'sender_bank_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
        view = df[['Receiver', 'sender_bank_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
        df['account_type'] = 2
        view = df[['OrderingAccount', 'account_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
        view = df[['BeneficiaryAccount', 'account_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
        df['address_type'] = 3
        view = df[['OrderingOriginAdd', 'address_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
        view = df[['BeneficiaryOriginAdd', 'address_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))

    # if 'sender_type' not in df:
    df['sender_type'] = 'sender'
    df['receiver_type'] = 'receiver'
    df['ordering_type'] = 'ordering'
    df['beneficiary_type'] = 'beneficiary'
    df['ordering_add_type'] = 'order_add'
    df['ben_add_type'] = 'ben_add'

    with timeit(logger, 'edge-list-init'):
        edge_list = list(df[['MessageId', 'Sender', 'sender_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
        edge_list += list(df[['MessageId', 'Receiver', 'receiver_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
        edge_list += list(df[['MessageId', 'OrderingAccount', 'ordering_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
        edge_list += list(df[['MessageId', 'BeneficiaryAccount', 'beneficiary_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
        edge_list += list(df[['MessageId', 'OrderingOriginAdd', 'ordering_add_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
        edge_list += list(df[['MessageId', 'BeneficiaryOriginAdd', 'ben_add_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]

    select = df['seed'] > 0
    view = df[select][['MessageId', 'Label']].drop_duplicates()
    seed_label = dict((k, v) for k, v in view.itertuples(index=False))

    return NaiveHetGraph(node_type, edge_list,
                         seed_label=seed_label, node_ts=node_ts)


def create_graph_data_from_edges(df):
    node_link_ts = df[['src', 'ts']].drop_duplicates()
    node_ts = dict(
        (node, ts)
        for node, ts in node_link_ts.itertuples(index=False)
    )

    view = df[['src', 'src_type']].drop_duplicates()
    node_type = dict(
        (node, tp)
        for node, tp in view.itertuples(index=False)
    )
    view = df[['dst', 'dst_type']].drop_duplicates()
    node_type.update(dict(
        (node, tp)
        for node, tp in view.itertuples(index=False)
    ))

    view = df[['src', 'src_label']].drop_duplicates()
    node_label = dict(
        (node, lbl)
        for node, lbl in view.itertuples(index=False)
    )

    if 'graph_edge_type' not in df:
        df['graph_edge_type'] = 'default'

    type_adj = {}
    node_gtypes = defaultdict(set)
    graph_edge_type = {}
    for (stype, etype, dtype), gdf in df.groupby(
            ['src_type', 'graph_edge_type', 'dst_type']):
        gtype = stype, etype, dtype
        adj = defaultdict(set)
        for u, v in gdf[['src', 'dst']].itertuples(index=False):
            node_gtypes[u].add(gtype)
            node_gtypes[v].add(gtype)
            adj[u].add(v)
            adj[v].add(u)
        type_adj[gtype] = dict((k, tuple(v)) for k, v in adj.items())
        graph_edge_type[gtype] = etype

    rval = GraphData(
        type_adj=type_adj,
        node_gtypes=node_gtypes,
        node_ts=node_ts, node_type=node_type,
        graph_edge_type=graph_edge_type,
        node_label=node_label)
    return rval


def create_naive_het_homo_graph_from_edges(df):
    logger = logging.getLogger('factory-naive-het-homo-graph')
    logger.setLevel(logging.INFO)

    with timeit(logger, 'node-type-init'):
        view = df[['src', 'src_ts']].drop_duplicates()
        node_ts = dict((node, ts)
                       for node, ts in view.itertuples(index=False)
                       )
        view = df[['src']].drop_duplicates()
        node_type = dict(
            (node[0], 'node_link_id')
            for node in view.itertuples(index=False)
        )

    with timeit(logger, 'node-seed-init'):
        select = df['src_seed'] > 0
        view = df[select][['src', 'src_label']].drop_duplicates()
        seed_label = dict((k, v) for k, v in view.itertuples(index=False))

    with timeit(logger, 'edge-list-init'):
        # edge_list = []
        # df_tmp = df[['src', 'dst']].drop_duplicates()
        # for i, row in tqdm.tqdm(df_tmp.iterrows(),
        #                         total=df_tmp.shape[0],
        #                         desc='iter-edges'):
        #     edge_list.append(tuple(row.tolist()) + ('default',))

        view = df[['src', 'dst']].drop_duplicates()
        view['graph_edge_type'] = 'default'

        edge_list = view.to_numpy().tolist()

    return NaiveHetGraph(node_type, edge_list,
                         seed_label=seed_label, node_ts=node_ts)


def prepare_batch(batch, ts_range, fstore, default_feature,
                  g: NaiveHetGraph,
                  device, non_blocking=False):
    encoded_seeds, encoded_ids, edge_ids = batch
    encoded_seeds = set(encoded_seeds)
    encode_to_new = dict((e, i) for i, e in enumerate(encoded_ids))
    mask = np.asarray([e in encoded_seeds for e in encoded_ids])
    decoded_ids = [g.node_decode[e] for e in encoded_ids]

    x = np.asarray([
        fstore.get(e, default_feature) for e in decoded_ids
    ])
    x = convert_tensor(torch.FloatTensor(x), device=device, non_blocking=non_blocking)

    edge_list = [g.edge_list_encoded[:, idx] for idx in edge_ids]
    f = lambda x: encode_to_new[x]
    f = np.vectorize(f)
    edge_list = [f(e) for e in edge_list]
    edge_list = [
        convert_tensor(torch.LongTensor(e), device=device, non_blocking=non_blocking)
        for e in edge_list]

    y = np.asarray([
        -1 if e not in encoded_seeds else g.seed_label_encoded[e]
        for e in encoded_ids
    ])
    # assert (y >= 0).sum() == len(encoded_seeds)

    y = torch.LongTensor(y)
    y = convert_tensor(y, device=device, non_blocking=non_blocking)
    mask = torch.BoolTensor(mask)
    mask = convert_tensor(mask, device=device, non_blocking=non_blocking)

    y = y[mask]

    node_type_encode = g.node_type_encode
    node_type = [node_type_encode[g.node_type[e]] for e in decoded_ids]
    node_type = torch.LongTensor(np.asarray(node_type))
    node_type = convert_tensor(
        node_type, device=device, non_blocking=non_blocking)

    edge_type = [[g.edge_list_type_encoded[eid] for eid in list_] for list_ in edge_ids]
    edge_type = [torch.LongTensor(np.asarray(e)) for e in edge_type]
    edge_type = [convert_tensor(e, device=device, non_blocking=non_blocking) for e in edge_type]

    return ((mask, x, edge_list, node_type, edge_type), y)

def read_data(args):
    uid_cols = ['MessageId', 'Timestamp', 'UETR', 'Sender', 'Receiver', 'OrderingAccount', 'BeneficiaryAccount',
                'Label', 'OrderingOriginAdd', 'BeneficiaryOriginAdd']
    acc_df = pd.read_csv(args.acc_path).drop('OrderingAccount', axis=1)
    bank_df = pd.read_csv(args.bank_path).drop('Sender', axis=1)
    train_df = pd.read_csv(args.train_feat_path)
    test_df = pd.read_csv(args.test_feat_path)
    feature_cols = list(train_df.columns)
    for i in uid_cols:
        feature_cols.remove(i)
    train_df = train_df[['MessageId'] + feature_cols]
    test_df = test_df[['MessageId'] + feature_cols]
    acc_df = acc_df[['id'] + feature_cols]
    bank_df = bank_df[['id'] + feature_cols]
    feat_dict = {}
    x = train_df.values
    for i in tqdm(range(x.shape[0])):
        key = int(x[i, 0])
        value = x[i, 1:]
        feat_dict[key] = value
    x = test_df.values
    for i in tqdm(range(x.shape[0])):
        key = int(x[i, 0])
        value = x[i, 1:]
        feat_dict[key] = value
    x = acc_df.values
    for i in tqdm(range(x.shape[0])):
        key = int(x[i, 0])
        value = x[i, 1:]
        feat_dict[key] = value
    x = bank_df.values
    for i in tqdm(range(x.shape[0])):
        key = int(x[i, 0])
        value = x[i, 1:]
        feat_dict[key] = value
    return feat_dict

def create_modified_het_graph_from_edges(df, feat_dict):
    print('we are here 1')
    logger = logging.getLogger('factory-naive-het-graph')
    logger.setLevel(logging.INFO)

    print('we are here 2')
    with timeit(logger, 'node-type-init'):
        view = df[['MessageId', 'Days']].drop_duplicates()
        node_ts = dict((k, v) for k, v in view.itertuples(index=False))
        df['src_type'] = 0
        view = df[['MessageId', 'src_type']].drop_duplicates()
        node_type = dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        )
        df['sender_bank_type'] = 1
        view = df[['Sender', 'sender_bank_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
        view = df[['Receiver', 'sender_bank_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
        df['account_type'] = 2
        view = df[['OrderingAccount', 'account_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
        view = df[['BeneficiaryAccount', 'account_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
        df['address_type'] = 3
        view = df[['OrderingOriginAdd', 'address_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
        view = df[['BeneficiaryOriginAdd', 'address_type']].drop_duplicates()
        node_type.update(dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        ))
    print('we are here 3')
    # if 'sender_type' not in df:
    df['sender_type'] = 'sender'
    df['receiver_type'] = 'receiver'
    df['ordering_type'] = 'ordering'
    df['beneficiary_type'] = 'beneficiary'
    df['ordering_add_type'] = 'order_add'
    df['ben_add_type'] = 'ben_add'

    with timeit(logger, 'edge-list-init'):
        edge_list = list(df[['MessageId', 'Sender', 'sender_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
        edge_list += list(df[['MessageId', 'Receiver', 'receiver_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
        edge_list += list(
            df[['MessageId', 'OrderingAccount', 'ordering_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
        edge_list += list(
            df[['MessageId', 'BeneficiaryAccount', 'beneficiary_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
        edge_list += list(
            df[['MessageId', 'OrderingOriginAdd', 'ordering_add_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]
        edge_list += list(
            df[['MessageId', 'BeneficiaryOriginAdd', 'ben_add_type']].drop_duplicates().itertuples(index=False))
        edge_list += [(e1, e0, t) for e0, e1, t in edge_list]

    select = df['seed'] > 0
    view = df[select][['MessageId', 'Label']].drop_duplicates()
    seed_label = dict((k, v) for k, v in view.itertuples(index=False))

    return ModifiedHetGraph(node_type, edge_list, feat_dict=feat_dict, seed_label=seed_label, node_ts=node_ts)

def prepare_batch_para(batch, device, non_blocking=False):
    x, y = batch
    mask, x, edge_list, node_type, edge_type = x
    x = convert_tensor(x, device=device, non_blocking=non_blocking)
    edge_list = [convert_tensor(e, device=device, non_blocking=non_blocking) for e in edge_list]
    y = convert_tensor(y, device=device, non_blocking=non_blocking)
    mask = convert_tensor(mask, device=device, non_blocking=non_blocking)
    y = y[mask]
    node_type = convert_tensor(node_type, device=device, non_blocking=non_blocking)
    edge_type = [convert_tensor(e, device=device, non_blocking=non_blocking) for e in edge_type]
    return ((mask, x, edge_list, node_type, edge_type), y)


