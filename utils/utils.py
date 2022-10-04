import os
import sys
path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
import time
import logging
import numpy as np
from contextlib import contextmanager
from utils.graph_loader import GraphData, NaiveHetGraph
from collections import defaultdict

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


def create_naive_het_graph_from_edges(df, store):
    logger = logging.getLogger('factory-naive-het-graph')
    logger.setLevel(logging.INFO)

    with timeit(logger, 'node-type-init'):
        view = df[['MessageId', 'Days']].drop_duplicates()
        node_ts = dict((k, v) for k, v in view.itertuples(index=False))
        df['src_type'] = 1
        view = df[['MessageId', 'src_type']].drop_duplicates()
        node_type = dict(
            (node, tp)
            for node, tp in view.itertuples(index=False)
        )
        df['sender_bank_type'] = 0
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
        df['account_type'] = 1
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
        df['address_type'] = 2
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
                         seed_label=seed_label, node_ts=node_ts, feat_store=store)


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

