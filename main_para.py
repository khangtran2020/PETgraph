import os
import logging
import tempfile
import subprocess
from functools import partial
import glob
from config import parse_args

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)

import fire
import tqdm
import joblib
import numpy as np
import pandas as pd
import datetime

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from utils.fstore import FeatureStore
from utils.utils import create_naive_het_graph_from_edges as _create_naive_het_graph_from_edges
from utils.graph_loader import NaiveHetGraph, ParallelHetDataLoader, NaiveHetDataLoader
from utils.model import GNN, HetNet as Net, HetNetLogi as NetLogi
from utils.utils import timeit


def prepare_data(rank, world_size, args, graph, pin_memory=False):
    dl_train = ParallelHetDataLoader(rank=rank, world_size=world_size,
                                     width=args.width, depth=args.depth,
                                     g=graph, ts_range=args.train_range, method=args.sample_method,
                                     batch_size=args.batch_size, n_batch=args.n_batch, seed_epoch=False,
                                     num_workers=args.num_workers, shuffle=True, pin_memory=pin_memory, seed=args.seed)

    dl_valid = ParallelHetDataLoader(rank=rank, world_size=world_size,
                                     width=args.width, depth=args.depth,
                                     g=graph, ts_range=args.valid_range, method=args.sample_method,
                                     batch_size=args.batch_size, n_batch=args.n_batch, seed_epoch=True,
                                     num_workers=args.num_workers, shuffle=True, pin_memory=pin_memory, seed=args.seed,
                                     cache_result=True)

    dl_test = ParallelHetDataLoader(rank=rank, world_size=world_size,
                                    width=args.width, depth=args.depth,
                                    g=graph, ts_range=args.test_range, method=args.sample_method,
                                    batch_size=args.batch_size, n_batch=args.n_batch, seed_epoch=True,
                                    num_workers=args.num_workers, shuffle=True, pin_memory=pin_memory, seed=args.seed,
                                    cache_result=True)
    return dl_train, dl_valid, dl_test


def prepare_batch(batch, ts_range, fstore, default_feature,
                  g: NaiveHetGraph, non_blocking=False):
    encoded_seeds, encoded_ids, edge_ids = batch
    encoded_seeds = set(encoded_seeds)
    encode_to_new = dict((e, i) for i, e in enumerate(encoded_ids))
    mask = np.asarray([e in encoded_seeds for e in encoded_ids])
    decoded_ids = [g.node_decode[e] for e in encoded_ids]

    x = np.asarray([
        fstore.get(e, default_feature) for e in decoded_ids
    ])
    x = torch.FloatTensor(x).cuda(non_blocking=non_blocking)
    edge_list = [g.edge_list_encoded[:, idx] for idx in edge_ids]
    f = lambda x: encode_to_new[x]
    f = np.vectorize(f)
    edge_list = [f(e) for e in edge_list]
    edge_list = [torch.LongTensor(e).cuda(non_blocking=non_blocking) for e in edge_list]
    y = np.asarray([
        -1 if e not in encoded_seeds else g.seed_label_encoded[e]
        for e in encoded_ids
    ])
    # assert (y >= 0).sum() == len(encoded_seeds)

    y = torch.LongTensor(y)
    y = y.cuda(non_blocking=non_blocking)
    mask = torch.BoolTensor(mask)
    mask = mask.cuda(non_blocking=non_blocking)
    y = y[mask]
    node_type_encode = g.node_type_encode
    node_type = [node_type_encode[g.node_type[e]] for e in decoded_ids]
    node_type = torch.LongTensor(np.asarray(node_type))
    node_type = node_type.cuda(non_blocking=non_blocking)

    edge_type = [[g.edge_list_type_encoded[eid] for eid in list_] for list_ in edge_ids]
    edge_type = [torch.LongTensor(np.asarray(e)) for e in edge_type]
    edge_type = [e.cuda(non_blocking=non_blocking) for e in edge_type]

    return ((mask, x, edge_list, node_type, edge_type), y)


def prepare_model(args):
    if args.conv_name != 'logi':
        if args.conv_name == '':
            gnn = None
        else:
            gnn = GNN(conv_name=args.conv_name,
                      n_in=args.num_feat,
                      n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers,
                      dropout=args.drop_out,
                      num_node_type=args.num_node_type,
                      num_edge_type=args.num_edge_type
                      )

        model = Net(gnn, args.num_feat, num_embed=args.n_hid, n_hidden=args.n_hid)
    else:
        model = NetLogi(args.num_feat)
    if args.continue_training:
        files = glob.glob(f'{args.dir_model}/model-{args.conv_name}-{args.seed}*')
        if len(files) > 0:
            files.sort(key=os.path.getmtime)
            load_file = files[-1]
            logger.info(f'Continue training from checkpoint {load_file}')
            model.load_state_dict(torch.load(load_file))
    return model


def prepare_optimizer(args, model):
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters())
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters())
    return optimizer


def train(gpu, args, graph, default_feat):
    print("Begin training process")
    rank = args.nr * args.gpus + gpu
    print("Current rank is: {}".format(rank))
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    print("Begin load data at rank {}".format(rank))
    dl_train, dl_valid, dl_test = prepare_data(rank=rank, world_size=args.world_size, args=args, graph=graph)
    print(
        "Done load data with train len: {}, valid len: {}, test len: {} at rank {}".format(len(dl_train), len(dl_valid),
                                                                                           len(dl_test), rank))
    model = prepare_model(args=args)
    torch.cuda.set_device(rank)
    print('GPU using is: {}'.format(torch.cuda.current_device()))
    model.cuda()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = prepare_optimizer(args=args, model=model)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code

    # start = datetime.now()
    total_step = len(dl_train)
    print("Train with num step for each epoch: {}".format(total_step))
    for epoch in range(args.epochs):
        for i, data in enumerate(dl_train):
            x, y = prepare_batch(batch=data, ts_range=args.train_range, fstore=store, default_feature=default_feat,
                                 g=graph, non_blocking=True)
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete")


def main(args):
    if not os.path.isdir(args.dir_model):
        os.makedirs(args.dir_model)
    with timeit(logger, 'edge-load'):
        df_edges = pd.read_csv(args.path_g)
    if args.debug:
        logger.info('Main in debug mode.')
        df_edges = df_edges.iloc[4079964:]
    if 'seed' not in df_edges:
        df_edges['seed'] = 1
    with timeit(logger, 'g-init'):
        g = _create_naive_het_graph_from_edges(df_edges)

    seed_set = set(df_edges.query('seed>0')['MessageId'])
    logger.info('#seed %d', len(seed_set))
    if args.debug:
        train_range = set(range(15, 22))
        valid_range = set(range(22, 24))
        test_range = set(range(24, 31))
    else:
        train_range = set(range(1, 22))
        valid_range = set(range(22, 24))
        test_range = set(range(24, 31))
    logger.info('Range Train %s\t Valid %s\t Test %s',
                train_range, valid_range, test_range)
    print(g.get_seed_nodes(train_range)[0])
    x0 = store.get(g.get_seed_nodes(train_range)[0], None)
    assert x0 is not None
    num_feat = x0.shape[0]
    args.num_node_type = len(g.node_type_encode)
    args.num_edge_type = len(g.edge_type_encode)
    args.num_feat = num_feat
    args.train_range = train_range
    args.valid_range = valid_range
    args.test_range = test_range
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mp.spawn(train, nprocs=args.gpus, args=(args, g, np.zeros_like(x0)))


if __name__ == "__main__":
    args = parse_args()
    args.batch_size = (args.batch_size_0, args.batch_size_1)
    args.world_size = args.gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    if args.conv_name == '' or args.conv_name == 'logi':
        args.width, args.depth = 1, 1

    stats = dict(
        batch_size=args.batch_size,
        width=args.width, depth=args.depth,
        n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.drop_out,
        conv_name=args.conv_name, optimizer=str(args.optimizer), clip=args.clip,
        max_epochs=args.n_step, patience=args.patient,
        seed=args.seed, path_g=args.path_g,
        sample_method=args.sample_method, path_feat_db=args.path_feat_db,
    )
    logger.info('Param %s', stats)
    store = FeatureStore(args.path_feat_db)
    main(args)
