import os
import logging
import tempfile
import subprocess
from functools import partial
import glob
from config import parse_args
import fire
import tqdm
import joblib
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ignite
from ignite.utils import convert_tensor
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.contrib.metrics import AveragePrecision, ROC_AUC
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler, PiecewiseLinear
import ignite.distributed as dist

# other things
from Utils.fstore import FeatureStore
from Utils.utils import create_modified_het_graph_from_edges as _create_modified_het_graph_from_edges
from Utils.utils import read_data as _read_data
from Graph.loader import NaiveHetDataLoader, ParallelHetDataLoader
from Graph.graph import NaiveHetGraph, ModifiedHetGraph
from Model.model import GNN, HetNet as Net, HetNetLogi as NetLogi
from Utils.utils import timeit, prepare_batch_para

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)


def prepare_data(rank, world_size, args, graph, pin_memory=False):
    dl_train = ParallelHetDataLoader(rank=rank, world_size=world_size,
                                     width=args.width, depth=args.depth,
                                     g=graph, ts_range=args.train_range, method=args.sample_method,
                                     batch_size=args.batch_size, n_batch=args.n_batch, seed_epoch=False,
                                     num_workers=args.num_workers, shuffle=False, pin_memory=pin_memory, seed=args.seed)

    dl_valid = ParallelHetDataLoader(rank=rank, world_size=world_size,
                                     width=args.width, depth=args.depth,
                                     g=graph, ts_range=args.valid_range, method=args.sample_method,
                                     batch_size=args.batch_size, n_batch=args.n_batch, seed_epoch=True,
                                     num_workers=args.num_workers, shuffle=False, pin_memory=pin_memory, seed=args.seed,
                                     cache_result=True)

    dl_test = ParallelHetDataLoader(rank=rank, world_size=world_size,
                                    width=args.width, depth=args.depth,
                                    g=graph, ts_range=args.test_range, method=args.sample_method,
                                    batch_size=args.batch_size, n_batch=args.n_batch, seed_epoch=True,
                                    num_workers=args.num_workers, shuffle=False, pin_memory=pin_memory, seed=args.seed,
                                    cache_result=True)
    return dl_train, dl_valid, dl_test

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
    model = dist.auto_model(model)
    return model

def prepare_optimizer(args, model):
    optimizer = None
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters())
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters())
    optimizer = dist.auto_optim(optimizer)
    return optimizer

def get_criterion():
    return nn.CrossEntropyLoss().to(dist.device())

def get_lr_scheduler(args, optimizer):
    milestones_values = [
        (0, 0.0),
        (args.num_iters_per_epoch * args.num_warmup_epochs, args.lr),
        (args.num_iters_per_epoch * args.n_step, 0.0),
    ]
    lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
    )
    return lr_scheduler

def get_save_handler(args):
    if args.with_clearml:
        from ignite.contrib.handlers.clearml_logger import ClearMLSaver
        return ClearMLSaver(dirname=args.save_path)
    return args.save_path

def load_checkpoint(resume_from):
    checkpoint_fp = Path(resume_from)
    assert (
        checkpoint_fp.exists()
    ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    return checkpoint

#
# def train_step(engine, batch):
#     x, y = batch[0], batch[1]
#     if x.device != device:
#         x = x.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)
#
#     model.train()
#
#     with autocast(enabled=with_amp):
#         y_pred = model(x)
#         loss = criterion(y_pred, y)
#
#     optimizer.zero_grad()
#     scaler.scale(loss).backward()  # If with_amp=False, this is equivalent to loss.backward()
#     scaler.step(optimizer)  # If with_amp=False, this is equivalent to optimizer.step()
#     scaler.update()  # If with_amp=False, this step does nothing
#
#     return {"batch loss": loss.item()}


def main(args):

    mem = joblib.Memory('./data/cache')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    create_modified_het_graph_from_edges = mem.cache(_create_modified_het_graph_from_edges)
    read_data = mem.cache(_read_data)
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
    if not os.path.isdir(args.dir_model):
        os.makedirs(args.dir_model)
    with timeit(logger, 'read-data'):
        index_dict, feat_dict = read_data(args)
    with timeit(logger, 'edge-load'):
        df_edges = pd.read_csv(args.path_g)
    if args.debug:
        logger.info('Main in debug mode.')
        df_edges = df_edges.iloc[3079964:]
    if 'seed' not in df_edges:
        df_edges['seed'] = 1
    with timeit(logger, 'g-init'):
        g = create_modified_het_graph_from_edges(df=df_edges, index_dict=index_dict, feat_dict=feat_dict)
    print(g)
    exit()


