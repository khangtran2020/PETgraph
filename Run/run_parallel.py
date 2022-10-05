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
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

# other things
from Utils.fstore import FeatureStore
from Utils.utils import create_modified_het_graph_from_edges as _create_modifed_het_graph_from_edges
from Graph.loader import NaiveHetDataLoader
from Graph.graph import NaiveHetGraph, ModifiedHetGraph
from Model.model import GNN, HetNet as Net, HetNetLogi as NetLogi
from Utils.utils import timeit, prepare_batch, read_data

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)

def main(args):
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
    mem = joblib.Memory('./data/cache')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    create_modified_het_graph_from_edges = mem.cache(_create_modifed_het_graph_from_edges)
    if args.conv_name == '' or args.conv_name == 'logi':
        args.width, args.depth = 1, 1
    with timeit(logger, 'read-data'):
        feat_dict = read_data(args)
    with timeit(logger, 'edge-load'):
        df_edges = pd.read_csv(args.path_g)
    if args.debug:
        logger.info('Main in debug mode.')
        df_edges = df_edges.iloc[3079964:]
    if 'seed' not in df_edges:
        df_edges['seed'] = 1
    with timeit(logger, 'g-init'):
        g = create_modified_het_graph_from_edges(df_edges)

