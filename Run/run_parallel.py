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
from Utils.utils import create_naive_het_graph_from_edges as _create_naive_het_graph_from_edges
from Graph.loader import NaiveHetDataLoader
from Graph.graph import NaiveHetGraph, ModifiedHetGraph
from Model.model import GNN, HetNet as Net, HetNetLogi as NetLogi
from Utils.utils import timeit
from Utils.utils import prepare_batch

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)

def main(args):
    pass