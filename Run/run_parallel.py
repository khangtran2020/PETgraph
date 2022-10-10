import os
import logging
import tempfile
import subprocess
from functools import partial
import glob

import ignite
from torch.cuda.amp import autocast

from config import parse_args
import fire
import tqdm
import time
import joblib
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ignite
from ignite.utils import convert_tensor, manual_seed, setup_logger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.contrib.metrics import AveragePrecision, ROC_AUC
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler, PiecewiseLinear
import ignite.distributed as dist
from ignite.engine import EventEnum, _prepare_batch
from ignite.engine.deterministic import DeterministicEngine
from ignite.contrib.engines import common
from ignite.contrib.handlers.tqdm_logger import ProgressBar
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


class ForwardEvents(EventEnum):
    FORWARD_STARTED = 'forward_started'
    FORWARD_COMPLETED = 'forward_completed'


def log_basic_info(logger, config):
    logger.info(f"Train on CIFAR10")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(
            f"- GPU Device: {torch.cuda.get_device_name(dist.get_local_rank())}"
        )
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    # logger.info("\n")
    # logger.info("Configuration:")
    # for key, value in config.items():
    #     logger.info(f"\t{key}: {value}")
    # logger.info("\n")

    if dist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {dist.backend()}")
        logger.info(f"\tworld size: {dist.get_world_size()}")
        logger.info("\n")


def udf_supervised_trainer(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[Callable, torch.nn.Module],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = _prepare_batch,
        output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
        deterministic: bool = False,
) -> Engine:
    device_type = device.type if isinstance(device, torch.device) else device

    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

        engine.fire_event(ForwardEvents.FORWARD_STARTED)
        y_pred = model(x)
        engine.fire_event(ForwardEvents.FORWARD_COMPLETED)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        return output_transform(x, y, y_pred, loss)

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)
    trainer.register_events(*ForwardEvents)

    return trainer


def setup_rank_zero(logger, config):
    device = dist.device()

    now = time.time()
    output_path = config.save_path
    folder_name = (
        f"{config.conv_name}_backend-{dist.backend()}-{dist.get_world_size()}_{now}"
    )
    output_path = Path(output_path) / folder_name
    if not output_path.exists():
        output_path.mkdir(parents=True)
    config.save_path = output_path.as_posix()
    logger.info(f"Output path: {config.save_path}")


def training(local_rank, config, g, result_dict):
    stats = {}
    rank = dist.get_rank()
    manual_seed(config.seed + rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")

    logger = setup_logger(name="PET-Training")
    log_basic_info(logger, config)

    if rank == 0:
        setup_rank_zero(logger, config)

    dl_train, dl_valid, dl_test = prepare_data(rank=rank, world_size=dist.get_world_size(), args=config, graph=g,
                                               pin_memory=True)
    model = prepare_model(config)
    optimizer = prepare_optimizer(config, model)
    criterion = get_criterion()
    config.num_iters_per_epoch = len(dl_train)
    pb = partial(prepare_batch_para, device=device)
    trainer = udf_supervised_trainer(model=model, optimizer=optimizer, loss_fn=criterion, device=dist.device(),
                                     prepare_batch=pb)
    scheduler = CosineAnnealingScheduler(
        optimizer, 'lr',
        start_value=0.05, end_value=1e-4,
        cycle_size=len(dl_train) * config.n_step)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    evaluator = create_supervised_evaluator(model, metrics={
        'accuracy': Accuracy(),
        'loss': Loss(criterion),
        'ap': AveragePrecision(
            output_transform=lambda out: (out[0][:, 1], out[1])),
        'auc': ROC_AUC(
            output_transform=lambda out: (out[0][:, 1], out[1])),
    }, device=device, prepare_batch=pb, amp_mode=True)
    pbar_train = tqdm.tqdm(desc='train', total=len(dl_train), ncols=100)
    t_epoch = Timer(average=True)
    t_epoch.pause()

    t_iter = Timer(average=True)
    t_iter.pause()

    @trainer.on(ForwardEvents.FORWARD_STARTED)
    def resume_timer(engine):
        t_epoch.resume()
        t_iter.resume()

    @trainer.on(ForwardEvents.FORWARD_COMPLETED)
    def pause_timer(engine):
        t_epoch.pause()
        t_iter.pause()
        t_iter.step()

    @trainer.on(Events.EPOCH_STARTED)
    def log_training_loss(engine):
        pbar_train.refresh()
        pbar_train.reset()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        pbar_train.update(1)
        pbar_train.set_description(
            'Train [Eopch %03d] Loss %.4f T-iter %.4f' % (
                engine.state.epoch, engine.state.output, t_iter.value()
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        t_epoch.step()
        evaluator.run(dl_valid)
        metrics = evaluator.state.metrics
        logger.info(
            '[Epoch %03d]\tLoss %.4f\tAccuracy %.4f\tAUC %.4f\tAP %.4f \tTime %.2f / %03d',
            engine.state.epoch,
            metrics['loss'], metrics['accuracy'],
            metrics['auc'], metrics['ap'],
            t_epoch.value(), t_epoch.step_count
        )
        if (rank == 0):
            result_dict['loss'].append(metrics['loss'])
            result_dict['acc'].append(metrics['accuracy'])
            result_dict['auc'].append(metrics['auc'])
            result_dict['ap'].append(metrics['ap'])

        t_iter.reset()
        t_epoch.pause()
        t_iter.pause()

    def score_function(engine):
        return engine.state.metrics['auc']

    pbar = ProgressBar()
    pbar.attach(evaluator, [])
    handler = EarlyStopping(patience=config.patient, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    cp = ModelCheckpoint(config.dir_model, f'model-{config.conv_name}-{config.seed}', n_saved=1,
                         create_dir=True,
                         score_function=lambda e: evaluator.state.metrics['auc'],
                         require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, cp, {config.conv_name: model})

    if rank == 0:
        evaluators = {"train": evaluator, "val": evaluator}
        tb_logger = common.setup_tb_logging(
            config.save_path, trainer, optimizer, evaluators=evaluators
        )

    try:
        trainer.run(dl_train, max_epochs=config.n_step)
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        df = pd.DataFrame.from_dict(result_dict)
        df.to_csv(config.res_path + 'train_proces_width_{}_depth_{}_batch_size_{}_{}.csv'.format(config.width,
                                                                                                    config.depth,
                                                                                                    config.batch_size[
                                                                                                        0],
                                                                                                    config.batch_size[
                                                                                                        1]))
        tb_logger.close()
        # path_model = cp.last_checkpoint
        # model.load_state_dict(torch.load(path_model))
        model.eval()
        with torch.no_grad():
            evaluator.run(dl_test)
        metrics = evaluator.state.metrics
        logger.info(
            'Test\tLoss %.2f\tAccuracy %.2f\tAUC %.4f\tAP %.4f',
            metrics['loss'], metrics['accuracy'],
            metrics['auc'], metrics['ap']
        )

        stats.update(dict(metrics))

        stats['epoch'] = trainer.state.epoch,

        row = pd.DataFrame([stats])
        if os.path.exists(config.path_result):
            result = pd.read_csv(config.path_result)
        else:
            result = pd.DataFrame()
        result = result.append(row)
        result.to_csv(config.path_result, index=False)


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
        print(len(index_dict))
    with timeit(logger, 'edge-load'):
        df_edges = pd.read_csv(args.path_g)
    if args.debug:
        logger.info('Main in debug mode.')
        df_edges = df_edges.iloc[3079964:]
    if 'seed' not in df_edges:
        df_edges['seed'] = 1
    with timeit(logger, 'g-init'):
        g = create_modified_het_graph_from_edges(df=df_edges, index_dict=index_dict, feat_dict=feat_dict)

    seed_set = set(df_edges.query('seed>0')['MessageId'])
    logger.info('#seed %d', len(seed_set))
    if args.debug:
        train_range = set(range(15, 23))
        valid_range = set(range(23, 24))
        test_range = set(range(24, 31))
    else:
        train_range = set(range(1, 22))
        valid_range = set(range(22, 24))
        test_range = set(range(24, 31))
    logger.info('Range Train %s\t Valid %s\t Test %s',
                train_range, valid_range, test_range)
    print(g.get_seed_nodes(train_range)[0])
    x0 = g.get_feat(idx=0)
    print(x0)
    assert x0 is not None
    args.num_feat = x0.shape[0]
    args.num_node_type = len(g.node_type_encode)
    args.num_edge_type = len(g.edge_type_encode)
    args.train_range = train_range
    args.valid_range = valid_range
    args.test_range = test_range
    print(args.num_feat, args.num_node_type)
    # exit()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    result_dict = {
        'loss': [],
        'acc': [],
        'auc': [],
        'ap': []
    }

    with dist.Parallel(backend=args.backend, nproc_per_node=args.nproc_per_node) as parallel:
        parallel.run(training, config=args, g=g, result_dict=result_dict)

    #
