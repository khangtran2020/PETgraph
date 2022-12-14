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

from ignite.utils import convert_tensor
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.contrib.metrics import AveragePrecision, ROC_AUC
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

from Utils.fstore import FeatureStore
from Utils.utils import create_naive_het_graph_from_edges as _create_naive_het_graph_from_edges
from Graph.loader import NaiveHetDataLoader
from Graph.graph import NaiveHetGraph, ModifiedHetGraph
from Model.model import GNN, HetNet as Net, HetNetLogi as NetLogi
from Utils.utils import timeit
from Utils.utils import prepare_batch

def main(args):
    """
    :param path_g:          path of graph file
    :param path_feat_db:    path of feature store db
    :param path_result:     path of output result csv file
    :param dir_model:       path of model saving
    :param conv_name:       model convolution layer type, choices ['', 'logi', 'gcn', 'gat', 'hgt', 'het-emb']
    :param sample_method:
    :param batch_size:      positive/negative samples per batch
    :param width:           sample width
    :param depth:           sample depth
    :param n_hid:           num of hidden state
    :param n_heads:
    :param n_layers:        num of convolution layers
    :param dropout:
    :param optimizer:
    :param clip:
    :param n_batch:
    :param max_epochs:
    :param patience:
    :param seed_epoch:      True -> iter on all seeds; False -> sample seed according to batch_size
    :param num_workers:
    :param seed:            random seed
    :param debug:           debug mode
    :param continue_training:
    :return:
    """
    mem = joblib.Memory('./data/cache')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    create_naive_het_graph_from_edges = mem.cache(_create_naive_het_graph_from_edges)
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

    with tempfile.TemporaryDirectory() as tmpdir:
        path_feat_db_temp = f'{tmpdir}/store.db'

        with timeit(logger, 'fstore-init'):
            subprocess.check_call(
                f'cp -r {args.path_feat_db} {path_feat_db_temp}',
                shell=True)

            store = FeatureStore(path_feat_db_temp)

        if not os.path.isdir(args.dir_model):
            os.makedirs(args.dir_model)
        with timeit(logger, 'edge-load'):
            df_edges = pd.read_csv(args.path_g)
        if args.debug:
            logger.info('Main in debug mode.')
            df_edges = df_edges.iloc[3079964:]
        if 'seed' not in df_edges:
            df_edges['seed'] = 1
        with timeit(logger, 'g-init'):
            g = create_naive_het_graph_from_edges(df_edges)
        # print(g)
        # exit()

        seed_set = set(df_edges.query('seed>0')['MessageId'])
        logger.info('#seed %d', len(seed_set))
        if args.debug:
            train_range = set(range(15, 22))
            valid_range = set(range(22, 24))
            test_range = set(range(24, 31))
        else:
            train_range = set(range(1,22))
            valid_range = set(range(22,24))
            test_range = set(range(24,31))
        logger.info('Range Train %s\t Valid %s\t Test %s',
                    train_range, valid_range, test_range)
        print(g.get_seed_nodes(train_range)[0])
        x0 = store.get(g.get_seed_nodes(train_range)[0], None)
        assert x0 is not None
        num_feat = x0.shape[0]

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        dl_train = NaiveHetDataLoader(
            width=args.width, depth=args.depth,
            g=g, ts_range=train_range, method=args.sample_method,
            batch_size=args.batch_size, n_batch=args.n_batch,
            seed_epoch=args.seed_epoch, num_workers=args.num_workers, shuffle=True)

        dl_valid = NaiveHetDataLoader(
            width=args.width, depth=args.depth,
            g=g, ts_range=valid_range, method=args.sample_method,
            batch_size=args.batch_size, n_batch=args.n_batch,
            seed_epoch=True, num_workers=args.num_workers, shuffle=False,
            cache_result=True)

        dl_test = NaiveHetDataLoader(
            width=args.width, depth=args.depth,
            g=g, ts_range=test_range, method=args.sample_method,
            batch_size=args.batch_size, n_batch=args.n_batch,
            seed_epoch=True, num_workers=args.num_workers, shuffle=False,
            cache_result=True)

        logger.info('Len dl train %d, valid %d, test %d.',
                    len(dl_train), len(dl_valid), len(dl_test))
        # for _ in tqdm.tqdm(dl_test, desc='gen-test-dl', ncols=80):
        #     pass

        num_node_type = len(g.node_type_encode)
        num_edge_type = len(g.edge_type_encode)
        logger.info('#node_type %d, #edge_type %d', num_node_type, num_edge_type)

        if args.conv_name != 'logi':
            if args.conv_name == '':
                gnn = None
            else:
                gnn = GNN(conv_name=args.conv_name,
                          n_in=num_feat,
                          n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers,
                          dropout=args.drop_out,
                          num_node_type=num_node_type,
                          num_edge_type=num_edge_type
                          )

            model = Net(gnn, num_feat, num_embed=args.n_hid, n_hidden=args.n_hid)
        else:
            model = NetLogi(num_feat)
        model.to(device)

        model_loaded = False
        if args.continue_training:
            files = glob.glob(f'{args.dir_model}/model-{args.conv_name}-{args.seed}*')
            if len(files) > 0:
                files.sort(key=os.path.getmtime)
                load_file = files[-1]
                logger.info(f'Continue training from checkpoint {load_file}')
                model.load_state_dict(torch.load(load_file))
                model_loaded = True

        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters())
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters())
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        elif args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters())

        pb = partial(
            prepare_batch, g=g, fstore=store, ts_range=train_range,
            default_feature=np.zeros_like(x0))

        loss = nn.CrossEntropyLoss()

        from ignite.engine import EventEnum, _prepare_batch
        from ignite.distributed import utils as idist
        from ignite.engine.deterministic import DeterministicEngine
        if idist.has_xla_support:
            import torch_xla.core.xla_model as xm

        from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

        class ForwardEvents(EventEnum):
            FORWARD_STARTED = 'forward_started'
            FORWARD_COMPLETED = 'forward_completed'

        """create UDF trainer to register forward events"""
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
            on_tpu = "xla" in device_type if device_type is not None else False

            if on_tpu and not idist.has_xla_support:
                raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

            def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
                model.train()
                optimizer.zero_grad()
                x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

                engine.fire_event(ForwardEvents.FORWARD_STARTED)
                y_pred = model(x)
                engine.fire_event(ForwardEvents.FORWARD_COMPLETED)

                loss = loss_fn(y_pred, y)
                loss.backward()

                if on_tpu:
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()

                return output_transform(x, y, y_pred, loss)

            trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)
            trainer.register_events(*ForwardEvents)

            return trainer

        trainer = udf_supervised_trainer(
            model, optimizer, loss,
            device=device, prepare_batch=pb)

        pb = partial(
            prepare_batch, g=g, fstore=store, ts_range=valid_range,
            default_feature=np.zeros_like(x0))
        evaluator = create_supervised_evaluator(model,
                                                metrics={
                                                    'accuracy': Accuracy(),
                                                    'loss': Loss(loss),
                                                    'ap': AveragePrecision(
                                                        output_transform=lambda out: (out[0][:, 1], out[1])),
                                                    'auc': ROC_AUC(
                                                        output_transform=lambda out: (out[0][:, 1], out[1])),
                                                }, device=device, prepare_batch=pb)

        if model_loaded:
            with torch.no_grad():
                evaluator.run(dl_test)
            metrics = evaluator.state.metrics
            logger.info(
                'Loaded model stat: Test\tLoss %.2f\tAccuracy %.2f\tAUC %.4f\tAP %.4f',
                metrics['loss'], metrics['accuracy'],
                metrics['auc'], metrics['ap']
            )

        scheduler = CosineAnnealingScheduler(
            optimizer, 'lr',
            start_value=0.05, end_value=1e-4,
            cycle_size=len(dl_train) * args.n_step)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

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
            t_iter.reset()
            t_epoch.pause()
            t_iter.pause()

        def score_function(engine):
            return engine.state.metrics['auc']

        handler = EarlyStopping(patience=args.patient, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, handler)

        cp = ModelCheckpoint(args.dir_model, f'model-{args.conv_name}-{args.seed}', n_saved=1,
                             create_dir=True,
                             score_function=lambda e: evaluator.state.metrics['auc'],
                             require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, cp, {args.conv_name: model})

        trainer.run(dl_train, max_epochs=args.n_step)

        path_model = cp.last_checkpoint
        model.load_state_dict(torch.load(path_model))
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
        if os.path.exists(args.path_result):
            result = pd.read_csv(args.path_result)
        else:
            result = pd.DataFrame()
        result = result.append(row)
        result.to_csv(args.path_result, index=False)
