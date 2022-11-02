import argparse


def add_general_group(group):
    group.add_argument("--save_path", type=str, default="results/", help="dir path for saving model file")
    group.add_argument("--res_path", type=str, default="results/", help="dir path for output file")
    group.add_argument("--seed", type=int, default=1, help="seed value")
    group.add_argument("--seed_epoch", type=bool, default=False, help="parameter from xfraud for caching data")
    group.add_argument("--mode", type=str, default='central', help="Mode of running ['central', 'parallel']")
    group.add_argument("--backend", type=str, default='nccl', help="backend for parallel training")
    group.add_argument("--nproc_per_node", type=int, default=4, help="number of GPUs")


def add_data_group(group):
    group.add_argument('--train_feat_path', type=str, default='data/processed_train.csv', help="training transaction files")
    group.add_argument('--test_feat_path', type=str, default='data/processed_test.csv',help="testing transaction files")
    group.add_argument('--acc_path', type=str, default='data/account_feature.csv',help="feature of accounts")
    group.add_argument('--add_path', type=str, default='data/address_feature.csv',help="feature of address")
    group.add_argument('--bank_path', type=str, default='data/bank_feature.csv',help="feature of bank")
    group.add_argument('--path_feat_db', type=str, default='data/feat_store.db', help="the directory of the feature database")
    group.add_argument('--path_g', type=str, default='data/full_graph.csv', help="the connections (edges) in the graph")

def add_model_group(group):
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    group.add_argument("--conv_name", type=str, default='het-emb', help="type of conv layer")
    group.add_argument('--sample_method', type=str, default='sage', help="node sampling method in GNN training")
    group.add_argument('--width', type=int, default=16, help='number of nodes to sample per layer')
    group.add_argument('--depth', type=int, default=6, help='equal to number of layers')
    group.add_argument('--batch_size_0', type=int, default=256, help="number of non-fraud transaction sampled in 1 batch")
    group.add_argument('--batch_size_1', type=int, default=64, help="number of fraud transaction sampled in 1 batch")
    group.add_argument('--n_hid', type=int, default=400, help='number hidden embedding dim')
    group.add_argument('--n_heads', type=int, default=8, help='number attention head')
    group.add_argument('--n_layers', type=int, default=6, help='number of layers')
    group.add_argument("--drop_out", type=float, default=0.2)
    group.add_argument("--clip", type=float, default=0.5)
    group.add_argument("--optimizer", type=str, default='adamw')
    group.add_argument("--n_batch", type=int, default=32)
    group.add_argument("--n_step", type=int, default=50, help='training step')
    group.add_argument("--patient", type=int, default=8)
    group.add_argument("--continue_training", type=bool, default=False)
    group.add_argument("--debug", type=bool, default=False)
    group.add_argument("--dir_model", type=str, default='model/')
    group.add_argument("--num_workers", type=int, default=0)
    group.add_argument("--path_result", type=str, default='results/exp_result.csv', help='name file for result printing')
    group.add_argument("--num_warmup_epochs", type=int, default=1)
    group.add_argument("--with_clearml", type=bool, default=False)
    group.add_argument("--with_amp", type=bool, default=False)


def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    general_group = parser.add_argument_group(title="General configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    return parser.parse_args()
