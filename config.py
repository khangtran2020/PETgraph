import argparse


def add_general_group(group):
    group.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    group.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    group.add_argument("--save_path", type=str, default="results/", help="dir path for output file")
    group.add_argument("--seed", type=int, default=1, help="seed value")
    group.add_argument("--mode", type=str, default='train', help="Mode of running")


def add_data_group(group):
    group.add_argument('--data_path', type=str, default='data/', help="used dataset")
    group.add_argument('--train_feat_path', type=str, default='processed_train.csv', help="Use embedding for LDP or not ")
    group.add_argument('--test_feat_path', type=str, default='processed_test.csv',
                       help="the directory used to save dataset")
    group.add_argument('--acc_path', type=str, default='account_feature.csv',
                       help="Use embedding for LDP or not ")
    group.add_argument('--add_path', type=str, default='address_feature.csv',
                       help="the directory used to save dataset")
    group.add_argument('--bank_path', type=str, default='bank_feature.csv',
                       help="the directory used to save dataset")


def add_model_group(group):
    group.add_argument("--lr", type=float, default=0.01, help="learning rate")
    group.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    group.add_argument("--conv_name", type=str, default='gcn', help="embedding dim")
    group.add_argument('--sample_method', type=str, default='sage')
    group.add_argument('--batch_size', action=tuple, default=(64,16), help="print training details")
    group.add_argument('--n_hid', type=int, default=400, help='print every x epoch')
    group.add_argument('--n_heads', type=int, default=8, help='evaluate every x epoch')
    group.add_argument('--n_layers', type=int, default=6)
    group.add_argument("--drop_out", type=float, default=0.2)
    group.add_argument("--clip", type=float, default=0.5)
    group.add_argument("--optimizer", type=str, default='adamw')
    group.add_argument("--n_batch", type=int, default=32)
    group.add_argument("--n_step", type=int, default=50)
    group.add_argument("--patient", type=int, default=8)
    group.add_argument("--continue_training", type=bool, default=False)
    group.add_argument("--debug", type=bool, default=False)

def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    general_group = parser.add_argument_group(title="General configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    return parser.parse_args()
