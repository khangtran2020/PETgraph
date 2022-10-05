import os
import logging
import joblib
import torch
from config import parse_args
from Run.run_centralize import main as run_central
from Run.run_parallel import main as run_para
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)
mem = joblib.Memory('./data/cache')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    args = parse_args()
    args.batch_size = (args.batch_size_0, args.batch_size_1)
    if args.main_mode == 'central':
        run_central(args=args)
    else:
        run_para(args)

