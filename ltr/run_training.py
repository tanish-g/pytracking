import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import ltr.admin.settings as ws_settings


def run_training(train_module, train_name,prune,s,ckpt_path,epochs,cudnn_benchmark=True):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'ltr/{}/{}/{}'.format(train_module,ckpt_path,train_name)
    settings.prune = False
    settings.s = s
    settings.ckpt_path = ckpt_path
    settings.max_epoch = epochs
    
    print('MAX_EPOCHS:',settings.max_epoch,' ','CKPT_TYPE:',settings.ckpt_path,' ','PRUNE:',settings.prune,' ','Sparsity_coeffecient:',settings.s)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--prune', type=bool,help='Sparsity Training')
    parser.add_argument('--s', type=float, default=1e-5, help='Sparsity Parameter')
    parser.add_argument('--ckpt',type=str,help='Model save path')
    parser.add_argument('--epochs',type=int,default=50,help='number of epochs to run')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    args = parser.parse_args()
    
    run_training(args.train_module, args.train_name,args.prune,args.s,args.ckpt,args.epochs,args.cudnn_benchmark)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
