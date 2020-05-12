import argparse
from solver import Solver


def parse():
    parser = argparse.ArgumentParser(description="Pytorch-Leo")
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-verbose', action='store_true', help='whether output info')
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument("-exp_name", default='example', type = str, help='help id experiment, model save at model_dir/dataset_Kshot/exp_name')
    
    parser.add_argument('-seed', default='100', type=int)

    # loading model path
    parser.add_argument('-load', default='./train_model/model.pth', help= 'load: model_dir')
    parser.add_argument('-save_checkpoint', action='store_true', help='whether to save checkpoints')
    parser.add_argument('-save_best', action='store_true', help='if true, only save the best model (highest valid acc)')

    parser.add_argument('-valid_every_step', default=5000, type=int)
    parser.add_argument('-print_every_step', default=250, type=int)
    parser.add_argument('-total_val_steps', default=100, type=int, help='how many batches sampled during validation')
    parser.add_argument('-total_test_instances', default=10000, type=int, help='how many instances sampled when testing')

    parser.add_argument('-N', default=5, type=int, help='N way classificatoin')
    parser.add_argument('-K', default=1, type=int, help='K shot')
    
    parser.add_argument('-no_cuda', action='store_true')

    parser.add_argument("-embedding_dir", default='../embeddings/', type=str, help='directory storing pretrained embeddings')
    parser.add_argument("-dataset", default='miniImageNet', type=str, help='miniImageNet or tieredImageNet')
    parser.add_argument("-pretraining_scheme", default='center', type=str, help='center or view')
    
    parser.add_argument('-disable_comet', action='store_true')
    
    parser.add_argument("-config", default='config.yml', help='config file')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver.test()