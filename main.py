import argparse
from solver import Solver
import yaml

def parse():
    parser = argparse.ArgumentParser(description="Pytorch-Leo")
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-verbose', action='store_true', help='whether output info')
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')

    parser.add_argument('-seed', default='100', type=int)

    # loading model path
    parser.add_argument('-load', default='./train_model/model.pth', help= 'load: model_dir')


    #training config
    # parser.add_argument('-batch_size', type=int, default=12, help='batch size')
    # parser.add_argument('-val_batch_size', type=int, default=200, help='meta valid batch size')
    # parser.add_argument('-test_batch_size', type=int, default=200, help='meta test batch size')

    # parser.add_argument('-num_steps', default = 100000, type = int)
    
    # parser.add_argument('-meta_val_steps', default=15, type=int)
    # parser.add_argument("-inner_update_step", default=5, type=int)
    # parser.add_argument("-finetuning_update_step", default=5, type=int)
    # parser.add_argument("-clip_value", default=0.1, type=float)

    parser.add_argument('-valid_every_step', default=5000, type=int)
    parser.add_argument('-print_every_step', default=250, type=int)
    parser.add_argument('-total_val_steps', default=100, type=int, help='how many batches sampled during validation')
    parser.add_argument('-total_test_instances', default=10000, type=int, help='how many instances sampled when testing')

    parser.add_argument('-N', default=5, type=int, help='N way classificatoin')
    parser.add_argument('-K', default=1, type=int, help='K shot')
    
    parser.add_argument('-no_cuda', action='store_true')

    # parser.add_argument("-inner_lr_init", default=1.0, type=float)
    # parser.add_argument("-finetuning_lr_init", default=0.001, type=float)
    # parser.add_argument("-outer_lr", default=0.0001, type=float)
    # parser.add_argument("-kl_weight", default=0.001, type=float)
    # parser.add_argument("-encoder_penalty_weight", default=1e-9, type=float)
    # parser.add_argument("-l2_penalty_weight", default=1e-8, type=float)
    # parser.add_argument("-orthogonality_penalty_weight", default=1e-3, type=float)
    
    

    #network config
    # parser.add_argument("-dropout", default=0.3, type=float)
    # parser.add_argument("-embed_size", default=640, type=int)
    # parser.add_argument("-hidden_size", default=64, type=int)

    # data
    # parser.add_argument("-corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/train_weibo_seq2seq_1.txt", type = str)
    # parser.add_argument("-test_corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/test_weibo_seq2seq_1.txt", type = str)
    # parser.add_argument("-valid_corpus", default = "../multi-response/data/weibo_utf8/seq2seq_data/valid_weibo_seq2seq_1.txt", type = str)
    parser.add_argument("-embedding_dir", default='../embeddings/', type=str, help='directory storing pretrained embeddings')
    parser.add_argument("-dataset", default='miniImageNet', type=str, help='miniImageNet or tieredImageNet')
    parser.add_argument("-pretraining_scheme", default='center', type=str, help='center or view')

    parser.add_argument("-exp_name", type = str, help='help id experiment, model save at model_dir/dataset_Kshot/exp_name')
    parser.add_argument('-disable_comet', action='store_true')
    parser.add_argument('-save_checkpoint', action='store_true', help='whether to save checkpoints')
    parser.add_argument('-save_best', action='store_true', help='if true, only save the best model (highest valid acc)')
    parser.add_argument("-config", default='config.yml', help='config file')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
    config = config[args.dataset]["%dshot"%args.K]

    solver = Solver(args, config)
    
    if args.train:
        solver.train()
    elif args.test:
        solver.test()