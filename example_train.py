import sys, os
from config import Config
from graph_completion.nets import *
from project_path import bin_dir
from torch.optim import Adagrad, SGD, Adam

# CUDA_LAUNCH_BLOCKING=1


config = Config()
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
except IndexError:
    config.set_cuda(False)


config.set_dim(128)
config.set_nheads(1)
config.set_align_gamma(1.0)
config.set_rule_gamma(0.12)
config.set_rel_align_gamma(1.0)
config.set_batch_size(4500)
config.set_num_layer(2)
config.set_dropout(0.2)
config.set_num_workers(4)
config.set_learning_rate(0.001)
config.set_l2_penalty(1e-2)
config.set_optimizer(Adagrad)
config.set_beta(0.5)
config.set_sparse(True)
config.set_shuffle(True)
config.set_update_cycle(5)
config.set_w_adj('rel_adj')
config.set_rule_scale(1.0)
config.set_pre_train(0)
config.set_rule_infer(True)
config.set_bootstrap(False)
config.set_train_big(True)
config.set_train_seed_ratio(0.3)
config.set_rule_transfer(False)
config.set_graph_completion(True)

# directory = bin_dir / 'DWY100k'
directory = bin_dir / 'dbp15k'
config.init(directory, 'dbp_yg', load=False)
config.set_net()
config.print_parameter()
config.init_log('dbp_yg:wo rel loss')
config.train()

# fr_en 1st
# zh_en 0.2 runned
# relation attention options: 'adj', 'rel_adj', 'caw'