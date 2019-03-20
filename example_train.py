import sys, os
from config import Config
from torch.optim import Adagrad

config = Config()
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
except IndexError:
    config.set_cuda(False)

config.set_dim(128)
config.set_align_gamma(1.0)
config.set_rule_gamma(0.12)
config.set_num_layer(2)
config.set_dropout(0.2)
config.set_learning_rate(0.001)
config.set_l2_penalty(1e-2)
config.set_update_cycle(5)
config.set_optimizer(Adagrad)
config.set_train_seed_ratio(0.3)
config.set_w_adj('rel_adj')

config.set_rule_infer(True)
config.set_rule_transfer(True)
config.set_graph_completion(True)

config.init('./bin/DBP15k/zh_en', True)
config.set_net()
config.print_parameter()
config.init_log('./log/test')
config.train()