import sys, os
from config import Config
from graph_completion.nets import *
from project_path import bin_dir

# CUDA_LAUNCH_BLOCKING=1

directory = bin_dir / 'dbp15k'
config = Config(directory)
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
except IndexError:
    config.set_cuda(False)
config.set_graph_completion(False)
config.init(load=False)
config.set_net(SpGATNet)
config.set_dim(128)
config.set_nheads(4)
config.set_num_layer(3)
config.set_learning_rate(0.001)
config.set_dropout(0.5)
config.set_gamma(3.0)
config.set_l2_penalty(0)
config.print_parameter()
config.train()
