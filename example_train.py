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
config.init(load=True)
config.set_net(SpGATNet)
config.print_parameter()
config.train()
