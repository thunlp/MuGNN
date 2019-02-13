import platform
from pathlib import Path

__all__ = ['kg_data_dir']

node = platform.node()
kg_data_dir = Path()
if node == 'achar':
    kg_data_dir = Path(r'E:\knowledge_graph')
elif node[:-1] == 'next-gpu':
    kg_data_dir = Path('/storage/zyliu/Data/knowledge_graph')
else:
    raise NotImplementedError('Check the environment to make sure whether it was in the supported list.')
