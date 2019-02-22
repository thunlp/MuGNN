from project_path import bin_dir
from graph_completion.reader import read_mapping, read_seeds
# from graph_completion.functions import str2int4triples
from pathlib import Path

bin_dir = bin_dir / 'dbp15k'

mini_dbp = Path(r'/mnt/e/knowledge_graph/Mini_dbp15k')

def str2int4triples(triples):
    return [(int(head), int(tail)) for head, tail in triples]


directories = bin_dir.glob('*')
for directory in directories:
    local_source_dir = mini_dbp / directory.name
    sr, tg = directory.name.split('_')
    # seeds = str2int4triples(read_seeds(directory / 'entity_seeds.txt'))
    seeds = read_seeds(directory / 'entity_seeds.txt')
    sr_entity2id = read_mapping(directory / ('entity2id_%s.txt' % sr))
    tg_entity2id = read_mapping(directory / ('entity2id_%s.txt' % tg))
    sr_id2e = {i:entity for entity, i in sr_entity2id.items()}
    tg_id2e = {i: entity for entity, i in tg_entity2id.items()}
    seeds = [(sr_id2e[sr], tg_id2e[tg]) for sr, tg in seeds]
    seeds = ['\t'.join(seed) for seed in seeds]
    seeds_o = [line.strip() for line in open(local_source_dir / 'ent_ILLs', 'r').readlines()]
    assert len(seeds) == len(seeds_o)
    for i in range(len(seeds)):
        assert seeds[i] == seeds_o[i]
        print(seeds[i])
        print(seeds_o[i])
        break