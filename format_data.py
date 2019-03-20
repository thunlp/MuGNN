import sys
from utils import reader
from graph_completion.rule_mining import mine_rule_with_amie, parse_and_dump_rules

def mine_rule_for_dbp15k(path_name):
    from project_path import bin_dir
    bin_dir = bin_dir / path_name
    language_pair_dirs = list(bin_dir.glob('*'))
    for directory in language_pair_dirs:
        local_bin_dir = directory / 'AMIE'
        file_paths = local_bin_dir.glob('triples_*.txt')
        for file_path in file_paths:
            file_name = file_path.name
            output_path = local_bin_dir / ('rule_for_' + file_name)
            mine_rule_with_amie(file_path, output_path)
    while True:
        info = input('\tIf amie ended, please input: "amie ended"\n')
        if info == 'amie ended':
            break
    for directory in language_pair_dirs:
        local_bin_dir = directory / 'AMIE'
        file_paths = local_bin_dir.glob('rule_for_triples_*.txt')
        for file_path in file_paths:
            language = file_path.name.split('_')[-1].split('.')[0]
            all2id = reader.read_mapping(local_bin_dir / ('all2id_' + language + '.txt'))
            parse_and_dump_rules(file_path, file_path, {i: item for item, i in all2id.items()})

def rule_mining_for_single_dataset(dataset_path):
    from pathlib import Path
    dataset_path = Path(dataset_path)
    local_bin_dir = dataset_path / 'AMIE'
    file_paths = local_bin_dir.glob('triples_*.txt')
    for file_path in file_paths:
        file_name = file_path.name
        output_path = local_bin_dir / ('rule_for_' + file_name)
        mine_rule_with_amie(file_path, output_path)
    while True:
        info = input('\tIf amie ended, please input: "amie ended"\n')
        if info == 'amie ended':
            break
    file_paths = local_bin_dir.glob('rule_for_triples_*.txt')
    for file_path in file_paths:
        language = file_path.name.split('_')[-1].split('.')[0]
        all2id = reader.read_mapping(local_bin_dir / ('all2id_' + language + '.txt'))
        parse_and_dump_rules(file_path, file_path, {i: item for item, i in all2id.items()})

if __name__ == '__main__':
    dataset_path = sys.argv[1]    
    rule_mining_for_single_dataset(dataset_path)
    