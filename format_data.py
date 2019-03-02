from project_path import bin_dir
from graph_completion import reader
from data.format_dbp15k import format_dbp15k_full, format_dbp15k
from graph_completion.rule_mining import mine_rule_with_amie, parse_and_dump_rules
from data.format_dwy100k import format_dwy100k

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


if __name__ == '__main__':
    
    TransE_conf = {
        'valid_ratio': 0.0,
        'test_ratio': 0.0,
    }
    # format_dwy100k()
    # exit()
    # print(bin_dir.parent)
    # exit()
    # format_dbp15k(bin_dir, 'dbp15k', 'dbp15k')
    mine_rule_for_dbp15k('DWY100k')

    # format_dbp15k_full(bin_dir, 'full_dbp15k', 'Full_DBP15k')
    # mine_rule_for_dbp15k('full_dbp15k')