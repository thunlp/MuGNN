import time
from rule.rule_mining import mine_rule_with_amie, parse_and_dump_rules
from data import reader

def mine_rule_for_dbp15k():
    from project_path import bin_dir
    bin_dir = bin_dir / 'dbp15k'
    language_pair_dirs = list(bin_dir.glob('*_en'))
    for directory in language_pair_dirs:
        local_bin_dir = directory / 'AMIE'
        file_paths = local_bin_dir.glob('triples_*.txt')
        for file_path in file_paths:
            file_name = file_path.name
            output_path = local_bin_dir / ('rule_for_' + file_name)
            mine_rule_with_amie(file_path, output_path)
    time.sleep(120)
    for directory in language_pair_dirs:
        local_bin_dir = directory / 'AMIE'
        file_paths = local_bin_dir.glob('rule_for_triples_*.txt')
        for file_path in file_paths:
            language = file_path.name.split('_')[-1][:2]
            all2id = reader.read_mapping(local_bin_dir / 'all2id_' + language + '.txt')
            parse_and_dump_rules(file_path, file_path, {i: item for item, i in all2id.items()})

            
def main():
    from project_path import bin_dir
    # path = bin_dir /
    mine_rule_for_dbp15k()
    # rule_parser(r'E:\文档\code\EAbyRule\bin\dbp15k\ja_en\AMIE\rule_for_triples_en.txt')



if __name__ == '__main__':
    main()