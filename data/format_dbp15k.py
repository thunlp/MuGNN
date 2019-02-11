from .kg_path import kg_data_dir
from pathlib import Path

'''
root\
   entity2id_en.txt
   entity2id_zh.txt
   relation2id_en.txt
   relation2id_zh.txt
   triples_zh.txt
   triples_en.txt
   entity_seeds.txt
   relation_seeds.txt
   JAPE\
      0_3\
         train_entity_seeds.txt
         train_relation_seeds.txt
         test_entity_seeds.txt
         test_relation_seeds.txt
   OpenKE\
      en\
         entity2id.txt
         relation2id.txt
         train2id.txt
         valid2id.txt
         type_constrain.txt
      zh\
'''


def format_dbp15k(bin_dir):
    bin_dir = Path(bin_dir)
    if not bin_dir.exists():
        bin_dir.mkdir()
    bin_dir = bin_dir / 'dbp15k'
    if bin_dir.exists:
        import shutil
        shutil.rmtree(bin_dir)
        bin_dir.mkdir()

    def _format_seeds(mapping_sr, mapping_tg, bin_dir, directory, language):
        def _dump_seed(file, file_name, bin_dir):
            with open(bin_dir / (file_name + '_seeds.txt'), 'w', encoding='utf8') as f:
                f.write(str(len(file)) + '\n')
                for seed_pair in file:
                    f.write(' '.join(str(i) for i in seed_pair) + '\n')

        file2path = {'entity': 'ent_ILLS', 'relation': 'rel_ILLS'}
        for seed_type, path in file2path.items():
            file2id_sr = mapping_sr[seed_type]
            file2id_tg = mapping_tg[seed_type]
            with open(directory / path, 'r', encoding='utf8') as f:
                lines = f.readlines()
            seed_pairs = [line.strip().split('\t') for line in lines]
            seed_pairs = [(file2id_sr[seed_pair[0]], file2id_tg[seed_pair[1]])
                          for seed_pair in seed_pairs]
            _dump_seed(seed_pairs, seed_type, bin_dir)

    def _format_single_language(directory, bin_dir, language):
        def _dump_mapping(file, file_name, bin_dir, language):
            with open(bin_dir / (file_name + '_' + language + '.txt'), 'w', encoding='utf8') as f:
                # print(file[:10])

                sorted_file = sorted(file.items(), key=lambda x: x[1])
                f.write(str(len(sorted_file)) + '\n')
                for item, i in sorted_file:
                    f.write(item + '\t' + str(i) + '\n')

        def _dump_triples(file, file_name, bin_dir, language):
            with open(bin_dir / (file_name + '_' + language + '.txt'), 'w', encoding='utf8') as f:
                f.write(str(len(file)) + '\n')
                for head, relation, tail in file:
                    f.write(str(head) + ' ' + str(tail) +
                            ' ' + str(relation) + '\n')

        entities = set()
        relations = set()
        if language == 'en':
            triples_path = 't_triples'
        else:
            triples_path = 's_triples'
        with open(directory / triples_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip().split('\t') for line in lines]
            for line in lines:
                entities.add(line[0])
                entities.add(line[2])
                relations.add(line[1])
        entity2id = {entity: i for i, entity in enumerate(entities)}
        relation2id = {relation: i for i,
                       relation in enumerate(relations)}
        triples = [(entity2id[line[0]], relation2id[line[1]],
                    entity2id[line[2]]) for line in lines]

        _dump_mapping(entity2id, 'entity2id', bin_dir, language)
        _dump_mapping(relation2id, 'relation2id', bin_dir, language)
        _dump_triples(triples, 'triples', bin_dir, language)
        return {'entity': entity2id, 'relation': relation2id}

    def _format_overall(directory, bin_dir, language_sr, language_tg='en'):
        '''
        sr: source
        tg: target
        '''
        bin_dir = bin_dir / (language_sr + '_' + language_tg)
        if not bin_dir.exists():
            bin_dir.mkdir()
        mapping_sr = _format_single_language(
            directory, bin_dir, language_sr)
        mapping_tg = _format_single_language(
            directory, bin_dir, language_tg)
        _format_seeds(mapping_sr, mapping_tg, bin_dir, directory, language)

    _local_data_dir = kg_data_dir / 'dbp15k'
    language_pair_paths = list(_local_data_dir.glob('*_en'))
    language2dir = {path.name.split(
        '_')[0]: path for path in language_pair_paths}
    for language, directory in language2dir.items():
        _format_overall(directory, bin_dir, language, 'en')


def main(bin_dir):
    format_dbp15k(bin_dir)
