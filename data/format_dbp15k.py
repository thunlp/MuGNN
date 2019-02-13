import shutil
import math
import random
from .kg_path import kg_data_dir
from .n2n import n2n
from pathlib import Path
from tools.print_time_info import print_time_info


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
        #  test_relation_seeds.txt JAPE中relation seed没有测试集
   OpenKE\
      en\
         entity2id.txt
         relation2id.txt
         train2id.txt
         valid2id.txt
         type_constrain.txt
      zh\
'''


def _dataset_split_validation_test(data2num, valid_ratio, test_ratio):
    '''
    1. test
    2. return num of data for each valid and test sets
    split rule: entity出现总数大于** 5 **的，可以分到valid里一个验证数据
    '''
    split_threshold = 5
    data2num_valid = {}
    data2num_test = {}
    data2num_train = {}
    count = 0
    count2 = 0
    for data, num in data2num.items():
        if num >= split_threshold:
            valid_num = math.ceil(num*valid_ratio)
            test_num = math.ceil(num*test_ratio)
        else:
            valid_num = 0
            test_num = 0

        data2num_valid[data] = valid_num
        data2num_test[data] = test_num
        if valid_num + test_num >= num:
            count += 1
            data2num_train[data] = 0
        else:
            data2num_train[data] = num - valid_num - test_num
    if count > 0:
        print_time_info('------------------------')
        print_time_info(count2)
        print_time_info(
            'Totally %d/%d of the data could not be split with the valid_ratio: %f, test ratio: %f.' % (count, len(data2num), valid_ratio, test_ratio))
    else:
        print_time_info('Check passed.')
    return data2num_train, data2num_valid, data2num_test


def _dump_mapping(file, file_name, bin_dir, write_num=True):
    with open(bin_dir / (file_name + '.txt'), 'w', encoding='utf8') as f:
        # print_time_info(file[:10])
        sorted_file = sorted(file.items(), key=lambda x: x[1])
        if write_num:
            f.write(str(len(sorted_file)) + '\n')
        for item, i in sorted_file:
            f.write(item + '\t' + str(i) + '\n')


def _dump_seeds(file, file_name, bin_dir, write_num=False):
    with open(bin_dir / (file_name + '_seeds.txt'), 'w', encoding='utf8') as f:
        if write_num:
            f.write(str(len(file)) + '\n')
        for seed_pair in file:
            f.write('\t'.join(str(i) for i in seed_pair) + '\n')


def _dump_triples(file, file_name, bin_dir, delimiter='\t', write_num=False):
    with open(bin_dir / (file_name + '.txt'), 'w', encoding='utf8') as f:
        if write_num:
            f.write(str(len(file)) + '\n')
        for head, tail, relation in file:
            f.write(str(head) + delimiter + str(tail) +
                    delimiter + str(relation) + '\n')


def _copy(from_name_list, to_name_list, from_bin_dir, to_bin_dir):
    from_path_list = [from_bin_dir / file_path for file_path in from_name_list]
    to_path_list = [to_bin_dir / file_path for file_path in to_name_list]
    for from_path, to_path in zip(from_path_list, to_path_list):
        shutil.copy(from_path, to_path)


def format_dbp15k(bin_dir, TransE_conf=None):
    '''
    TransE_conf: for split triple dataset
    '''
    if not TransE_conf:
        TransE_conf = {
            'valid_ratio': 0.1,
            'test_ratio': 0,
        }

    bin_dir = Path(bin_dir)
    if not bin_dir.exists():
        bin_dir.mkdir()
    bin_dir = bin_dir / 'dbp15k'
    if bin_dir.exists:
        import shutil
        shutil.rmtree(bin_dir)
        bin_dir.mkdir()

    def _format_seeds(mapping_sr, mapping_tg, bin_dir, directory, language):

        file2path = {'entity': 'ent_ILLS', 'relation': 'rel_ILLS'}
        type2seeds = {}
        for seed_type, path in file2path.items():
            file2id_sr = mapping_sr[seed_type]
            file2id_tg = mapping_tg[seed_type]
            with open(directory / path, 'r', encoding='utf8') as f:
                lines = f.readlines()
            seed_pairs = [line.strip().split('\t') for line in lines]
            seed_pairs = [(file2id_sr[seed_pair[0]], file2id_tg[seed_pair[1]])
                          for seed_pair in seed_pairs]
            _dump_seeds(seed_pairs, seed_type, bin_dir, write_num=False)
            type2seeds[seed_type] = seed_pairs
        return type2seeds

    def _format_single_language(directory, bin_dir, language):

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
        triples = [(entity2id[line[0]], entity2id[line[2]],
                    relation2id[line[1]]) for line in lines]
        _dump_mapping(entity2id, 'entity2id_' + language, bin_dir)
        _dump_mapping(relation2id, 'relation2id_' + language, bin_dir)
        _dump_triples(triples, 'triples_' + language, bin_dir)
        return {'entity': entity2id, 'relation': relation2id}, triples

    def _format_overall(directory, bin_dir, language_sr, language_tg='en'):
        '''
        sr: source
        tg: target
        '''
        if not bin_dir.exists():
            bin_dir.mkdir()
        mapping_sr, triples_sr = _format_single_language(
            directory, bin_dir, language_sr)
        mapping_tg, triples_tg = _format_single_language(
            directory, bin_dir, language_tg)
        type2seeds = _format_seeds(
            mapping_sr, mapping_tg, bin_dir, directory, language)

        return (mapping_sr, mapping_tg, triples_sr, triples_tg, type2seeds)

    def _format_amie(bin_dir, language2triples, language2mapping):
        local_bin_dir = bin_dir / 'AMIE'
        if not local_bin_dir.exists():
            local_bin_dir.mkdir()
        for language, triples in language2triples.items():
            entity2id = language2mapping[language]['entity']
            relation2id = language2mapping[language]['relation']
            all2id = {}
            for entity in entity2id:
                all2id[entity] = len(all2id)
            for relation in relation2id:
                all2id[relation] = len(all2id)

            _dump_mapping(all2id, 'all2id_' + language, local_bin_dir)
            id2entity = {i: entity for entity, i in entity2id.items()}
            id2relation = {i: relation for relation, i in relation2id.items()}
            triples = [(id2entity[head], id2relation[relation], id2entity[tail])
                       for head, tail, relation in triples]
            triples = [[all2id[item] for item in triple] for triple in triples]
            triples = [["<" + str(item) + ">" for item in triple]
                       for triple in triples]
            _dump_triples(triples, 'triples_' + language, local_bin_dir)

    _local_data_dir = kg_data_dir / 'dbp15k'
    language_pair_paths = list(_local_data_dir.glob('*_en'))
    language2dir = {path.name.split(
        '_')[0]: path for path in language_pair_paths}

    for language, directory in language2dir.items():
        local_bin_dir = bin_dir / (language + '_' + 'en')
        mapping_sr, mapping_tg, triples_sr, triples_tg, type2seeds = _format_overall(
            directory, local_bin_dir, language, 'en')
        _format_JAPE(directory, local_bin_dir, mapping_sr, mapping_tg)
        _format_OpenKE(directory, local_bin_dir, {
                       language: triples_sr, 'en': triples_tg}, TransE_conf['valid_ratio'], TransE_conf['test_ratio'])
        _format_amie(local_bin_dir, {language: triples_sr, 'en': triples_tg}, {
                     language: mapping_sr, 'en': mapping_tg})


def _format_OpenKE(directory, bin_dir, language2triples, valid_ratio, test_ratio):
    '''
    此处生成OpenKE的数据, 按照9:1划分训练和验证集
    OpenKE\
      en\
         entity2id.txt
         relation2id.txt
         train2id.txt
         valid2id.txt
         type_constrain.txt
      zh\
    '''

    def _split_dataset(triples, valid_ratio, test_ratio):
        '''
        random split
        force test data equals valid data if test ratio is 0
        force valid data equals 0.1 * train data is valid ratio is 0
        '''
        random_seed = 1.0
        triple_num = len(triples)
        valid_num = round(valid_ratio * triple_num)
        test_num = round(test_ratio * triple_num)
        random.seed(random_seed)
        random.shuffle(triples)
        test_data = triples[:test_num]
        valid_data = triples[test_num: test_num+valid_num]
        train_data = triples[test_num+valid_num:]
        if not valid_data:
            valid_data = train_data[round(0.9*len(train_data)):]
        if not test_data:
            test_data = valid_data
        return train_data, valid_data, test_data

    local_bin_dir = bin_dir / 'OpenKE'
    if not local_bin_dir.exists():
        local_bin_dir.mkdir()

    for language, triples in language2triples.items():
        language_bin_dir = local_bin_dir / language
        if not language_bin_dir.exists():
            language_bin_dir.mkdir()

        from_name_list = ['entity2id' + '_' + language +
                          '.txt', 'relation2id' + '_' + language + '.txt']
        to_name_list = ['entity2id.txt', 'relation2id.txt']
        _copy(from_name_list, to_name_list, bin_dir, language_bin_dir)
        train_data, valid_data, test_data = _split_dataset(
            triples, valid_ratio, test_ratio)
        _dump_triples(train_data, 'train2id', language_bin_dir,
                      delimiter=' ', write_num=True)
        _dump_triples(valid_data, 'valid2id', language_bin_dir,
                      delimiter=' ', write_num=True)
        _dump_triples(test_data, 'test2id', language_bin_dir,
                      delimiter=' ', write_num=True)
        n2n(language_bin_dir)


def _format_JAPE(directory, bin_dir, mapping_sr, mapping_tg):
    def _read_mapping(path):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip().split('\t') for line in lines]
            return dict(lines)

    def _get_local_mapping(directory):
        id2relation_sr = _read_mapping(directory / 'rel_ids_1')
        id2relation_tg = _read_mapping(directory / 'rel_ids_2')
        id2entity_sr = _read_mapping(directory / 'ent_ids_1')
        id2entity_tg = _read_mapping(directory / 'ent_ids_2')
        return {'entity': id2entity_sr, 'relation': id2relation_sr}, {'entity': id2entity_tg, 'relation': id2relation_tg}

    def _read_seeds(path, mapping_sr, mapping_tg):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip().split('\t') for line in lines]
            return [(mapping_sr[line[0]], mapping_tg[line[1]])
                    for line in lines]

    def _get_seeds(mapping_sr, mapping_tg, directory):
        train_entity_seeds = _read_seeds(
            directory / 'sup_ent_ids', mapping_sr['entity'], mapping_tg['entity'])
        test_entity_seeds = _read_seeds(
            directory / 'ref_ent_ids', mapping_sr['entity'], mapping_tg['entity'])
        train_relation_seeds = _read_seeds(
            directory / 'sup_rel_ids', mapping_sr['relation'], mapping_tg['relation'])
        return train_entity_seeds, test_entity_seeds, train_relation_seeds

    def _transform2id(seed_pairs, mapping_sr, mapping_tg):
        return [(mapping_sr[seed_pair[0]], mapping_tg[seed_pair[1]]) for seed_pair in seed_pairs]

    bin_dir = bin_dir / 'JAPE'
    if not bin_dir.exists():
        bin_dir.mkdir()

    percent2dir = list(directory.glob('0_*'))
    for directory in percent2dir:
        local_bin_dir = bin_dir / directory.name
        if not local_bin_dir.exists():
            local_bin_dir.mkdir()
        local_mapping_sr, local_mapping_tg = _get_local_mapping(directory)
        train_entity_seeds, test_entity_seeds, train_relation_seeds = _get_seeds(
            local_mapping_sr, local_mapping_tg, directory)
        train_entity_seeds = _transform2id(
            train_entity_seeds, mapping_sr['entity'], mapping_tg['entity'])
        test_entity_seeds = _transform2id(
            test_entity_seeds, mapping_sr['entity'], mapping_tg['entity'])
        train_relation_seeds = _transform2id(
            train_relation_seeds, mapping_sr['relation'], mapping_tg['relation'])
        _dump_seeds(train_entity_seeds, 'train_entity', local_bin_dir)
        _dump_seeds(test_entity_seeds, 'test_entity', local_bin_dir)
        _dump_seeds(train_relation_seeds, 'train_relation', local_bin_dir)


def main(bin_dir):
    TransE_conf = {
        'valid_ratio': 0.0,
        'test_ratio': 0.0,
    }
    format_dbp15k(bin_dir, TransE_conf)
    # Path()
