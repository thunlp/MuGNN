import json
from tools.print_time_info import print_time_info


def read_mapping(path):
    def _parser(lines):
        for idx, line in enumerate(lines):
            name, i = line.strip().split('\t')
            lines[idx] = (name, int(i))
        return dict(lines)
    # lambda lines: dict([line.strip().split('\t') for line in lines])
    return read_file(path, _parser)


def read_triples(path):
    '''
    triple pattern: (head_id, tail_id, relation_id)
    '''
    return read_file(path, lambda lines: [tuple([int(item) for item in line.strip().split('\t')]) for line in lines])


def read_seeds(path):
    return read_file(path, lambda lines: [tuple([int(item) for item in line.strip().split('\t')]) for line in lines])


def read_rules(path, relation2id):
    def _read_rules(lines):
        lines = [json.loads(line) for line in lines]
        for i in range(len(lines)):
            premises, hypothesis, conf = lines[i]
            premises = tuple([tuple([head, tail, relation2id[relation]])
                              for head, tail, relation in premises])
            hypothesis = tuple(
                [hypothesis[0], hypothesis[1], relation2id[hypothesis[2]]])
            lines[i] = (premises, hypothesis, float(conf))
        return lines

    return read_file(path, _read_rules)


def read_file(path, parse_func):
    num = -1
    with open(path, 'r', encoding='utf8') as f:
        line = f.readline().strip()
        if line.isdigit():
            num = int(line)
        else:
            f.seek(0)
        lines = f.readlines()

    lines = parse_func(lines)

    if len(lines) != num and num >= 0:
        print_time_info('File: %s has corruptted, data_num: %d/%d.' %
                        (path, num, len(lines)))
        raise ValueError()
    return lines
