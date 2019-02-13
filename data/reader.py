import json
from tools.print_time_info import print_time_info


def read_mapping(path):
    return read_file(path, lambda lines: dict([line.strip().split('\t') for line in lines]))


def read_triples(path):
    return read_file(path, lambda lines: [tuple(line.strip().split('\t')) for line in lines])


def read_seeds(path):
    return read_file(path, lambda lines: [tuple(line.strip().split('\t')) for line in lines])


def read_rules(path, relation2id):
    def _read_rules(lines):
        lines = [json.loads(line) for line in lines]
        for i in range(len(lines)):
            premises, hypothesis, conf = lines[i]
            premises = [tuple(head, tail, relation2id[relation])
                        for head, tail, relation in premises]
            hypothesis = tuple(
                hypothesis[0], hypothesis[1], relation2id[hypothesis[2]])
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
