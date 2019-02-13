import json
from tools.print_time_info import print_time_info


def read_mapping(path):
    return read_file(path, lambda lines: dict([line.strip().split('\t') for line in lines]))


def read_triples(path):
    return read_file(path, lambda lines: [tuple(line.strip().split('\t')) for line in lines])


def read_seeds(path):
    return read_file(path, lambda lines: [tuple(line.strip().split('\t')) for line in lines])


def read_rules(path):
    return read_file(path, lambda lines: [[[tuple(pp) for pp in p] for p in json.loads(line)] for line in lines])


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
