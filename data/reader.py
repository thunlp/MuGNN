def read_mapping(path):
    with open(path, 'r', encoding='utf8') as f:
        line = f.readline().strip()
        if line.isdigit():
            num = int(line)
        else:
            f.seek(0)
        lines = f.readlines()
    item2id = [line.strip().split('\t') for line in lines]
    return dict(item2id) 