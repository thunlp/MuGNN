import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TipleDataset(Dataset):
    def __init__(self, triples, nega_sapmle_num, cuda):
        self.triples = triples
        self.triple_set = set(triples)
        self.nega_sample_num = nega_sapmle_num
        self.relation_dict = {}
        for head, tail, relation in triples:
            if relation not in self.relation_dict:
                relation_dict[relation] = {'h':{head}, 't': {tail,}}
            else:
                relation_dict[relation]['h'].add(head)
                relation_dict[relation]['t'].add(tail)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, t, r = self.triples[idx]
        for i in range(self.nega_sample_num):
            pass

class AliagnmentDataset(Dataset):
    """Seed alignment dataset."""

    def __init__(self, seeds, nega_sample_num, num_sr, num_tg, cuda):
        self.cuda = cuda
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.nega_sample_num = nega_sample_num
        self.seeds = [[int(sr), int(tg)] for sr, tg in seeds]
        # self.init_negative_sample()

    def init_negative_sample(self):
        self.data = []
        nega_sample_num = self.nega_sample_num
        for seed in self.seeds:
            sr, tg = seed
            nega_sr = []
            nega_tg = []
            for _ in range(nega_sample_num):
                can_sr = random.randint(0, self.num_sr - 2)
                can_tg = random.randint(0, self.num_tg - 2)
                if can_sr >= sr:
                    can_sr += 1
                if can_tg >= tg:
                    can_tg += 1
                nega_sr.append(can_sr)
                nega_tg.append(can_tg)
            sr_data = [sr] + nega_sr + [sr] * nega_sample_num
            tg_data = [tg] + [tg] * nega_sample_num + nega_tg
            self.data.append((sr_data, tg_data))

    def __len__(self):
        return len(self.seeds)

    # def __getitem__(self, idx):
    #     sr_data, tg_data = self.data[idx]
    #     # the first data is the positive one
    #     sr_data = torch.tensor(sr_data, dtype=torch.int64)
    #     tg_data = torch.tensor(tg_data, dtype=torch.int64)
    #     return sr_data, tg_data

    def __getitem__(self, idx):
        nega_sample_num = self.nega_sample_num
        sr, tg = self.seeds[idx]
        # the first data is the positive one
        nega_sr = []
        nega_tg = []
        for _ in range(nega_sample_num):
            can_sr = random.randint(0, self.num_sr - 2)
            can_tg = random.randint(0, self.num_tg - 2)
            if can_sr >= sr:
                can_sr += 1
            if can_tg >= tg:
                can_tg += 1
            nega_sr.append(can_sr)
            nega_tg.append(can_tg)
        sr_data = [sr] + nega_sr + [sr] * nega_sample_num
        tg_data = [tg] + [tg] * nega_sample_num + nega_tg
        sr_data = torch.tensor(sr_data, dtype=torch.int64)
        tg_data = torch.tensor(tg_data, dtype=torch.int64)
        return sr_data, tg_data


if __name__ == '__main__':
    from project_path import bin_dir
    from data.reader import read_seeds, read_mapping
    directory = bin_dir / 'dbp15k' / 'fr_en'
    seeds = read_seeds(directory / 'entity_seeds.txt')
    entity_sr = read_mapping(directory / 'entity2id_fr.txt')
    entity_tg = read_mapping(directory / 'entity2id_en.txt')
    ad = AliagnmentDataset(seeds, 24,
                           len(entity_sr), len(entity_tg))
    ad_loader = DataLoader(ad, batch_size=1, shuffle=True, num_workers=4)
    for i, batch in enumerate(ad_loader):
        sr_data, tg_data = batch
        print(sr_data)
        print(tg_data)
        break