import random
import pickle
import torch
from tools.print_time_info import print_time_info
from torch.utils.data import Dataset, DataLoader

class TripleDataset(Dataset):
    def __init__(self, triples, nega_sapmle_num):
        self.triples = triples
        self.triple_set = set(triples)
        self.nega_sample_num = nega_sapmle_num
        h, t, r = list(zip(*triples))
        self.hs = list(set(h))
        self.ts = list(set(t))
        self.r2e = {}
        for head, tail, relation in triples:
            if relation not in self.r2e:
                self.r2e[relation] = {'h':{head,}, 't': {tail,}}
            else:
                self.r2e[relation]['h'].add(head)
                self.r2e[relation]['t'].add(tail)
        self.init()

    def init(self):
        r2e = self.r2e
        def exists(h, t, r):
            return (h, t, r) in self.triple_set
        def _init_one(h, t, r):
            nega_sample_num = self.nega_sample_num
            h_a = list({ele for ele in r2e[r]['h'] if ele != h})
            t_a = list({ele for ele in r2e[r]['t'] if ele != t})
            random.shuffle(h_a)
            random.shuffle(t_a)
            nega_h = [hh for i, hh in enumerate(h_a) if not exists(hh, t, r) and i < nega_sample_num]
            nega_t = [tt for i, tt in enumerate(t_a) if not exists(h, tt, r) and i < nega_sample_num]
            while len(nega_h) < nega_sample_num:
                hh = random.choice(self.hs)
                if not exists(hh, t, r):
                    nega_h.append(hh)
            while len(nega_t) < nega_sample_num:
                tt = random.choice(self.ts)
                if not exists(h, tt, r):
                    nega_t.append(tt)
            h_list = [h] + nega_h + [h] * nega_sample_num
            t_list = [t] + [t] * nega_sample_num + nega_t
            r_list = [r] * len(h_list)
            return h_list, t_list, r_list
        self.data = []
        for h, t, r in self.triples:
            self.data.append(_init_one(h, t, r))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print_time_info('Successfully saved triple dataset to %s.' % path)

    @classmethod
    def restore(self, path):
        with open(path, 'rb') as f:
            print_time_info('Successfully loaded triple dataset from %s.' % path)
            return pickle.load(f)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h_list, t_list, r_list = self.data[idx]
        h_list = torch.tensor(h_list, dtype=torch.int64)
        t_list = torch.tensor(t_list, dtype=torch.int64)
        r_list = torch.tensor(r_list, dtype=torch.int64)
        return h_list, t_list, r_list

    def get_all(self):
        h_list = []
        t_list = []
        r_list = []
        for h, t, r in self:
            h_list.append(h.view(1, -1))
            t_list.append(t.view(1, -1))
            r_list.append(r.view(1, -1))
        return torch.cat(h_list, dim=0), torch.cat(t_list, dim=0), torch.cat(r_list, dim=0)


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