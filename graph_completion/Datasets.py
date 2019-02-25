import random
import pickle
import torch
from tools.print_time_info import print_time_info
from torch.utils.data import Dataset, DataLoader


class EpochDataset(Dataset):
    def __init__(self, dataset, epoch=1000):
        super(EpochDataset, self).__init__()
        assert isinstance(dataset, Dataset)
        self.epoch = epoch
        self.dataset = dataset
        self.epoch_data = next(iter(DataLoader(self.dataset, batch_size=len(self.dataset))))

    def __len__(self):
        return self.epoch

    def __getitem__(self, idx):
        return self.epoch_data

    def get_data(self):
        data = [d.squeeze(0) for d in self[0]]
        return data


class TripleDataset(Dataset):
    def __init__(self, triples, nega_sapmle_num, corruput=False):
        self.corruput = corruput
        self.triples = triples
        self.triple_set = set(triples)
        self.nega_sample_num = nega_sapmle_num
        h, t, r = list(zip(*triples))
        self.hs = list(set(h))
        self.ts = list(set(t))
        self.r2e = {}
        for head, tail, relation in triples:
            if relation not in self.r2e:
                self.r2e[relation] = {'h': {head, }, 't': {tail, }}
            else:
                self.r2e[relation]['h'].add(head)
                self.r2e[relation]['t'].add(tail)
        self.postive_data = [[], [], []]
        self.negative_data = [[], [], []]
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
            h_list = nega_h + [h] * nega_sample_num
            t_list = [t] * nega_sample_num + nega_t
            return h_list, t_list

        for h, t, r in self.triples:
            nega_h, nega_t = _init_one(h, t, r)
            self.negative_data[0] += nega_h
            self.negative_data[1] += nega_t
            self.negative_data[2] += [r] * len(nega_h)
            self.postive_data[0] += [h] * len(nega_h)
            self.postive_data[1] += [t] * len(nega_h)
            self.postive_data[2] += [r] * len(nega_h)

        if self.corruput:
            z = list(zip(*self.negative_data))
            random.shuffle(z)
            self.negative_data = list(zip(*z))

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
        pos_h = self.postive_data[0][idx]
        pos_t = self.postive_data[1][idx]
        pos_r = self.postive_data[2][idx]
        neg_h = self.negative_data[0][idx]
        neg_t = self.negative_data[1][idx]
        neg_r = self.negative_data[2][idx]
        h_list = torch.tensor([pos_h, neg_h], dtype=torch.int64)
        t_list = torch.tensor([pos_t, neg_t], dtype=torch.int64)
        r_list = torch.tensor([pos_r, neg_r], dtype=torch.int64)
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

    def __init__(self, seeds, nega_sample_num, num_sr, num_tg, cuda, corruput=False):
        self.cuda = cuda
        self.corruput = corruput
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.nega_sample_num = nega_sample_num
        self.seeds = [[int(sr), int(tg)] for sr, tg in seeds]
        self.positive_data = [[], []]
        self.negative_data = [[], []]
        self.init()

    def get_seeds(self):
        sr, tg = list(zip(*self.seeds))
        return torch.tensor(sr), torch.tensor(tg)

    def update_negative_sample(self, sr_nega, tg_nega):
        nega_sr = []
        nega_tg = []
        for i, seed in enumerate(self.seeds):
            sr, tg = seed
            # assert sr == self.positive_data[0][i * self.nega_sample_num + 1]
            # assert tg == self.positive_data[1][i * self.nega_sample_num + 1]
            # assert len(set(sr_nega[sr])) == self.nega_sample_num
            # assert len(sr_nega[sr]) == self.nega_sample_num
            # assert len(set(tg_nega[tg])) == self.nega_sample_num
            # assert len(tg_nega[tg]) == self.nega_sample_num
            nega_sr += sr_nega[sr]
            nega_tg += tg_nega[tg]
        # print('The length of negative data:', len(nega_sr), len(nega_tg))
        # print('The length of positive data:', len(pos_sr), len(pos_tg))
        self.negative_data = [nega_sr, nega_tg]

    def init(self):
        nega_sample_num = self.nega_sample_num
        for seed in self.seeds:
            sr, tg = seed
            nega_sr = []
            nega_tg = []
            can_srs = random.choices(range(0, self.num_sr - 1), k=nega_sample_num)
            can_tgs = random.choices(range(0, self.num_tg - 1), k=nega_sample_num)
            for can_sr, can_tg in zip(can_srs, can_tgs):
                if can_sr >= sr:
                    can_sr += 1
                if can_tg >= tg:
                    can_tg += 1
                nega_sr.append(can_sr)
                nega_tg.append(can_tg)
            self.negative_data[0] += nega_sr
            self.negative_data[1] += nega_tg
            self.positive_data[0] += [sr] * nega_sample_num
            self.positive_data[1] += [tg] * nega_sample_num

        if self.corruput:
            z = list(zip(*self.negative_data))
            random.shuffle(z)
            self.negative_data = list(zip(*z))

    def __len__(self):
        return len(self.positive_data[0])

    def __getitem__(self, idx):
        pos_sr = self.positive_data[0][idx]
        pos_tg = self.positive_data[1][idx]
        neg_sr = self.negative_data[0][idx]
        neg_tg = self.negative_data[1][idx]
        # the first data is the positive one
        sr_data = torch.tensor([pos_sr, neg_sr], dtype=torch.int64)
        tg_data = torch.tensor([pos_tg, neg_tg], dtype=torch.int64)
        return sr_data, tg_data

    # def __getitem__(self, idx):
    #     nega_sample_num = self.nega_sample_num
    #     sr, tg = self.seeds[idx]
    #     # the first data is the positive one
    #     nega_sr = []
    #     nega_tg = []
    #     for _ in range(nega_sample_num):
    #         can_sr = random.randint(0, self.num_sr - 2)
    #         can_tg = random.randint(0, self.num_tg - 2)
    #         if can_sr >= sr:
    #             can_sr += 1
    #         if can_tg >= tg:
    #             can_tg += 1
    #         nega_sr.append(can_sr)
    #         nega_tg.append(can_tg)
    #     sr_data = [sr] + nega_sr + [sr] * nega_sample_num
    #     tg_data = [tg] + [tg] * nega_sample_num + nega_tg
    #     sr_data = torch.tensor(sr_data, dtype=torch.int64)
    #     tg_data = torch.tensor(tg_data, dtype=torch.int64)
    #     return sr_data, tg_data


if __name__ == '__main__':
    pass
