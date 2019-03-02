import random
import torch
from tools.print_time_info import print_time_info
from tools.timeit import timeit
from torch.utils.data import Dataset, DataLoader
from graph_completion.CrossGraphCompletion import CrossGraphCompletion


class EpochDataset(object):
    def __init__(self, dataset, batch_num=1):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.batch_num = batch_num
        data_num = len(dataset)
        if data_num % batch_num == 0:
            batch_size = data_num // batch_num
        else:
            batch_size = data_num // batch_num + 1
        self.batch_size = batch_size
        self.data = [data for data in self.get_data_loader()]
        assert len(self.data) == len(self)

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def get_data(self):
        return next(iter(DataLoader(self.dataset, batch_size=len(self.dataset))))


class RuleDataset(Dataset):
    def __init__(self, cgc, data_name, triples, relations, nega_sample_num):
        self.triples = set(triples)
        assert len(self.triples) == len(triples)
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.data_name = data_name
        self.premise_pad = len(self.triples)
        print_time_info('premise pad number: %d' % self.premise_pad)
        self.nega_sample_num = nega_sample_num
        self.relations = relations
        self.h = []
        self.t = []
        self.pos_r = []
        self.neg_r = []
        self.premises = []

        self.check_p = -100
        self.init()

    @property
    def new_triple_premises(self):
        data = getattr(self.cgc, self.data_name)
        if self.check_p < 0:
            self.check_p = len(data)
        else:
            print('boot check in Rule dataset: ', self.check_p, len(data))
        return data

    def init(self):
        triples = self.triples
        relations = self.relations
        premise_pad = self.premise_pad
        nega_sample_num = self.nega_sample_num
        self.h = []
        self.t = []
        self.pos_r = []
        self.neg_r = []
        self.premises = []
        for new_triple, premises in self.new_triple_premises.items():
            h, t, r = new_triple
            neg_rs = random.sample(relations, k=nega_sample_num)
            neg_rs = [neg_r for neg_r in neg_rs if (h, t, neg_r) not in triples]
            while len(neg_rs) < nega_sample_num:
                neg_r = random.choice(relations)
                if (h, t, neg_r) not in triples:
                    neg_rs.append(neg_r)
            self.neg_r += neg_rs
            self.pos_r += [r] * nega_sample_num
            self.h += [h] * nega_sample_num
            self.t += [t] * nega_sample_num
            if len(premises) == 1:
                premises.append(premise_pad)
            assert len(premises) == 2
            premises = [premise * 2 * nega_sample_num for premise in premises]  # for rule & transe alignment
            self.premises += [premises] * nega_sample_num  #
        assert len(self.h) == len(self.t) == len(self.pos_r) == len(self.neg_r) == len(self.premises)
        return self

    def __len__(self):
        return len(self.h)

    def __getitem__(self, idx):
        h = torch.tensor([self.h[idx]], dtype=torch.int64)
        t = torch.tensor([self.t[idx]], dtype=torch.int64)
        r = torch.tensor([self.pos_r[idx], self.neg_r[idx]], dtype=torch.int64)
        premise = torch.tensor(self.premises[idx], dtype=torch.int64)
        return h, t, r, premise

    def get_all(self):
        h_all = torch.tensor(self.h, dtype=torch.int64).view(-1, 1)
        t_all = torch.tensor(self.t, dtype=torch.int64).view(-1, 1)
        r_all = torch.tensor(list(zip(self.pos_r, self.neg_r)), dtype=torch.int64)
        premise_all = torch.tensor(self.premises, dtype=torch.int64)
        return h_all, t_all, r_all, premise_all


class BatchRuleDataset(RuleDataset):
    def __init__(self, batch_num, cgc, data_name, triples, relations, nega_sample_num):
        super(BatchRuleDataset, self).__init__(cgc, data_name, triples, relations, nega_sample_num)
        self.batch_num = batch_num
        if len(self.h) % batch_num == 0:
            self.batch_size = len(self.h) // batch_num
        else:
            self.batch_size = len(self.h) // batch_num + 1
        assert self.batch_num * self.batch_size >= len(self.h)
        assert (self.batch_num - 1) * self.batch_size < len(self.h)

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        h = torch.tensor(self.h[start:end], dtype=torch.int64).view(-1, 1)
        t = torch.tensor(self.t[start:end], dtype=torch.int64).view(-1, 1)
        r = torch.tensor(list(zip(self.pos_r[start:end], self.neg_r[start:end])), dtype=torch.int64)
        premise = torch.tensor(self.premises[start:end], dtype=torch.int64)
        return h, t, r, premise


class TripleDataset(Dataset):
    def __init__(self, triples, nega_sapmle_num):
        self.triples = triples
        self.triple_set = set(triples)
        self.nega_sample_num = nega_sapmle_num
        h, t, r = list(zip(*triples))
        self.hs = list(set(h))
        self.ts = list(set(t))
        r2e = {}
        for head, tail, relation in triples:
            if relation not in r2e:
                r2e[relation] = {'h': {head, }, 't': {tail, }}
            else:
                r2e[relation]['h'].add(head)
                r2e[relation]['t'].add(tail)
        self.r2e = {r: {k: list(box) for k, box in es.items()} for r, es in r2e.items()}
        self.postive_data = [[], [], []]
        self.negative_data = [[], [], []]
        self.init()

    def init(self):
        r2e = self.r2e
        nega_sample_num = self.nega_sample_num

        def exists(h, t, r):
            return (h, t, r) in self.triple_set

        def _init_one(h, t, r):
            h_a = r2e[r]['h']
            t_a = r2e[r]['t']
            nega_h = random.sample(h_a, min(nega_sample_num + 1, len(h_a)))
            nega_t = random.sample(t_a, min(nega_sample_num + 1, len(t_a)))
            nega_h = [hh for hh in nega_h if not exists(hh, t, r)][:nega_sample_num]
            nega_t = [tt for tt in nega_t if not exists(h, tt, r)][:nega_sample_num]
            while len(nega_h) < nega_sample_num:
                hh = random.choice(self.hs)
                if not exists(hh, t, r):
                    nega_h.append(hh)
            while len(nega_t) < nega_sample_num:
                tt = random.choice(self.ts)
                if not exists(h, tt, r):
                    nega_t.append(tt)
            nega_h = nega_h + len(nega_h) * [h]
            nega_t = len(nega_t) * [t] + nega_t
            return nega_h, nega_t

        self.postive_data = [[], [], []]
        self.negative_data = [[], [], []]
        for h, t, r in self.triples:
            nega_h, nega_t = _init_one(h, t, r)
            self.negative_data[0] += nega_h
            self.negative_data[1] += nega_t
            self.negative_data[2] += [r] * len(nega_h)
            self.postive_data[0] += [h] * len(nega_h)
            self.postive_data[1] += [t] * len(nega_h)
            self.postive_data[2] += [r] * len(nega_h)
        return self

    def __len__(self):
        return len(self.postive_data[0])

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
        h_all = torch.tensor(list(zip(self.postive_data[0], self.negative_data[0])), dtype=torch.int64)
        t_all = torch.tensor(list(zip(self.postive_data[1], self.negative_data[1])), dtype=torch.int64)
        r_all = torch.tensor(list(zip(self.postive_data[2], self.negative_data[2])), dtype=torch.int64)
        return h_all, t_all, r_all


class BatchTripleDataset(TripleDataset):
    def __init__(self, batch_num, triples, nega_sapmle_num):
        super(BatchTripleDataset, self).__init__(triples, nega_sapmle_num)
        self.batch_num = batch_num
        if len(self.postive_data[0]) % batch_num == 0:
            self.batch_size = len(self.postive_data[0]) // batch_num
        else:
            self.batch_size = len(self.postive_data[0]) // batch_num + 1
        assert self.batch_num * self.batch_size >= len(self.postive_data[0])
        assert (self.batch_num - 1) * self.batch_size < len(self.postive_data[0])

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        h = torch.tensor(list(zip(self.postive_data[0][start:end], self.negative_data[0][start:end])),
                         dtype=torch.int64)
        t = torch.tensor(list(zip(self.postive_data[1][start:end], self.negative_data[1][start:end])),
                         dtype=torch.int64)
        r = torch.tensor(list(zip(self.postive_data[2][start:end], self.negative_data[2][start:end])),
                         dtype=torch.int64)
        return h, t, r


class AliagnmentDataset(Dataset):
    """Seed alignment dataset."""

    def __init__(self, cgc, data_name, nega_sample_num, num_sr, num_tg, cuda):
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.data_name = data_name
        self.cuda = cuda
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.nega_sample_num = nega_sample_num
        self.positive_data = [[], []]
        self.negative_data = [[], []]
        self.init()

    @property
    def seeds(self):
        seeds = [[int(sr), int(tg)] for sr, tg in getattr(self.cgc, self.data_name)]
        return seeds

    def get_seeds(self):
        sr, tg = list(zip(*self.seeds))
        return torch.tensor(sr), torch.tensor(tg)

    def update_negative_sample(self, sr_nega, tg_nega):
        nega_sr = []
        nega_tg = []
        for i, seed in enumerate(self.seeds):
            sr, tg = seed
            nega_sr += sr_nega[sr]
            nega_tg += tg_nega[tg]
        self.negative_data = [nega_sr, nega_tg]

    def init(self):
        nega_sample_num = self.nega_sample_num
        pos_sr = []
        pos_tg = []
        nega_sr = []
        nega_tg = []
        for sr, tg in self.seeds:
            can_srs = random.choices(range(0, self.num_sr - 1), k=nega_sample_num)
            can_tgs = random.choices(range(0, self.num_tg - 1), k=nega_sample_num)
            for can_sr, can_tg in zip(can_srs, can_tgs):
                if can_sr >= sr:
                    can_sr += 1
                if can_tg >= tg:
                    can_tg += 1
                nega_sr.append(can_sr)
                nega_tg.append(can_tg)
            pos_sr += [sr] * nega_sample_num
            pos_tg += [tg] * nega_sample_num
        self.positive_data = [pos_sr, pos_tg]
        self.negative_data = [nega_sr, nega_tg]

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

    def get_all(self):
        sr_all = torch.tensor(list(zip(self.positive_data[0], self.negative_data[0])), dtype=torch.int64)
        tg_all = torch.tensor(list(zip(self.positive_data[1], self.negative_data[1])), dtype=torch.int64)
        return sr_all, tg_all


if __name__ == '__main__':
    pass
