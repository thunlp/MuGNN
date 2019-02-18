import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AliagnmentDataset(Dataset):
    """Seed alignment dataset."""

    def __init__(self, seeds, nega_sample_num, num_sr, num_tg):
        assert nega_sample_num % 2 == 0
        seeds = np.asarray([[int(sr), int(tg)] for sr, tg in seeds])
        self.positive_num = len(seeds)
        self.nega_sample_num = nega_sample_num
        self.seeds = np.tile(seeds, (nega_sample_num + 1, 1))
        self.num_sr = num_sr
        self.num_tg = num_tg
        # the index at which we start to replace tg for negative sampling
        self.s_index = (self.__len__() - self.positive_num) // 2 + self.positive_num

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, idx):
        sr, tg = self.seeds[idx]
        if idx < self.positive_num:
            return torch.tensor(sr, dtype=torch.int), torch.tensor(tg, dtype=torch.int), torch.tensor(self.nega_sample_num, dtype=torch.int)
        if idx < self.s_index:
            nega_sr = random.randint(0, self.num_sr)
            while nega_sr == sr:
                nega_sr = random.randint(0, self.num_sr)
            sr = nega_sr
        else:
            nega_tg = random.randint(0, self.num_tg)
            while nega_tg == tg:
                nega_tg = random.randint(0, self.num_tg)
            tg = nega_tg
        return torch.tensor(sr, dtype=torch.int), torch.tensor(tg, dtype=torch.int), torch.tensor(-1, dtype=torch.int)


if __name__ == '__main__':
    from project_path import bin_dir
    from data.reader import read_seeds, read_mapping
    directory = bin_dir / 'dbp15k' / 'fr_en'
    seeds = read_seeds(directory / 'entity_seeds.txt')
    entity_sr = read_mapping(directory / 'entity2id_fr.txt')
    entity_tg = read_mapping(directory / 'entity2id_en.txt')
    ad = AliagnmentDataset(seeds, 24,
                           len(entity_sr), len(entity_tg))
    ad_loader = DataLoader(ad, batch_size=640, shuffle=True, num_workers=4)
    iteriter = iter(ad_loader)
    i = 0
    while True:
        try:
            a = next(iteriter)
            print('1')
        except StopIteration:
            iteriter = iter(ad_loader)
            a = next(iteriter)
            print('2')
            input('')