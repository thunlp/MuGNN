import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AliagnmentDataset(Dataset):
    """Seed alignment dataset."""

    def __init__(self, seeds, nega_sample_num, num_sr, num_tg):
        assert nega_sample_num % 2 == 0
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.nega_sample_num = nega_sample_num
        self.seeds = [[int(sr), int(tg)] for sr, tg in seeds]

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, idx):
        sr, tg = self.seeds[idx]
        nega_sample_num = self.nega_sample_num
        nega_sr = []
        nega_tg = []
        for _ in range(nega_sample_num):
            can_sr = random.randint(0, self.num_sr-1)
            can_tg = random.randint(0, self.num_tg-1)
            if can_sr >= sr:
                can_sr += 1
            if can_tg >= tg:
                can_tg += 1
            nega_sr.append(can_sr)
            nega_tg.append(can_tg)
        sr_data = [sr] + nega_sr + [sr] * nega_sample_num
        tg_data = [tg] + [tg] * nega_sample_num + nega_tg
        # the first data is the positive one
        return torch.tensor(sr_data, dtype=torch.int32), torch.tensor(tg_data, dtype=torch.int32)


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