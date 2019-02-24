import random
from torch.utils.data import Dataset, DataLoader

class a(Dataset):
    def __init__(self):
        self.len = 10
    def __getitem__(self, idx):
        return random.randint(0,10)
    def __len__(self):
        return 10

b = DataLoader(a(), shuffle=False, num_workers=4)

for ele in b:
    print(ele, end='')
print('')
for ele in b:
    print(ele, end='')