from torch.utils.data import Dataset, DataLoader

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, seeds, nega_sample_num, num_sr, num_tg):
        self.seeds = [(int(sr), int(tg)) for sr, tg in seeds]
        self.positive_num = len(self.seeds)
        self.num = self.positive_num * (nega_sample_num + 1)


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample