"""Dataset and DataLoader utilities for casting defect images."""

from pathlib import Path

from torch.utils.data import Dataset


class CastingDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError
