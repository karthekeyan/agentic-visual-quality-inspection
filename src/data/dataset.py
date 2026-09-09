"""Dataset for casting defect images.

Expects the folder layout shipped by the "Casting Product Image Data for
Quality Inspection" dataset:

    <root_dir>/
        ok_front/   *.jpeg   (label 0 -- "ok", per config.data.classes)
        def_front/  *.jpeg   (label 1 -- "defective")

`root_dir` should point at a single split, e.g. .../train or .../test.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

# Folder name -> class index. Order matches config.data.classes = [ok, defective].
CLASS_TO_FOLDER = {0: "ok_front", 1: "def_front"}


class CastingDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # list of (path: Path, label: int)

        for label, folder_name in CLASS_TO_FOLDER.items():
            class_dir = self.root_dir / folder_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected class folder not found: {class_dir}"
                )
            # Sorted for a deterministic sample order, which matters because
            # train/val splitting (src/data/split.py) relies on index
            # positions being stable across dataset instances built from
            # the same root_dir.
            for path in sorted(class_dir.iterdir()):
                if path.is_file():
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root_dir}")

    @property
    def targets(self):
        """Per-sample labels, e.g. for stratified splitting or sampler weights."""
        return [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
