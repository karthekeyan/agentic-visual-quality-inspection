"""Image transforms for the casting defect dataset.

Augmentation is intentionally light: casting images are captured from a fixed
camera rig, so aggressive geometric/color jitter would create samples that
don't resemble real inspection-line images. Horizontal flip and a small
rotation account for the part being placed in either orientation on the rig
without distorting defect shapes (e.g. cracks, blowholes) that the model
needs to learn to recognize.
"""

from torchvision import transforms

# Standard ImageNet statistics. We keep these (rather than computing
# dataset-specific mean/std) because the model is a pretrained ResNet18
# backbone (see config/config.yaml: model.pretrained) whose early layers
# expect inputs normalized the way they were during pretraining.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transforms(image_size):
    """Transform pipeline for training: resize + light augmentation + normalize.

    Args:
        image_size: (height, width) tuple, e.g. config["data"]["image_size"].
    """
    return transforms.Compose(
        [
            transforms.Resize(tuple(image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size):
    """Transform pipeline for validation/test: resize + normalize only.

    No augmentation — val/test must reflect the true data distribution so
    that metrics are representative of real inspection performance.
    """
    return transforms.Compose(
        [
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def denormalize(tensor):
    """Undo IMAGENET_MEAN/STD normalization for visualization (e.g. matplotlib).

    Returns a new tensor with values approximately in [0, 1]; does not
    mutate the input.
    """
    mean = tensor.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
