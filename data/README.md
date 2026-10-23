# Oxford-IIIT Pet Dataset

This directory stores the local dataset and is intentionally excluded from Git.

## Recommended: download with Torchvision

The project notebook uses the following method, which downloads and extracts the dataset automatically:

```python
from torchvision.datasets import OxfordIIITPet

trainval_dataset = OxfordIIITPet(
    root="data",
    split="trainval",
    target_types=["category", "segmentation"],
    download=True,
)

test_dataset = OxfordIIITPet(
    root="data",
    split="test",
    target_types="segmentation",
    download=True,
)
```

Run this from the repository root. Torchvision will create `data/oxford-iiit-pet/` and reuse the downloaded files on later runs.

## Official download links

- Dataset page: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Images (approximately 755 MB): https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
- Annotations and trimaps (approximately 18.2 MB): https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

For manual installation, extract both archives under:

```text
data/oxford-iiit-pet/
├── images/
└── annotations/
```

Do not commit the downloaded archives, extracted images, or annotations to GitHub.
